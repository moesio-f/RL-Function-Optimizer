# coding=utf-8
# Copyright 2020 The TF-Agents Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Taken from: https://github.com/tensorflow/agents/blob/v0.7.1/tf_agents/train/learner.py
# Commit c6ce6d6b73792c36d3fd33d109668b08fa12c784 | 7 Dec 2020

# Lint as: python3
"""Learner implementation for Agents. Refer to the examples dir."""

import gin

import tensorflow as tf
from tf_agents.replay_buffers.reverb_replay_buffer import ReverbReplayBuffer
from tf_agents.train.learner import Learner
from tf_agents.typing import types


@gin.configurable
class ReverbLearnerPER(Learner):
    def __init__(self,
                 root_dir,
                 train_step,
                 agent,
                 reverb_replay_buffer: ReverbReplayBuffer,
                 initial_is_weight_exp: types.Float = 0.5,
                 final_is_weight_exp: types.Float = 1.0,
                 is_weight_exp_steps: types.Int = 200000,
                 priority_clip_min: types.Float = 0.05,
                 priority_clip_max: types.Float = 100.0,
                 experience_dataset_fn=None,
                 triggers=None,
                 checkpoint_interval=100000,
                 summary_interval=1000,
                 max_checkpoints_to_keep=3,
                 use_kwargs_in_agent_train=False,
                 strategy=None):

        self._initial_is_weight_exp = tf.constant(initial_is_weight_exp, dtype=tf.float32)
        self._final_is_weight_exp = tf.constant(final_is_weight_exp, dtype=tf.float32)
        self._is_weight_exp_update = tf.constant((final_is_weight_exp - initial_is_weight_exp) / is_weight_exp_steps,
                                                 dtype=tf.float32)
        self._is_weight_exp = tf.Variable(self._initial_is_weight_exp, dtype=tf.float32, name='is_weight_exp')

        self._priority_clip_min = priority_clip_min
        self._priority_clip_max = priority_clip_max

        # Construir função de atualização das prioridades do dataset
        # SampleInfo('key', 'probability', 'table_size', 'priority')
        def update_priorities(sample, loss):
            (experience, sample_info) = sample
            td_errors = loss.extra.td_error_per_element
            priorities = tf.clip_by_value(tf.math.abs(td_errors),
                                          clip_value_min=priority_clip_min,
                                          clip_value_max=priority_clip_max)
            reverb_replay_buffer.update_priorities(sample_info.key[:, 0], tf.cast(priorities, dtype=tf.float64))

        update_priorities_fn = update_priorities

        super(ReverbLearnerPER, self).__init__(root_dir,
                                               train_step=train_step,
                                               agent=agent,
                                               experience_dataset_fn=experience_dataset_fn,
                                               after_train_strategy_step_fn=update_priorities_fn,
                                               triggers=triggers,
                                               checkpoint_interval=checkpoint_interval,
                                               summary_interval=summary_interval,
                                               max_checkpoints_to_keep=max_checkpoints_to_keep,
                                               use_kwargs_in_agent_train=use_kwargs_in_agent_train,
                                               strategy=strategy)

    def single_train_step(self, iterator):
        (experience, sample_info) = next(iterator)

        is_weights = self._get_is_weights(sample_info)

        if self.use_kwargs_in_agent_train:
            loss_info = self.strategy.run(self._agent.train, kwargs=dict(experience=experience,
                                                                         weights=is_weights))
            self.strategy.run(self.after_train_strategy_step_fn, kwargs=dict(experience=(experience, sample_info),
                                                                             loss_info=loss_info))
        else:
            loss_info = self.strategy.run(self._agent.train, args=(experience, is_weights))
            self.strategy.run(self.after_train_strategy_step_fn, args=((experience, sample_info), loss_info))

        return loss_info

    def _get_is_weights(self, sample_info):
        probs = tf.cast(sample_info.probability[:, 0], dtype=tf.float32)
        size = tf.cast(sample_info.table_size[:, 0], dtype=tf.float32)

        weights = tf.math.pow(tf.multiply(size, probs), tf.math.negative(self._is_weight_exp))
        weights = tf.math.divide(weights, tf.reduce_max(weights))

        self._is_weight_exp.assign(tf.clip_by_value(self._is_weight_exp + self._is_weight_exp_update,
                                                    clip_value_min=self._initial_is_weight_exp,
                                                    clip_value_max=self._final_is_weight_exp))

        return tf.cast(weights, dtype=tf.float32)

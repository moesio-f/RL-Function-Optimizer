"""Learner implementation for Reverb used as
a Prioritized Experience Replay (PER) Buffer."""

import gin

import tensorflow as tf
from tf_agents.replay_buffers import reverb_replay_buffer as reverb_rb
from tf_agents.train import learner
from tf_agents.typing import types


@gin.configurable
class ReverbLearnerPER(learner.Learner):
  """Reverb Learner class for Prioritized Replay Experience (PER)."""

  def __init__(self,
               root_dir,
               train_step,
               agent,
               reverb_replay_buffer: reverb_rb.ReverbReplayBuffer,
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

    self._initial_is_weight_exp = tf.constant(initial_is_weight_exp,
                                              dtype=tf.float32)
    self._final_is_weight_exp = tf.constant(final_is_weight_exp,
                                            dtype=tf.float32)
    self._is_weight_exp_update = tf.constant(
      (final_is_weight_exp - initial_is_weight_exp) / is_weight_exp_steps,
      dtype=tf.float32)
    self._is_weight_exp = tf.Variable(self._initial_is_weight_exp,
                                      dtype=tf.float32, name='is_weight_exp')

    self._priority_clip_min = priority_clip_min
    self._priority_clip_max = priority_clip_max

    # Construir função de atualização das prioridades do dataset
    # SampleInfo('key', 'probability', 'table_size', 'priority')
    def update_priorities(sample, loss):
      _, sample_info = sample
      td_errors = loss.extra.td_error_per_element
      priorities = tf.clip_by_value(tf.math.abs(td_errors),
                                    clip_value_min=priority_clip_min,
                                    clip_value_max=priority_clip_max)
      reverb_replay_buffer.update_priorities(sample_info.key[:, 0],
                                             tf.cast(priorities, tf.float64))

    update_priorities_fn = update_priorities

    super().__init__(root_dir,
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
      loss_info = self.strategy.run(self._agent.train,
                                    kwargs=dict(experience=experience,
                                                weights=is_weights))
      self.strategy.run(self.after_train_strategy_step_fn,
                        kwargs=dict(experience=(experience, sample_info),
                                    loss_info=loss_info))
    else:
      loss_info = self.strategy.run(self._agent.train,
                                    args=(experience, is_weights))
      self.strategy.run(self.after_train_strategy_step_fn,
                        args=((experience, sample_info), loss_info))

    return loss_info

  def _get_is_weights(self, sample_info):
    probs = tf.cast(sample_info.probability[:, 0], tf.float32)
    size = tf.cast(sample_info.table_size[:, 0], tf.float32)

    weights = tf.math.pow(tf.multiply(size, probs),
                          tf.math.negative(self._is_weight_exp))
    weights = tf.math.divide(weights, tf.reduce_max(weights))

    self._is_weight_exp.assign(
      tf.clip_by_value(self._is_weight_exp + self._is_weight_exp_update,
                       clip_value_min=self._initial_is_weight_exp,
                       clip_value_max=self._final_is_weight_exp))

    return tf.cast(weights, tf.float32)

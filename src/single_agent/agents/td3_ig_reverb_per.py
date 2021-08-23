"""Twin Delayed Deep Deterministic policy gradient (TD3) using Reverb
and PER (Prioritized Experience Replay)

Reverb: https://deepmind.com/research/open-source/Reverb
PER: https://arxiv.org/abs/1511.05952
TD3: https://arxiv.org/abs/1802.09477.
"""

import collections
from typing import Optional

import gin
import tensorflow as tf  # pylint: disable=g-explicit-tensorflow-version-import
import tensorflow_probability as tfp
from tf_agents.agents import tf_agent
from tf_agents.trajectories import time_step as ts
from tf_agents.typing import types
from tf_agents.utils import nest_utils
from tf_agents.utils import object_identity

from src.single_agent.agents import td3_inverting_gradients


class Td3ReverbInfo(
  collections.namedtuple('Td3ReverbInfo', ('actor_loss', 'critic_loss',
                                           'td_error_per_element',))):
  pass


@gin.configurable
class Td3AgentReverb(td3_inverting_gradients.Td3AgentInvertingGradients):
  """A TD3 Inverting Gradients Agent which uses PER by Reverb."""

  def _loss(self, experience: types.NestedTensor, weights: types.Tensor) -> \
        Optional[tf_agent.LossInfo]:
    raise NotImplementedError("Loss not implemented.")

  def _train(self, experience, weights=None):
    transition = self._as_transition(experience)
    time_steps, policy_steps, next_time_steps = transition
    actions = policy_steps.action

    trainable_critic_variables = list(object_identity.ObjectIdentitySet(
      self._critic_network_1.trainable_variables +
      self._critic_network_2.trainable_variables))
    with tf.GradientTape(watch_accessed_variables=False) as tape:
      assert trainable_critic_variables, ('No trainable critic variables to '
                                          'optimize.')
      tape.watch(trainable_critic_variables)
      critic_loss, td_errors_mean = self.critic_loss(time_steps, actions,
                                                     next_time_steps,
                                                     weights=weights,
                                                     training=True)
    tf.debugging.check_numerics(critic_loss, 'Critic loss is inf or nan.')
    critic_grads = tape.gradient(critic_loss, trainable_critic_variables)
    self._apply_gradients(critic_grads, trainable_critic_variables,
                          self._critic_optimizer)

    trainable_actor_variables = self._actor_network.trainable_variables
    with tf.GradientTape(watch_accessed_variables=False) as tape:
      assert trainable_actor_variables, ('No trainable actor variables to '
                                         'optimize.')
      tape.watch(trainable_actor_variables)
      actor_loss = self.actor_loss(time_steps, weights=weights, training=True)
    tf.debugging.check_numerics(actor_loss, 'Actor loss is inf or nan.')

    # We only optimize the actor every actor_update_period training steps.
    def optimize_actor():
      actor_grads = self.actor_grads(time_steps, weights=weights, training=True)
      return self._apply_gradients(actor_grads, trainable_actor_variables,
                                   self._actor_optimizer)

    remainder = tf.math.mod(self.train_step_counter, self._actor_update_period)
    tf.cond(pred=tf.equal(remainder, 0), true_fn=optimize_actor,
            false_fn=tf.no_op)

    self.train_step_counter.assign_add(1)
    self._update_target()

    total_loss = actor_loss + critic_loss

    return tf_agent.LossInfo(total_loss,
                             Td3ReverbInfo(actor_loss, critic_loss,
                                           td_errors_mean))

  def critic_loss(self,
                  time_steps: ts.TimeStep,
                  actions: types.Tensor,
                  next_time_steps: ts.TimeStep,
                  weights: Optional[types.Tensor] = None,
                  training: bool = False) -> (types.Tensor,):
    """Computes the critic loss for TD3 training.

    Args:
      time_steps: A batch of timesteps.
      actions: A batch of actions.
      next_time_steps: A batch of next timesteps.
      weights: Optional scalar or element-wise (per-batch-entry) importance
        weights.
      training: Whether this loss is being used for training.

    Returns:
      critic_loss: A scalar critic loss.
    """
    with tf.name_scope('critic_loss'):
      target_actions, _ = self._target_actor_network(
        next_time_steps.observation, next_time_steps.step_type,
        training=training)

      # Add gaussian noise to each action before computing target q values
      def add_noise_to_action(action):  # pylint: disable=missing-docstring
        dist = tfp.distributions.Normal(loc=tf.zeros_like(action),
                                        scale=self._target_policy_noise * \
                                              tf.ones_like(action))
        noise = dist.sample()
        noise = tf.clip_by_value(noise, -self._target_policy_noise_clip,
                                 self._target_policy_noise_clip)
        return action + noise

      noisy_target_actions = tf.nest.map_structure(add_noise_to_action,
                                                   target_actions)

      # Target q-values are the min of the two networks
      target_q_input_1 = (next_time_steps.observation, noisy_target_actions)
      target_q_values_1, _ = self._target_critic_network_1(
        target_q_input_1,
        next_time_steps.step_type,
        training=False)
      target_q_input_2 = (next_time_steps.observation, noisy_target_actions)
      target_q_values_2, _ = self._target_critic_network_2(
        target_q_input_2,
        next_time_steps.step_type,
        training=False)
      target_q_values = tf.minimum(target_q_values_1, target_q_values_2)

      td_targets = tf.stop_gradient(
        self._reward_scale_factor * next_time_steps.reward +
        self._gamma * next_time_steps.discount * target_q_values)

      pred_input_1 = (time_steps.observation, actions)
      pred_td_targets_1, _ = self._critic_network_1(
        pred_input_1, time_steps.step_type, training=training)
      pred_input_2 = (time_steps.observation, actions)
      pred_td_targets_2, _ = self._critic_network_2(
        pred_input_2, time_steps.step_type, training=training)
      pred_td_targets_all = [pred_td_targets_1, pred_td_targets_2]

      if self._debug_summaries:
        tf.compat.v2.summary.histogram(
          name='td_targets', data=td_targets, step=self.train_step_counter)
        with tf.name_scope('td_targets'):
          tf.compat.v2.summary.scalar(
            name='mean',
            data=tf.reduce_mean(input_tensor=td_targets),
            step=self.train_step_counter)
          tf.compat.v2.summary.scalar(
            name='max',
            data=tf.reduce_max(input_tensor=td_targets),
            step=self.train_step_counter)
          tf.compat.v2.summary.scalar(
            name='min',
            data=tf.reduce_min(input_tensor=td_targets),
            step=self.train_step_counter)

        for td_target_idx in range(2):
          pred_td_targets = pred_td_targets_all[td_target_idx]
          td_errors = td_targets - pred_td_targets
          with tf.name_scope('critic_net_%d' % (td_target_idx + 1)):
            tf.compat.v2.summary.histogram(
              name='td_errors', data=td_errors, step=self.train_step_counter)
            tf.compat.v2.summary.histogram(
              name='pred_td_targets',
              data=pred_td_targets,
              step=self.train_step_counter)
            with tf.name_scope('td_errors'):
              tf.compat.v2.summary.scalar(
                name='mean',
                data=tf.reduce_mean(input_tensor=td_errors),
                step=self.train_step_counter)
              tf.compat.v2.summary.scalar(
                name='mean_abs',
                data=tf.reduce_mean(input_tensor=tf.abs(td_errors)),
                step=self.train_step_counter)
              tf.compat.v2.summary.scalar(
                name='max',
                data=tf.reduce_max(input_tensor=td_errors),
                step=self.train_step_counter)
              tf.compat.v2.summary.scalar(
                name='min',
                data=tf.reduce_min(input_tensor=td_errors),
                step=self.train_step_counter)
            with tf.name_scope('pred_td_targets'):
              tf.compat.v2.summary.scalar(
                name='mean',
                data=tf.reduce_mean(input_tensor=pred_td_targets),
                step=self.train_step_counter)
              tf.compat.v2.summary.scalar(
                name='max',
                data=tf.reduce_max(input_tensor=pred_td_targets),
                step=self.train_step_counter)
              tf.compat.v2.summary.scalar(
                name='min',
                data=tf.reduce_min(input_tensor=pred_td_targets),
                step=self.train_step_counter)

      critic_loss = (self._td_errors_loss_fn(td_targets, pred_td_targets_1)
                     + self._td_errors_loss_fn(td_targets, pred_td_targets_2))
      if nest_utils.is_batched_nested_tensors(
            time_steps, self.time_step_spec, num_outer_dims=2):
        # Sum over the time dimension.
        critic_loss = tf.reduce_sum(
          input_tensor=critic_loss, axis=range(1, critic_loss.shape.rank))

      if weights is not None:
        critic_loss *= weights

      return tf.reduce_mean(input_tensor=critic_loss), \
             tf.stop_gradient(td_targets -
                              tf.math.divide(pred_td_targets_all[0] +
                                             pred_td_targets_all[1], 2.0))

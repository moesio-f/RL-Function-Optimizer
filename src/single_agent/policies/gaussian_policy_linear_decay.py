"""A policy that wraps a given policy and adds Gaussian noise with
linear scale decay."""

from typing import Optional, Text

import tensorflow as tf
import tensorflow_probability as tfp
from tf_agents.policies import tf_policy
from tf_agents.trajectories import policy_step
from tf_agents.trajectories import time_step as ts
from tf_agents.typing import types

from src.single_agent.policies import custom_gaussian_policy

tfd = tfp.distributions


class GaussianPolicyLinearDecay(custom_gaussian_policy.CustomGaussianPolicy):
  """Actor Policy with Gaussian exploration noise that linearly decays over
  time."""

  def __init__(self, wrapped_policy: tf_policy.TFPolicy,
               initial_scale: types.Float = 1.0,
               final_scale: types.Float = 0.1,
               num_steps: types.Int = 10000,
               clip: bool = True,
               name: Optional[Text] = None):
    """Builds a GaussianPolicyLinearDecay wrapping wrapped_policy.

        Args:
          wrapped_policy: A policy to wrap and add Gaussian noise to.
          initial_scale: Initial stddev of the Gaussian distribution
            from which noise is drawn. Default 1.0
          final_scale: Final stddev of the Gaussian distribution
            from which noise is drawn. Default 0.1
          num_steps: Number of iterations to reach the final_scale starting
            from initial_scale. Default 10000
          clip: Whether to clip actions to spec. Default True.
          name: The name of this policy.
        """
    if final_scale > initial_scale:
      raise ValueError("Final scale can't be greater than initial scale!")

    if num_steps <= 0:
      raise ValueError("Number of steps can't be non-positive!")

    super().__init__(wrapped_policy,
                     initial_scale,
                     clip,
                     name)
    self._scale_decay = tf.constant((initial_scale - final_scale) / num_steps,
                                    dtype=self._action_spec.dtype)
    self._initial_scale = tf.constant(initial_scale,
                                      dtype=self._action_spec.dtype)
    self._final_scale = tf.constant(final_scale, dtype=self._action_spec.dtype)
    self._current_scale = tf.Variable(self._initial_scale,
                                      dtype=self._action_spec.dtype)

  def _action(self, time_step: ts.TimeStep,
              policy_state: types.NestedTensor,
              seed: Optional[types.Seed] = None):
    seed_stream = tfp.util.SeedStream(seed=seed, salt='gaussian_noise')

    action_step = self._wrapped_policy.action(time_step, policy_state,
                                              seed_stream())

    def _add_noise(action, distribution):
      return action + distribution.sample(seed=seed_stream())

    actions = tf.nest.map_structure(_add_noise, action_step.action,
                                    self._noise_distribution)

    self._current_scale.assign(
      tf.clip_by_value(self._current_scale - self._scale_decay,
                       clip_value_min=self._final_scale,
                       clip_value_max=self._initial_scale))

    self._noise_distribution.scale = tf.ones(self._action_spec.shape,
                                             dtype=self._action_spec.dtype) \
                                     * self._current_scale

    return policy_step.PolicyStep(actions, action_step.state, action_step.info)

  def _distribution(self, time_step, policy_state):
    raise NotImplementedError('Distributions are not implemented yet.')

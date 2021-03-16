import tensorflow as tf
import tensorflow_probability as tfp

from tf_agents.policies import tf_policy
from tf_agents.trajectories import policy_step
from tf_agents.typing import types
from typing import Optional, Text
from tf_agents.policies.gaussian_policy import GaussianPolicy

tfd = tfp.distributions


class GaussianLinearScaleDecayPolicy(GaussianPolicy):
    def __init__(self, wrapped_policy: tf_policy.TFPolicy,
                 initial_scale: types.Float = 1.,
                 final_scale: types.Float = 0.1,
                 num_steps: types.Int = 10000,
                 clip: bool = True,
                 name: Optional[Text] = None):
        if final_scale > initial_scale:
            raise ValueError("Final scale can't be greater than initial scale!")

        if num_steps < 0:
            raise ValueError("Number of steps can't be negative!")

        super(GaussianLinearScaleDecayPolicy, self).__init__(wrapped_policy,
                                                             initial_scale,
                                                             clip,
                                                             name)
        self._scale_decay = tf.constant((initial_scale - final_scale) / num_steps,
                                        dtype=self._action_spec.dtype)
        self._initial_scale = tf.constant(initial_scale, dtype=self._action_spec.dtype)
        self._final_scale = tf.constant(final_scale, dtype=self._action_spec.dtype)

    def _action(self, time_step, policy_state, seed):
        seed_stream = tfp.util.SeedStream(seed=seed, salt='gaussian_noise')

        action_step = self._wrapped_policy.action(time_step, policy_state,
                                                  seed_stream())

        def _add_noise(action, distribution):
            return action + distribution.sample(seed=seed_stream())

        actions = tf.nest.map_structure(_add_noise, action_step.action,
                                        self._noise_distribution)

        def _create_normal_distribution(action_spec):
            scale = tf.clip_by_value(self._noise_distribution.scale[0] - self._scale_decay,
                                     clip_value_min=self._final_scale,
                                     clip_value_max=self._initial_scale)

            if tf.math.reduce_any(tf.math.less_equal(scale, self._final_scale)):
                return self._noise_distribution

            return tfd.Normal(
                loc=tf.zeros(action_spec.shape, dtype=action_spec.dtype),
                scale=tf.ones(action_spec.shape, dtype=action_spec.dtype) * scale)

        self._noise_distribution = tf.nest.map_structure(_create_normal_distribution, self._action_spec)

        return policy_step.PolicyStep(actions, action_step.state, action_step.info)

    def _distribution(self, time_step, policy_state):
        raise NotImplementedError('Distributions are not implemented yet.')

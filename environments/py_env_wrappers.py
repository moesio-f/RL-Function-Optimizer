"""Py environments wrappers."""

import numpy as np
from tf_agents.environments.py_environment import PyEnvironment
from tf_agents.environments.wrappers import PyEnvironmentBaseWrapper

__all__ = ['RewardClip', 'RewardScale']


class RewardClip(PyEnvironmentBaseWrapper):
  """Clips rewards in a given interval [a,b]."""
  def __init__(self, env: PyEnvironment, min_reward, max_reward):
    super().__init__(env)

    if min_reward > max_reward:
      raise ValueError("Minimum reward can't be greater than Maximum reward.")

    self._min_reward = np.array(min_reward, dtype=np.float32)
    self._max_reward = np.array(max_reward, dtype=np.float32)

  @property
  def min_reward(self):
    return self._min_reward

  @property
  def max_reward(self):
    return self._max_reward

  def _step(self, action):
    time_step = self._env.step(action)
    time_step = time_step._replace(reward=np.clip(time_step.reward,
                                                  a_min=self.min_reward,
                                                  a_max=self.max_reward))
    return time_step


class RewardScale(PyEnvironmentBaseWrapper):
  """Multiplies rewards by a scalar."""
  def __init__(self, env: PyEnvironment, scale_factor):
    super().__init__(env)

    self._scale_factor = np.array(scale_factor, dtype=np.float32)

  @property
  def scale_factor(self):
    return self._scale_factor

  def _step(self, action):
    time_step = self._env.step(action)
    time_step = time_step._replace(reward=self.scale_factor * time_step.reward)
    return time_step

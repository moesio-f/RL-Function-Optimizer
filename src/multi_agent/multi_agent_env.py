import numpy as np
from typing import Optional, Text, Any

from tf_agents.environments import py_environment
from tf_agents.typing import types
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts

import src.functions.base


class MultiAgentFunctionEnv(py_environment.PyEnvironment):
  """Multi-agent function environment."""

  def __init__(self,
               function: src.functions.base.Function,
               dims: int,
               n_agents: int,
               clip_actions: bool = False) -> None:
    super().__init__()

    self.func = function
    self.dims = dims
    self.n_agents = n_agents
    self.state_shape = (n_agents, dims)

    self._action_spec = array_spec.BoundedArraySpec(self.state_shape,
                                                    np.float32, -1.0, 1.0,
                                                    'action')
    self._reward_spec = array_spec.ArraySpec((n_agents,), np.float32, 'reward')

    self._observation_spec = array_spec.BoundedArraySpec(self.state_shape,
                                                         np.float32,
                                                         function.domain.min,
                                                         function.domain.max,
                                                         'observation')
    self._actor_obs_spec = array_spec.BoundedArraySpec((dims,), np.float32,
                                                       function.domain.min,
                                                       function.domain.max,
                                                       'actor_observation')

    if not clip_actions:
      self._action_spec = array_spec.ArraySpec.from_spec(self._action_spec)

  def actor_observation_spec(self):
    return self._actor_obs_spec

  def action_spec(self) -> types.NestedArraySpec:
    return self._action_spec

  def observation_spec(self) -> types.NestedArraySpec:
    return self._observation_spec

  def reward_spec(self) -> types.NestedArraySpec:
    return self._reward_spec

  def get_info(self) -> types.NestedArray:
    raise NotImplementedError('This environment has not implemented '
                              '`get_info()`.')

  def get_state(self) -> Any:
    raise NotImplementedError('This environment has not implemented '
                              '`get_state()`.')

  def set_state(self, state: Any) -> None:
    raise NotImplementedError('This environment has not implemented '
                              '`set_state()`.')

  def _initial_state(self) -> np.ndarray:
    domain_min, domain_max = self.func.domain
    state = np.random.uniform(domain_min, domain_max, self.state_shape)
    return state.astype(dtype=np.float32, copy=False)

  def _reset(self) -> ts.TimeStep:
    self.state = self._initial_state()
    return ts.restart(self.state)

  def _step(self, action: types.NestedArray) -> ts.TimeStep:
    domain_min, domain_max = self.func.domain
    self.state = self.state + action
    self.state = np.clip(self.state, domain_min, domain_max)
    reward = -self.func(self.state.T)
    return ts.transition(self.state, np.sum(reward))

  def render(self, mode: Text = 'rgb_array') -> Optional[types.NestedArray]:
    return super().render(mode=mode)

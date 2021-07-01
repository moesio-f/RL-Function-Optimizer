"""PyEnvironments that implement mathematical functions as environments."""

import collections
from typing import Optional, Text, Any

import numpy as np
from numpy.random import default_rng
from tf_agents.environments import py_environment
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts
from tf_agents.typing import types

import functions.base

MAX_STEPS = 50000


class FunctionEnvironmentInfo(
  collections.namedtuple('FunctionEnvironmentInfo',
                         ('position', 'objective_value'))):
  pass


class PyFunctionEnvironment(py_environment.PyEnvironment):
  """Single-agent function environment.
  Given a function f: D -> I, where D is a subset of R^d
  and I is a subset of R, the environment's specs are as follows:
    the observations (s in D) are positions in the domain and have shape (d);
    the actions (a in D) are the possible steps and have shape (d);
    the rewards are r = -f(s + a) and have shape (1);

  The environment resets after every 500000 iterations.
  Every new state (position), s + a, is in the domain (clipped when needed).
  Actions can be restricted: ai in [min, max].
  """
  def __init__(self, function: functions.base.Function, dims,
               clip_actions: bool = False) -> None:
    super().__init__()
    self._rng = default_rng()
    self.func = function
    # TODO(4a5463e): Corrigir possíveis erros no Drawer. (Erro para Griewank
    #  e SumSquares) self.drawer = FunctionDrawer(function)
    self._dims = dims

    self._episode_ended = False
    self._steps_taken = 0

    self._state = self.__initial_state()

    self._last_objective_value = self.func(self._state)
    self._last_position = self._state

    self._action_spec = array_spec.BoundedArraySpec(shape=(self._dims,),
                                                    dtype=np.float32,
                                                    minimum=-1.0,
                                                    maximum=1.0,
                                                    name='action')
    if not clip_actions:
      self._action_spec = array_spec.ArraySpec.from_spec(self._action_spec)

    self._observation_spec = array_spec.BoundedArraySpec(
      shape=(self._dims,),
      dtype=np.float32,
      minimum=function.domain.min,
      maximum=function.domain.max,
      name='observation')

  def action_spec(self):
    return self._action_spec

  def observation_spec(self):
    return self._observation_spec

  def get_info(self):
    return FunctionEnvironmentInfo(position=self._last_position,
                                   objective_value=self._last_objective_value)

  def get_state(self):
    state = (self._state, self._steps_taken, self._episode_ended)
    return state

  def set_state(self, state):
    _state, _steps_taken, _episode_ended = state
    self._state = _state
    self._steps_taken = _steps_taken
    self._episode_ended = _episode_ended

  def _step(self, action):
    if self._episode_ended:
      return self.reset()

    self._state = self._state + action
    domain_min, domain_max = self.func.domain
    self._state = np.clip(self._state, domain_min, domain_max)

    self._steps_taken += 1
    if self._steps_taken > MAX_STEPS:
      self._episode_ended = True

    obj_value = self.func(self._state)
    reward = -obj_value
    self._last_objective_value = obj_value
    self._last_position = self._state

    if self._episode_ended:
      return ts.termination(self._state, reward)

    return ts.transition(self._state, reward)

  def _reset(self):
    self._state = self.__initial_state()
    self._episode_ended = False
    self._steps_taken = 0
    self._last_objective_value = self.func(self._state)
    self._last_position = self._state
    return ts.restart(self._state)

  def render(self, mode: str = 'human'):
    # TODO(4a5463e): Revisar Drawer.
    """
    if self._steps_taken == 0:
        self.drawer.clear()
        self.drawer.draw_mesh(alpha=0.4, cmap='coolwarm')
        self.drawer.scatter(self._state[:2])
    self.drawer.update_scatter(self._state[:2])
    """
    raise NotImplementedError("Not Implemented yet.")

  def __initial_state(self) -> np.ndarray:
    domain_min, domain_max = self.func.domain
    state = self._rng.uniform(size=(self._dims,),
                              low=domain_min,
                              high=domain_max)
    return state.astype(dtype=np.float32, copy=False)


class MultiAgentFunctionEnv(py_environment.PyEnvironment):
  """Multi-agent function environment."""
  def __init__(self,
               function: functions.base.Function,
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

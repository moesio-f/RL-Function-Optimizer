"""Ambiente utilizado para a minimização de funções matemáticas com RL."""

import collections

import numpy as np
from numpy.random import default_rng
from tf_agents.environments import py_environment
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts
from tf_agents.typing import types

from src.functions import core

# Quantidade máxima de iterações entre o agente e ambiente.
MAX_STEPS = 50000


class PyFunctionEnv(py_environment.PyEnvironment):
  """Ambiente para a minimização de função.
  Dada uma função f: D -> I, onde D é um subconjunto de R^d
  e I é um subconjunto de R, as especificações do ambiente são:
    as observações (s em D) são posições do domínio;
    as ações (a em R^d) são os possíveis passos;
    as recompensas são r = -f(s + a).
  """

  def get_info(self) -> types.NestedArray:
    return self._state

  def __init__(self, function: core.Function, dims,
               bounded_actions_spec: bool = True):
    super().__init__()
    self._rng = default_rng()
    self.func = function
    self._dims = dims

    self._action_spec = array_spec.BoundedArraySpec(shape=(self._dims,),
                                                    dtype=np.float32,
                                                    minimum=-1.0,
                                                    maximum=1.0,
                                                    name='action')
    if not bounded_actions_spec:
      self._action_spec = array_spec.ArraySpec.from_spec(self._action_spec)

    self._observation_spec = array_spec.BoundedArraySpec(
      shape=(self._dims,),
      dtype=np.float32,
      minimum=function.domain.min,
      maximum=function.domain.max,
      name='observation')

    self._episode_ended = False
    self._steps_taken = 0

    self._state = self.__initial_state()

  def action_spec(self):
    return self._action_spec

  def observation_spec(self):
    return self._observation_spec

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
    if self._steps_taken >= MAX_STEPS:
      self._episode_ended = True

    obj_value = self.func(self._state)
    reward = -obj_value

    if self._episode_ended:
      return ts.termination(self._state, reward)

    return ts.transition(self._state, reward)

  def _reset(self):
    self._state = self.__initial_state()
    self._episode_ended = False
    self._steps_taken = 0
    return ts.restart(self._state)

  def render(self, mode: str = 'human'):
    raise NotImplementedError("Not Implemented yet.")

  def __initial_state(self) -> np.ndarray:
    domain_min, domain_max = self.func.domain
    state = self._rng.uniform(size=(self._dims,),
                              low=domain_min,
                              high=domain_max)
    return state.astype(dtype=np.float32, copy=False)

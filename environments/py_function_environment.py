from collections import namedtuple
from typing import List, Optional, Text
from utils.render import FunctionDrawer

import numpy as np
from functions.function import Function
from numpy.random import default_rng
from tf_agents.environments import py_environment
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts
from tf_agents.typing import types


class FunctionEnvironmentInfo(namedtuple('FunctionEnvironmentInfo', ('position', 'objective_value'))):
    pass


class PyFunctionEnvironment(py_environment.PyEnvironment):
    def __init__(self, function: Function, dims, clip_actions: bool = False) -> None:
        super().__init__()
        self._rng = default_rng()
        self.func = function
        self.drawer = FunctionDrawer(function)
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
        min, max = self.func.domain
        self._state = np.clip(self._state, min, max)

        self._steps_taken += 1
        if self._steps_taken > 500000:
            self._episode_ended = True

        obj_value = self.func(self._state)
        reward = -obj_value
        self._last_objective_value = obj_value
        self._last_position = self._state

        if self._episode_ended:
            return ts.termination(self._state, reward)
        else:
            return ts.transition(self._state, reward)

    def _reset(self):
        self._state = self.__initial_state()
        self._episode_ended = False
        self._steps_taken = 0
        self._last_objective_value = self.func(self._state)
        self._last_position = self._state
        return ts.restart(self._state)

    def render(self, mode: str = 'human'):
        if self._steps_taken == 0:
            self.drawer.clear()
            self.drawer.draw_mesh(alpha=0.4, cmap='coolwarm')
            self.drawer.scatter(self._state[:2])
        self.drawer.update_scatter(self._state[:2])
    
    def __initial_state(self) -> np.ndarray:
        min, max = self.func.domain
        state = self._rng.uniform(size=(self._dims,), low=min, high=max)
        return state.astype(dtype=np.float32, copy=False)


class MultiAgentFunctionEnv(py_environment.PyEnvironment):
    def __init__(self, 
                 function: Function,
                 dims: int,
                 n_agents: int, 
                 clip_actions: bool = False) -> None:
        super().__init__()

        self.func = function
        self.dims = dims
        self.n_agents = n_agents
        self.state_shape = (n_agents, dims)
        
        self._action_spec = array_spec.BoundedArraySpec(self.state_shape, np.float32, -1.0, 1.0, 'action')
        self._reward_spec = array_spec.ArraySpec((n_agents,), np.float32, 'reward')

        self._observation_spec = array_spec.BoundedArraySpec(self.state_shape, np.float32,
                function.domain.min, function.domain.max, 'observation')
        self._actor_obs_spec = array_spec.BoundedArraySpec((dims,), np.float32,
                function.domain.min, function.domain.max, 'actor_observation')
        
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
    
    def _initial_state(self) -> np.ndarray:
        min, max = self.func.domain
        state = np.random.uniform(min, max, self.state_shape)
        return state.astype(dtype=np.float32, copy=False)
    
    def _reset(self) -> ts.TimeStep:
        self.state = self._initial_state()
        return ts.restart(self.state)
    
    def _step(self, action: types.NestedArray) -> ts.TimeStep:
        min, max = self.func.domain
        self.state = self.state + action
        self.state = np.clip(self.state, min, max)
        reward = -self.func(self.state.T)
        return ts.transition(self.state, np.sum(reward))

    def render(self, mode: Text) -> Optional[types.NestedArray]:
        return super().render(mode=mode)
import numpy as np

from tf_agents.environments import py_environment
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts
from collections import namedtuple

hypercube = namedtuple('hypercube', ['min', 'max'])


class FunctionEnv(py_environment.PyEnvironment):
    def __init__(self, function, domain: hypercube, dims) -> None:
        super().__init__()
        self._function = function
        self._domain = domain
        self._dims = dims
        self._best_solution = np.finfo(np.float32).max
        self._episode_ended = False
        self._steps_taken = 0
        self._state = np.random.uniform(size=(dims,), low=domain.min, high=domain.max) \
            .astype(dtype=np.float32, copy=False)

        self._action_spec = array_spec.BoundedArraySpec(
            shape=(dims,), dtype=np.float32,
            minimum=-1.0, maximum=1.0, name='action')

        self._observation_spec = array_spec.BoundedArraySpec(
            shape=(dims,), dtype=np.float32,
            minimum=domain.min, maximum=domain.max, name='observation')

    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    def get_info(self):
        return self._best_solution

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
        self._state = np.clip(self._state, a_min=self._domain.min, a_max=self._domain.max)

        self._steps_taken += 1
        if self._steps_taken > 2000:
            self._episode_ended = True

        reward = -self._function(self._state)
        if reward < self._best_solution:
            self._best_solution = reward

        if self._episode_ended:
            return ts.termination(self._state, reward)
        else:
            return ts.transition(self._state, reward)

    def _reset(self):
        self._state = np.random.uniform(size=(self._dims,), low=self._domain.min, high=self._domain.max) \
            .astype(dtype=np.float32, copy=False)
        self._episode_ended = False
        self._steps_taken = 0
        return ts.restart(self._state)

    def _render(self):
        # TODO: Implementar método para renderizar
        pass
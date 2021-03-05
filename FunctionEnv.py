import numpy as np

from tf_agents.environments import py_environment
from tf_agents.specs import array_spec
from collections import namedtuple

hypercube = namedtuple('hypercube', ['min', 'max'])

class FunctionEnv(py_environment.PyEnvironment):
    def __init__(self, function, domain: hypercube(), dims) -> None:
        super().__init__()
        self._function = function
        self._domain = domain
        self._dims = dims
        self._state = np.random.uniform((dims,))
        
        self._action_spec = array_spec.ArraySpec(
            shape=(dims,), dtype=np.float32, name='action')

        self._observation_spec = array_spec.BoundedArraySpec(
            shape=(dims,), dtype=np.float32, 
            minimum=domain.min, maximum=domain.max, name='observation')

    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    def _step(self, action):
        pass

    def _reset(self):
        pass

    def render(self):
        pass
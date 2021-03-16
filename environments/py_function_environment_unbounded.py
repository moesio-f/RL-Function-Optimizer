import numpy as np

from tf_agents.specs import array_spec
from tf_agents.typing import types
from environments.py_function_environment import PyFunctionEnvironment


class PyFunctionEnvironmentUnbounded(PyFunctionEnvironment):
    def _set_action_spec(self) -> types.Spec:
        return array_spec.ArraySpec(shape=(self._dims,), dtype=np.float32, name='action')

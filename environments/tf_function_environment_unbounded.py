import tensorflow as tf
from tf_agents.specs.tensor_spec import TensorSpec
from tf_agents.typing import types
from environments.tf_function_environment import TFFunctionEnvironment


class TFFunctionEnvironmentUnbounded(TFFunctionEnvironment):
    def _set_action_spec(self) -> types.Spec:
        return TensorSpec(shape=(self._dims,), dtype=tf.float32, name='action')

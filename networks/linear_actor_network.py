"""Simple Actor network."""

import gin
import tensorflow as tf  # pylint: disable=g-explicit-tensorflow-version-import

from tf_agents.networks import network
from tf_agents.networks import utils


@gin.configurable
class LinearActorNetwork(network.Network):
  """Creates an actor network with linear activation in
  the final layer."""

  def __init__(self,
               input_tensor_spec,
               output_tensor_spec,
               fc_layer_params=None,
               dropout_layer_params=None,
               conv_layer_params=None,
               activation_fn=tf.keras.activations.relu,
               kernel_initializer=None,
               last_kernel_initializer=None,
               name='LinearActorNetwork'):
    """Creates an instance of `LinearActorNetwork`.

    Args:
      input_tensor_spec: A nest of `tensor_spec.TensorSpec` representing the
        inputs.
      output_tensor_spec: A nest of `tensor_spec.BoundedTensorSpec` representing
        the outputs.
      fc_layer_params: Optional list of fully_connected parameters, where each
        item is the number of units in the layer.
      dropout_layer_params: Optional list of dropout layer parameters, each item
        is the fraction of input units to drop or a dictionary of parameters
        according to the keras.Dropout documentation. The additional parameter
        `permanent`, if set to True, allows to apply dropout at inference for
        approximated Bayesian inference. The dropout layers are interleaved with
        the fully connected layers; there is a dropout layer after each fully
        connected layer, except if the entry in the list is None. This list must
        have the same length of fc_layer_params, or be None.
      conv_layer_params: Optional list of convolution layers parameters, where
        each item is a length-three tuple indicating (filters, kernel_size,
        stride).
      activation_fn: Activation function, e.g. tf.nn.relu, slim.leaky_relu, ...
      kernel_initializer: kernel initializer for all layers except for the value
        regression layer. If None, a VarianceScaling initializer will be used.
      last_kernel_initializer: kernel initializer for the value regression
         layer. If None, a RandomUniform initializer will be used.
      name: A string representing name of the network.

    Raises:
      ValueError: If `input_tensor_spec` or `action_spec` contains more than one
        item, or if the action data type is not `float`.
    """

    super().__init__(
      input_tensor_spec=input_tensor_spec,
      state_spec=(),
      name=name)

    if len(tf.nest.flatten(input_tensor_spec)) > 1:
      raise ValueError('Only a single observation is supported by this network')

    flat_action_spec = tf.nest.flatten(output_tensor_spec)
    if len(flat_action_spec) > 1:
      raise ValueError('Only a single action is supported by this network')
    self._single_action_spec = flat_action_spec[0]

    if self._single_action_spec.dtype not in [tf.float32, tf.float64]:
      raise ValueError('Only float actions are supported by this network.')

    if kernel_initializer is None:
      kernel_initializer = tf.compat.v1.keras.initializers.VarianceScaling(
        scale=1. / 3., mode='fan_in', distribution='uniform')
    if last_kernel_initializer is None:
      last_kernel_initializer = tf.keras.initializers.RandomUniform(
        minval=-0.003, maxval=0.003)

    self._mlp_layers = utils.mlp_layers(
      conv_layer_params,
      fc_layer_params,
      dropout_layer_params,
      activation_fn=activation_fn,
      kernel_initializer=kernel_initializer,
      name='input_mlp')

    self._mlp_layers.append(
      tf.keras.layers.Dense(
        flat_action_spec[0].shape.num_elements(),
        activation=tf.keras.activations.linear,
        kernel_initializer=last_kernel_initializer,
        name='action'))

    self._output_tensor_spec = output_tensor_spec

  def call(self, observations, step_type=(), network_state=(), training=False):
    del step_type  # unused.
    observations = tf.nest.flatten(observations)
    output = tf.cast(observations[0], tf.float32)
    for layer in self._mlp_layers:
      output = layer(output, training=training)

    actions = tf.cast(
      tf.reshape(output, [-1] + self._single_action_spec.shape.as_list()),
      self._single_action_spec.dtype)
    output_actions = tf.nest.pack_sequence_as(self._output_tensor_spec,
                                              [actions])

    return output_actions, network_state

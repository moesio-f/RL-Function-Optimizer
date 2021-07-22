"""The Normal (Gaussian) distribution class with
variable loc (mean) and scale (stddev)."""

import numpy as np
import tensorflow as tf
from tensorflow_probability.python.bijectors import \
  identity as identity_bijector
from tensorflow_probability.python.bijectors import \
  softplus as softplus_bijector
from tensorflow_probability.python.distributions import distribution
from tensorflow_probability.python.distributions import kullback_leibler
from tensorflow_probability.python.internal import assert_util
from tensorflow_probability.python.internal import dtype_util
from tensorflow_probability.python.internal import parameter_properties
from tensorflow_probability.python.internal import prefer_static as ps
from tensorflow_probability.python.internal import reparameterization
from tensorflow_probability.python.internal import samplers
from tensorflow_probability.python.internal import special_math
from tensorflow_probability.python.internal import tensor_util


class CustomNormal(distribution.Distribution):
  """The Normal distribution with location `loc` and `scale` parameters exposed
  as tf.Variables. """
  def __init__(self,
               loc,
               scale,
               validate_args=False,
               allow_nan_stats=True,
               name='Normal'):
    """Construct Normal distributions with mean and stddev `loc` and `scale`.

    The parameters `loc` and `scale` must be shaped in a way that supports
    broadcasting (e.g. `loc + scale` is a valid operation).

    Args:
      loc: Floating point tensor; the means of the distribution(s).
      scale: Floating point tensor; the stddevs of the distribution(s).
        Must contain only positive values.
      validate_args: Python `bool`, default `False`. When `True` distribution
        parameters are checked for validity despite possibly degrading runtime
        performance. When `False` invalid inputs may silently render incorrect
        outputs.
      allow_nan_stats: Python `bool`, default `True`. When `True`,
        statistics (e.g., mean, mode, variance) use the value "`NaN`" to
        indicate the result is undefined. When `False`, an exception is raised
        if one or more of the statistic's batch members are undefined.
      name: Python `str` name prefixed to Ops created by this class.

    Raises:
      TypeError: if `loc` and `scale` have different `dtype`.
    """
    parameters = dict(locals())
    with tf.name_scope(name) as name:
      dtype = dtype_util.common_dtype([loc, scale], dtype_hint=tf.float32)
      self._loc = tf.Variable(
        tensor_util.convert_nonref_to_tensor(loc, dtype=dtype, name='loc'),
        dtype=dtype, name='loc')
      self._scale = tf.Variable(
        tensor_util.convert_nonref_to_tensor(scale, dtype=dtype, name='scale'),
        dtype=dtype, name='scale')
      super().__init__(
        dtype=dtype,
        reparameterization_type=reparameterization.FULLY_REPARAMETERIZED,
        validate_args=validate_args,
        allow_nan_stats=allow_nan_stats,
        parameters=parameters,
        name=name)

  @classmethod
  def _parameter_properties(cls, dtype, num_classes=None):
    # pylint: disable=g-long-lambda
    return dict(
      loc=parameter_properties.ParameterProperties(),
      scale=parameter_properties.ParameterProperties(
        default_constraining_bijector_fn=(
          lambda: softplus_bijector.Softplus(low=dtype_util.eps(dtype)))))
    # pylint: enable=g-long-lambda

  @property
  def loc(self):
    """Distribution parameter for the mean."""
    return self._loc

  @loc.setter
  def loc(self, value):
    dtype = dtype_util.common_dtype([self.loc, self.scale, value],
                                    dtype_hint=tf.float32)
    new_loc = tensor_util.convert_nonref_to_tensor(value, dtype=dtype)
    self._loc.assign(new_loc)

  @property
  def scale(self):
    """Distribution parameter for standard deviation."""
    return self._scale

  @scale.setter
  def scale(self, value):
    dtype = dtype_util.common_dtype([self.scale, self.loc, value],
                                    dtype_hint=tf.float32)
    new_scale = tensor_util.convert_nonref_to_tensor(value, dtype=dtype)
    self._scale.assign(new_scale)

  def _batch_shape_tensor(self, loc=None, scale=None):
    return ps.broadcast_shape(
      ps.shape(self.loc if loc is None else loc),
      ps.shape(self.scale if scale is None else scale))

  def _batch_shape(self):
    return tf.broadcast_static_shape(self.loc.shape, self.scale.shape)

  def _event_shape_tensor(self):
    return tf.constant([], dtype=tf.int32)

  def _event_shape(self):
    return tf.TensorShape([])

  def _sample_n(self, n, seed=None):
    loc = tf.convert_to_tensor(self.loc)
    scale = tf.convert_to_tensor(self.scale)
    shape = ps.concat([[n], self._batch_shape_tensor(loc=loc, scale=scale)],
                      axis=0)
    sampled = samplers.normal(
      shape=shape, mean=0., stddev=1., dtype=self.dtype, seed=seed)
    return sampled * scale + loc

  def _log_prob(self, x):
    scale = tf.convert_to_tensor(self.scale)
    log_unnormalized = -0.5 * tf.math.squared_difference(
      x / scale, self.loc / scale)
    log_normalization = tf.constant(
      0.5 * np.log(2. * np.pi), dtype=self.dtype) + tf.math.log(scale)
    return log_unnormalized - log_normalization

  def _log_cdf(self, x):
    return special_math.log_ndtr(self._z(x))

  def _cdf(self, x):
    return special_math.ndtr(self._z(x))

  def _log_survival_function(self, x):
    return special_math.log_ndtr(-self._z(x))

  def _survival_function(self, x):
    return special_math.ndtr(-self._z(x))

  def _entropy(self):
    log_normalization = tf.constant(
      0.5 * np.log(2. * np.pi), dtype=self.dtype) + tf.math.log(self.scale)
    entropy = 0.5 + log_normalization
    return entropy * tf.ones_like(self.loc)

  def _mean(self):
    return self.loc * tf.ones_like(self.scale)

  def _quantile(self, p):
    return tf.math.ndtri(p) * self.scale + self.loc

  def _stddev(self):
    return self.scale * tf.ones_like(self.loc)

  _mode = _mean

  def _z(self, x, scale=None):
    """Standardize input `x` to a unit normal."""
    with tf.name_scope('standardize'):
      return (x - self.loc) / (self.scale if scale is None else scale)

  def _default_event_space_bijector(self):
    return identity_bijector.Identity(validate_args=self.validate_args)

  def _parameter_control_dependencies(self, is_init):
    assertions = []

    if is_init:
      try:
        self._batch_shape()
      except ValueError:
        raise ValueError(
          'Arguments `loc` and `scale` must have compatible shapes; '
          'loc.shape={}, scale.shape={}.'.format(
            self.loc.shape, self.scale.shape))
      # We don't bother checking the shapes in the dynamic case because
      # all member functions access both arguments anyway.

    if not self.validate_args:
      assert not assertions  # Should never happen.
      return []

    if is_init != tensor_util.is_ref(self.scale):
      assertions.append(assert_util.assert_positive(
        self.scale, message='Argument `scale` must be positive.'))

    return assertions

  def _variance(self, **kwargs):
    raise NotImplementedError('variance is not implemented: {}'.format(
      type(self).__name__))

  def _covariance(self, **kwargs):
    raise NotImplementedError('covariance is not implemented: {}'.format(
      type(self).__name__))


@kullback_leibler.RegisterKL(CustomNormal, CustomNormal)
def _kl_normal_normal(a, b, name=None):
  """Calculate the batched KL divergence KL(a || b) with a and b Normal.

  Args:
    a: instance of a Normal distribution object.
    b: instance of a Normal distribution object.
    name: Name to use for created operations.
      Default value: `None` (i.e., `'kl_normal_normal'`).

  Returns:
    kl_div: Batchwise KL(a || b)
  """
  with tf.name_scope(name or 'kl_normal_normal'):
    b_scale = tf.convert_to_tensor(b.scale)  # We'll read it thrice.
    diff_log_scale = tf.math.log(a.scale) - tf.math.log(b_scale)
    return (
          0.5 * tf.math.squared_difference(a.loc / b_scale, b.loc / b_scale) +
          0.5 * tf.math.expm1(2. * diff_log_scale) -
          diff_log_scale)

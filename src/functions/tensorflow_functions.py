"""TensorFlow implementation of many different functions."""

import numpy as np
import tensorflow as tf

from src.functions import core


class Ackley(core.Function):
  """Ackley function as defined in:
  https://www.sfu.ca/~ssurjano/ackley.html."""

  def __init__(self, domain: core.Domain = core.Domain(min=-32.768, max=32.768),
               a=20,
               b=0.2, c=2 * np.math.pi):
    super().__init__(domain)
    self._a = a
    self._b = b
    self._c = c

  def __call__(self, x: tf.Tensor):
    if x.dtype != tf.float32:
      x = tf.cast(x, tf.float32)

    d = x.shape[0]
    return -self.a * tf.exp(
      -self.b * tf.sqrt(tf.reduce_sum(x * x, axis=0) / d)) - \
           tf.exp(
             tf.reduce_sum(tf.cos(self.c * x), axis=0) / d) + self.a + np.math.e

  @property
  def a(self):
    return self._a

  @property
  def b(self):
    return self._b

  @property
  def c(self):
    return self._c


class Griewank(core.Function):
  """Griewank function as defined in:
  https://www.sfu.ca/~ssurjano/griewank.html."""

  def __init__(self, domain: core.Domain = core.Domain(min=-600.0, max=600.0)):
    super().__init__(domain)

  def __call__(self, x: tf.Tensor):
    if x.dtype != tf.float32:
      x = tf.cast(x, tf.float32)

    griewank_sum = tf.reduce_sum(x ** 2, axis=0) / 4000.0
    den = tf.range(1, x.shape[0] + 1, dtype=x.dtype)
    prod = tf.cos(x / tf.sqrt(den))
    prod = tf.reduce_prod(prod, axis=0)
    return griewank_sum - prod + 1


class Rastrigin(core.Function):
  def __init__(self, domain: core.Domain = core.Domain(min=-5.12, max=5.12)):
    super().__init__(domain)

  def __call__(self, x: tf.Tensor):
    if x.dtype != tf.float32:
      x = tf.cast(x, tf.float32)

    d = x.shape[0]
    return 10 * d + tf.reduce_sum(x ** 2 - 10 * tf.cos(x * 2 * np.math.pi),
                                  axis=0)


class Levy(core.Function):
  """Levy function as defined in:
  https://www.sfu.ca/~ssurjano/levy.html."""

  def __init__(self, domain: core.Domain = core.Domain(min=-10.0, max=10.0)):
    super().__init__(domain)

  def __call__(self, x: tf.Tensor):
    if x.dtype != tf.float32:
      x = tf.cast(x, tf.float32)

    pi = np.math.pi
    d = x.shape[0] - 1
    w = 1 + (x - 1) / 4

    term1 = tf.sin(pi * w[0]) ** 2
    term3 = (w[d] - 1) ** 2 * (1 + tf.sin(2 * pi * w[d]) ** 2)

    wi = w[0:d]
    levy_sum = tf.reduce_sum(
      (wi - 1) ** 2 * (1 + 10 * tf.sin(pi * wi + 1) ** 2), axis=0)
    return term1 + levy_sum + term3


class Rosenbrock(core.Function):
  """Rosenbrock function as defined in:
  https://www.sfu.ca/~ssurjano/rosen.html."""

  def __init__(self, domain: core.Domain = core.Domain(min=-5.0, max=10.0)):
    super().__init__(domain)

  def __call__(self, x: tf.Tensor):
    if x.dtype != tf.float32:
      x = tf.cast(x, tf.float32)

    xi = x[:-1]
    xnext = x[1:]
    return tf.reduce_sum(100 * (xnext - xi ** 2) ** 2 + (xi - 1) ** 2, axis=0)


class Zakharov(core.Function):
  """Zakharov function as defined in:
  https://www.sfu.ca/~ssurjano/zakharov.html."""

  def __init__(self, domain: core.Domain = core.Domain(min=-5.0, max=10.0)):
    super().__init__(domain)

  def __call__(self, x: tf.Tensor):
    if x.dtype != tf.float32:
      x = tf.cast(x, tf.float32)

    d = x.shape[0]

    sum1 = tf.reduce_sum(x * x, axis=0)
    sum2 = tf.reduce_sum(x * tf.range(1, (d + 1), dtype=x.dtype) / 2, axis=0)
    return sum1 + sum2 ** 2 + sum2 ** 4


class Bohachevsky(core.Function):
  """Bohachevsky function (f1, 2 dims only) as defined in:
  https://www.sfu.ca/~ssurjano/boha.html."""

  def __init__(self, domain: core.Domain = core.Domain(min=-100.0, max=100.0)):
    super().__init__(domain)

  def __call__(self, x: tf.Tensor):
    if x.dtype != tf.float32:
      x = tf.cast(x, tf.float32)

    d = x.shape[0]
    assert d == 2

    return tf.pow(x[0], 2) + tf.math.multiply(2, tf.pow(x[1], 2)) - \
           tf.math.multiply(0.3, tf.cos(3 * np.pi * x[0])) - \
           tf.math.multiply(0.4, tf.cos(4 * np.pi * x[1])) + 0.7


class SumSquares(core.Function):
  """SumSquares function as defined in:
  https://www.sfu.ca/~ssurjano/sumsqu.html."""

  def __init__(self, domain: core.Domain = core.Domain(min=-10.0, max=10.0)):
    super().__init__(domain)

  def __call__(self, x: tf.Tensor):
    if x.dtype != tf.float32:
      x = tf.cast(x, tf.float32)

    mul = tf.range(1, x.shape[0] + 1, dtype=x.dtype)
    return tf.reduce_sum((x ** 2) * mul, axis=0)


class Sphere(core.Function):
  """Sphere function as defined in:
  https://www.sfu.ca/~ssurjano/spheref.html."""

  def __init__(self, domain: core.Domain = core.Domain(min=-5.12, max=5.12)):
    super().__init__(domain)

  def __call__(self, x: tf.Tensor):
    if x.dtype != tf.float32:
      x = tf.cast(x, tf.float32)

    return tf.reduce_sum(x * x, axis=0)


class RotatedHyperEllipsoid(core.Function):
  """Rotated Hyper-Ellipsoid function as defined in:
  https://www.sfu.ca/~ssurjano/rothyp.html.
  TODO: Melhorar implementação."""

  def __init__(self,
               domain: core.Domain = core.Domain(min=-65.536, max=65.536)):
    super().__init__(domain)

  def __call__(self, x: tf.Tensor):
    if x.dtype != tf.float32:
      x = tf.cast(x, tf.float32)

    x = tf.cast(x, tf.float32)
    d = x.shape[0]

    return tf.reduce_sum(tf.convert_to_tensor(
      [tf.reduce_sum(x[0:(i + 1)] ** 2, axis=0) for i in range(d)],
      dtype=tf.float32), axis=0)


class DixonPrice(core.Function):
  """Dixon-Price function as defined in:
  https://www.sfu.ca/~ssurjano/dixonpr.html."""

  def __init__(self, domain: core.Domain = core.Domain(-10, 10)):
    super().__init__(domain)

  def __call__(self, x: tf.Tensor):
    if x.dtype != tf.float32:
      x = tf.cast(x, tf.float32)

    x0 = x[0]
    d = x.shape[0]
    term1 = (x0 - 1) ** 2
    ii = tf.range(2.0, d + 1, dtype=tf.float32)
    xi = x[1:]
    xold = x[:-1]
    dixon_sum = ii * (2 * xi ** 2 - xold) ** 2
    term2 = tf.reduce_sum(dixon_sum, axis=0)
    return term1 + term2


def list_all_functions() -> [core.Function]:
  return [Ackley(), Griewank(), Rastrigin(), Levy(), Rosenbrock(), Zakharov(),
          Bohachevsky(), SumSquares(), Sphere(), RotatedHyperEllipsoid(),
          DixonPrice()]


# Calcula os gradientes da função 'fun' na posição 'pos'.
# Só pode ser utilizada quando 'fun' é uma função implementada com
#   o TensorFlow.
def get_grads(fun: core.Function, pos: tf.Tensor):
  if pos.dtype != tf.float32:
    pos = tf.cast(pos, tf.float32)

  with tf.GradientTape() as tape:
    tape.watch(pos)
    y = fun(pos)

  return tape.gradient(y, pos), y

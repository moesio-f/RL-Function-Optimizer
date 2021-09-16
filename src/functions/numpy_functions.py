"""Implementação das diferentes funções de benchmark em NumPy."""

import numpy as np

from src.functions import core
from src.functions import tensorflow_functions as tff


class Ackley(core.Function):
  """Ackley function as defined in:
  https://www.sfu.ca/~ssurjano/ackley.html."""

  def __init__(self, domain: core.Domain = core.Domain(min=-32.768, max=32.768),
               a=20, b=0.2, c=2 * np.math.pi):
    super().__init__(domain)
    self._a = a
    self._b = b
    self._c = c

  def __call__(self, x: np.ndarray):
    if x.dtype != np.float32:
      x = x.astype(np.float32, casting='same_kind')

    d = x.shape[0]
    return -self.a * np.exp(-self.b * np.sqrt(np.sum(x * x, axis=0) / d)) - \
           np.exp(np.sum(np.cos(self.c * x), axis=0) / d) + self.a + np.math.e

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

  def __call__(self, x: np.ndarray):
    if x.dtype != np.float32:
      x = x.astype(np.float32, casting='same_kind')

    griewank_sum = np.sum(x ** 2, axis=0) / 4000.0
    den = np.arange(start=1, stop=(x.shape[0] + 1), dtype=x.dtype)
    prod = np.cos(x / np.sqrt(den))
    prod = np.prod(prod, axis=0)
    return griewank_sum - prod + 1


class Rastrigin(core.Function):
  """Rastrigin function as defined in:
  https://www.sfu.ca/~ssurjano/rastr.html."""

  def __init__(self, domain: core.Domain = core.Domain(min=-5.12, max=5.12)):
    super().__init__(domain)

  def __call__(self, x: np.ndarray):
    if x.dtype != np.float32:
      x = x.astype(np.float32, casting='same_kind')

    d = x.shape[0]
    return 10 * d + np.sum(x ** 2 - 10 * np.cos(x * 2 * np.math.pi), axis=0)


class Levy(core.Function):
  """Levy function as defined in:
  https://www.sfu.ca/~ssurjano/levy.html."""

  def __init__(self, domain: core.Domain = core.Domain(min=-10.0, max=10.0)):
    super().__init__(domain)

  def __call__(self, x: np.ndarray):
    if x.dtype != np.float32:
      x = x.astype(np.float32, casting='same_kind')

    pi = np.math.pi
    d = x.shape[0] - 1
    w = 1 + (x - 1) / 4

    term1 = np.sin(pi * w[0]) ** 2
    term3 = (w[d] - 1) ** 2 * (1 + np.sin(2 * pi * w[d]) ** 2)

    wi = w[0:d]
    levy_sum = np.sum((wi - 1) ** 2 * (1 + 10 * np.sin(pi * wi + 1) ** 2),
                      axis=0)
    return term1 + levy_sum + term3


class Rosenbrock(core.Function):
  """Rosenbrock function as defined in:
  https://www.sfu.ca/~ssurjano/rosen.html."""

  def __init__(self, domain: core.Domain = core.Domain(min=-5.0, max=10.0)):
    super().__init__(domain)

  def __call__(self, x: np.ndarray):
    if x.dtype != np.float32:
      x = x.astype(np.float32, casting='same_kind')

    xi = x[:-1]
    xnext = x[1:]
    return np.sum(100 * (xnext - xi ** 2) ** 2 + (xi - 1) ** 2, axis=0)


class Zakharov(core.Function):
  """Zakharov function as defined in:
  https://www.sfu.ca/~ssurjano/zakharov.html."""

  def __init__(self, domain: core.Domain = core.Domain(min=-5.0, max=10.0)):
    super().__init__(domain)

  def __call__(self, x: np.ndarray):
    if x.dtype != np.float32:
      x = x.astype(np.float32, casting='same_kind')

    d = x.shape[0]

    sum1 = np.sum(x * x, axis=0)
    sum2 = np.sum(x * np.arange(start=1, stop=(d + 1), dtype=x.dtype) / 2,
                  axis=0)
    return sum1 + sum2 ** 2 + sum2 ** 4


class Bohachevsky(core.Function):
  """Bohachevsky function (f1, 2 dims only) as defined in:
  https://www.sfu.ca/~ssurjano/boha.html."""

  def __init__(self, domain: core.Domain = core.Domain(min=-100.0, max=100.0)):
    super().__init__(domain)

  def __call__(self, x: np.ndarray):
    if x.dtype != np.float32:
      x = x.astype(np.float32, casting='same_kind')

    d = x.shape[0]
    assert d == 2

    return np.power(x[0], 2) + 2 * np.power(x[1], 2) - \
           0.3 * np.cos(3 * np.pi * x[0]) - \
           0.4 * np.cos(4 * np.pi * x[1]) + 0.7


class SumSquares(core.Function):
  """SumSquares function as defined in:
  https://www.sfu.ca/~ssurjano/sumsqu.html."""

  def __init__(self, domain: core.Domain = core.Domain(min=-10.0, max=10.0)):
    super().__init__(domain)

  def __call__(self, x: np.ndarray):
    if x.dtype != np.float32:
      x = x.astype(np.float32, casting='same_kind')

    d = x.shape[0]
    mul = np.arange(start=1, stop=(d + 1), dtype=x.dtype)
    return np.sum((x ** 2) * mul, axis=0)


class Sphere(core.Function):
  """Sphere function as defined in:
  https://www.sfu.ca/~ssurjano/spheref.html."""

  def __init__(self, domain: core.Domain = core.Domain(min=-5.12, max=5.12)):
    super().__init__(domain)

  def __call__(self, x: np.ndarray):
    if x.dtype != np.float32:
      x = x.astype(np.float32, casting='same_kind')

    return np.sum(x * x, axis=0)


class RotatedHyperEllipsoid(core.Function):
  """Rotated Hyper-Ellipsoid function as defined in:
  https://www.sfu.ca/~ssurjano/rothyp.html.
  TODO: Melhorar implementação."""

  def __init__(self,
               domain: core.Domain = core.Domain(min=-65.536, max=65.536)):
    super().__init__(domain)

  def __call__(self, x: np.ndarray):
    if x.dtype != np.float32:
      x = x.astype(np.float32, casting='same_kind')

    d = x.shape[0]

    return np.sum([np.sum(x[0:(i + 1)] ** 2, axis=0) for i in range(d)],
                  dtype=np.float32, axis=0)


class DixonPrice(core.Function):
  """Dixon-Price function as defined in:
  https://www.sfu.ca/~ssurjano/dixonpr.html."""

  def __init__(self, domain: core.Domain = core.Domain(-10, 10)):
    super().__init__(domain)

  def __call__(self, x: np.ndarray):
    if x.dtype != np.float32:
      x = x.astype(np.float32, casting='same_kind')

    x0 = x[0]
    d = x.shape[0]
    term1 = (x0 - 1) ** 2
    ii = np.arange(2.0, d + 1)
    xi = x[1:]
    xold = x[:-1]
    dixon_sum = ii * (2 * xi ** 2 - xold) ** 2
    term2 = np.sum(dixon_sum, axis=0)
    return term1 + term2


# Função utilitária para obter uma função equivalente de Numpy em Tensorflow.
def get_tf_function(function: core.Function):
  domain = function.domain
  f = getattr(tff, function.name)
  return f(domain)


def list_all_functions() -> [core.Function]:
  return [Ackley(), Griewank(), Rastrigin(), Levy(), Rosenbrock(), Zakharov(),
          Bohachevsky(), SumSquares(), Sphere(), RotatedHyperEllipsoid(),
          DixonPrice()]

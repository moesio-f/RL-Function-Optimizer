import tensorflow as tf
import numpy as np
from functions.function import Function, Domain


class Ackley(Function):
    def __init__(self, domain: Domain = Domain(min=-32.768, max=32.768), a=20, b=0.2, c=2 * np.math.pi):
        super().__init__(domain)
        self._a = a
        self._b = b
        self._c = c

    def __call__(self, x: tf.Tensor):
        if x.dtype != tf.float32:
            x = tf.cast(x, dtype=tf.float32)

        d = x.shape[0]
        return -self.a * tf.exp(-self.b * tf.sqrt(tf.reduce_sum(x * x, axis=0) / d)) - \
               tf.exp(tf.reduce_sum(tf.cos(self.c * x), axis=0) / d) + self.a + np.math.e

    @property
    def a(self):
        return self._a

    @property
    def b(self):
        return self._b

    @property
    def c(self):
        return self._c


class Griewank(Function):
    def __init__(self, domain: Domain = Domain(min=-600.0, max=600.0)):
        super().__init__(domain)

    def __call__(self, x: tf.Tensor):
        if x.dtype != tf.float32:
            x = tf.cast(x, dtype=tf.float32)

        griewank_sum = tf.reduce_sum(x ** 2, axis=0) / 4000.0
        den = tf.range(1, x.shape[0] + 1, dtype=x.dtype)
        prod = tf.cos(x / tf.sqrt(den))
        prod = tf.reduce_prod(prod, axis=0)
        return griewank_sum - prod + 1


class Rastrigin(Function):
    def __init__(self, domain: Domain = Domain(min=-5.12, max=5.12)):
        super().__init__(domain)

    def __call__(self, x: tf.Tensor):
        if x.dtype != tf.float32:
            x = tf.cast(x, dtype=tf.float32)

        d = x.shape[0]
        return 10 * d + tf.reduce_sum(x ** 2 - 10 * tf.cos(x * 2 * np.math.pi), axis=0)


class Levy(Function):
    def __init__(self, domain: Domain = Domain(min=-10.0, max=10.0)):
        super().__init__(domain)

    def __call__(self, x: tf.Tensor):
        if x.dtype != tf.float32:
            x = tf.cast(x, dtype=tf.float32)

        pi = np.math.pi
        d = x.shape[0] - 1
        w = 1 + (x - 1) / 4

        term1 = tf.sin(pi * w[0]) ** 2
        term3 = (w[d] - 1) ** 2 * (1 + tf.sin(2 * pi * w[d]) ** 2)

        wi = w[0:d]
        levy_sum = tf.reduce_sum((wi - 1) ** 2 * (1 + 10 * tf.sin(pi * wi + 1) ** 2), axis=0)
        return term1 + levy_sum + term3


class Rosenbrock(Function):
    def __init__(self, domain: Domain = Domain(min=-5.0, max=10.0)):
        super().__init__(domain)

    def __call__(self, x: tf.Tensor):
        if x.dtype != tf.float32:
            x = tf.cast(x, dtype=tf.float32)

        rosen_sum = 0.0
        d = x.shape[0]

        for i in range(d - 1):
            rosen_sum += 100 * (x[i + 1] - x[i] ** 2) ** 2 + (x[i] - 1.0) ** 2

        return rosen_sum


class Zakharov(Function):
    def __init__(self, domain: Domain = Domain(min=-5.0, max=10.0)):
        super().__init__(domain)

    def __call__(self, x: tf.Tensor):
        if x.dtype != tf.float32:
            x = tf.cast(x, dtype=tf.float32)

        d = x.shape[0]

        sum1 = tf.reduce_sum(x * x, axis=0)
        sum2 = tf.reduce_sum(x * tf.range(1, (d + 1), dtype=x.dtype) / 2, axis=0)
        return sum1 + sum2 ** 2 + sum2 ** 4


class PowerSum(Function):
    def __init__(self, domain: Domain = Domain(min=0.0, max=5.0), b=tf.constant([8, 18, 44, 114], dtype=tf.float32)):
        super().__init__(domain)
        self._b = b

    def __call__(self, x: tf.Tensor):
        if x.dtype != tf.float32:
            x = tf.cast(x, dtype=tf.float32)

        d = x.shape[0]
        bd = self.b.shape[0]

        return tf.reduce_sum(tf.pow(tf.convert_to_tensor([tf.reduce_sum(tf.pow(x, (i + 1))) - self.b[i % bd]
                                                          for i in range(d)], dtype=tf.float32), tf.constant(2.0)), axis=0)

    @property
    def b(self):
        return self._b


class Bohachevsky(Function):
    def __init__(self, domain: Domain = Domain(min=-100.0, max=100.0)):
        super().__init__(domain)

    def __call__(self, x: tf.Tensor):
        if x.dtype != tf.float32:
            x = tf.cast(x, dtype=tf.float32)

        d = x.shape[0]
        assert d == 2

        return x[0] ** 2 + 2 * (x[1] ** 2) - 0.3 * tf.cos(3 * np.pi * x[0]) - 0.4 * tf.cos(4 * np.pi * x[1]) + 0.7


class SumSquares(Function):
    def __init__(self, domain: Domain = Domain(min=-10.0, max=10.0)):
        super().__init__(domain)

    def __call__(self, x: tf.Tensor):
        if x.dtype != tf.float32:
            x = tf.cast(x, dtype=tf.float32)

        mul = tf.range(1, x.shape[0], dtype=x.dtype)
        return tf.reduce_sum((x ** 2) * mul, axis=0)


class Sphere(Function):
    def __init__(self, domain: Domain = Domain(min=-5.12, max=5.12)):
        super().__init__(domain)

    def __call__(self, x: tf.Tensor):
        if x.dtype != tf.float32:
            x = tf.cast(x, dtype=tf.float32)

        return tf.reduce_sum(x * x, axis=0)


class RotatedHyperEllipsoid(Function):
    def __init__(self, domain: Domain = Domain(min=-65.536, max=65.536)):
        super().__init__(domain)

    def __call__(self, x: tf.Tensor):
        if x.dtype != tf.float32:
            x = tf.cast(x, dtype=tf.float32)

        x = tf.cast(x, dtype=tf.float32)
        d = x.shape[0]

        return tf.reduce_sum(tf.convert_to_tensor([tf.reduce_sum(x[0:(i + 1)] ** 2, axis=0) for i in range(d)],
                                                  dtype=tf.float32), axis=0)

class DixonPrice(Function):
    def __init__(self, domain: Domain = Domain(-10, 10)):
        super().__init__(domain)
    
    def __call__(self, x: tf.Tensor):
        x0 = x[0]
        d = x.shape[0]
        term1 = (x0 - 1)**2
        ii = tf.range(2.0, d+1)
        xi = x[1:]
        xold = x[:-1]
        sum = ii * (2*xi**2 - xold) ** 2
        term2 = tf.reduce_sum(sum, axis=0)
        return term1 + term2
        """ 
        sum1 = (x[0] - 1) ** 2
        sum2 = 0
        for i in range(1, x.shape[0]):
            sum2 += i * (2 * x[i]**2 - x[i -1]) ** 2
        return sum1 + sum2 """


def list_all_functions() -> [Function]:
    return [Ackley(), Griewank(), Rastrigin(), Levy(), Rosenbrock(), Zakharov(),
            PowerSum(), Bohachevsky(), SumSquares(), Sphere(), RotatedHyperEllipsoid(), DixonPrice()]

import numpy as np
from functions.function import Function, Domain


class Ackley(Function):
    def __init__(self, domain: Domain = Domain(min=-32.768, max=32.768), a=20, b=0.2, c=2 * np.math.pi):
        super().__init__(domain)
        self._a = a
        self._b = b
        self._c = c

    def __call__(self, x: np.ndarray, *args, **kwargs):
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


class Griewank(Function):
    def __init__(self, domain: Domain = Domain(min=-600.0, max=600.0)):
        super().__init__(domain)

    def __call__(self, x: np.ndarray, *args, **kwargs):
        if x.dtype != np.float32:
            x = x.astype(np.float32, casting='same_kind')

        griewank_sum = np.sum(x ** 2, axis=0) / 4000.0
        den = np.arange(start=1, stop=(x.shape[0] + 1), dtype=x.dtype)
        prod = np.cos(x / np.sqrt(den))
        prod = np.prod(prod, axis=0)
        return griewank_sum - prod + 1


class Rastrigin(Function):
    def __init__(self, domain: Domain = Domain(min=-5.12, max=5.12)):
        super().__init__(domain)

    def __call__(self, x: np.ndarray, *args, **kwargs):
        if x.dtype != np.float32:
            x = x.astype(np.float32, casting='same_kind')

        d = x.shape[0]
        return 10 * d + np.sum(x ** 2 - 10 * np.cos(x * 2 * np.math.pi), axis=0)


class Levy(Function):
    def __init__(self, domain: Domain = Domain(min=-10.0, max=10.0)):
        super().__init__(domain)

    def __call__(self, x: np.ndarray, *args, **kwargs):
        if x.dtype != np.float32:
            x = x.astype(np.float32, casting='same_kind')

        pi = np.math.pi
        d = x.shape[0] - 1
        w = 1 + (x - 1) / 4

        term1 = np.sin(pi * w[0]) ** 2
        term3 = (w[d] - 1) ** 2 * (1 + np.sin(2 * pi * w[d]) ** 2)

        wi = w[0:d]
        levy_sum = np.sum((wi - 1) ** 2 * (1 + 10 * np.sin(pi * wi + 1) ** 2), axis=0)
        return term1 + levy_sum + term3


class Rosenbrock(Function):
    def __init__(self, domain: Domain = Domain(min=-5.0, max=10.0)):
        super().__init__(domain)

    def __call__(self, x, *args, **kwargs):
        if x.dtype != np.float32:
            x = x.astype(np.float32, casting='same_kind')

        rosen_sum = 0.0
        d = x.shape[0]

        for i in range(d - 1):
            rosen_sum += 100 * (x[i + 1] - x[i] ** 2) ** 2 + (x[i] - 1.0) ** 2

        return rosen_sum


class Zakharov(Function):
    def __init__(self, domain: Domain = Domain(min=-5.0, max=10.0)):
        super().__init__(domain)

    def __call__(self, x, *args, **kwargs):
        if x.dtype != np.float32:
            x = x.astype(np.float32, casting='same_kind')

        d = x.shape[0]

        sum1 = np.sum(x * x, axis=0)
        sum2 = np.sum(x * np.arange(start=1, stop=(d + 1), dtype=x.dtype) / 2, axis=0)
        return sum1 + sum2 ** 2 + sum2 ** 4


class Bohachevsky(Function):
    def __init__(self, domain: Domain = Domain(min=-100.0, max=100.0)):
        super().__init__(domain)

    def __call__(self, x, *args, **kwargs):
        if x.dtype != np.float32:
            x = x.astype(np.float32, casting='same_kind')

        d = x.shape[0]
        assert d == 2

        return x[0] ** 2 + 2 * (x[1] ** 2) - 0.3 * np.cos(3 * np.pi * x[0]) - 0.4 * np.cos(4 * np.pi * x[1]) + 0.7


class SumSquares(Function):
    def __init__(self, domain: Domain = Domain(min=-10.0, max=10.0)):
        super().__init__(domain)

    def __call__(self, x, *args, **kwargs):
        if x.dtype != np.float32:
            x = x.astype(np.float32, casting='same_kind')

        d = x.shape[0]
        mul = np.arange(start=1, stop=(d + 1), dtype=x.dtype)
        return np.sum((x ** 2) * mul, axis=0)


class Sphere(Function):
    def __init__(self, domain: Domain = Domain(min=-5.12, max=5.12)):
        super().__init__(domain)

    def __call__(self, x, *args, **kwargs):
        if x.dtype != np.float32:
            x = x.astype(np.float32, casting='same_kind')

        return np.sum(x * x, axis=0)


class RotatedHyperEllipsoid(Function):
    def __init__(self, domain: Domain = Domain(min=-65.536, max=65.536)):
        super().__init__(domain)

    def __call__(self, x, *args, **kwargs):
        if x.dtype != np.float32:
            x = x.astype(np.float32, casting='same_kind')

        d = x.shape[0]

        return np.sum([np.sum(x[0:(i + 1)] ** 2, axis=0) for i in range(d)], dtype=np.float32, axis=0)


def list_all_functions() -> [Function]:
    return [Ackley(), Griewank(), Rastrigin(), Levy(), Rosenbrock(), Zakharov(),
            Bohachevsky(), SumSquares(), Sphere(), RotatedHyperEllipsoid()]

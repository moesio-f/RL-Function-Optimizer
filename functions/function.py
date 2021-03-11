import abc
from collections import namedtuple
Domain = namedtuple('hypercube', ['min', 'max'])


class Function(object):
    def __init__(self, domain: Domain):
        assert domain is not None
        self._domain = domain

    @abc.abstractmethod
    def __call__(self, x, *args, **kwargs):
        pass

    @property
    def domain(self) -> Domain:
        return self._domain

    @domain.setter
    def domain(self, new_domain: Domain):
        self._domain = new_domain

    @property
    def name(self):
        return self.__str__()

    def __str__(self):
        return self.__class__.__name__

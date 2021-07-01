"""Utility methods for hyperparameters choices."""

import numpy as np
from numpy.random import default_rng

from functions.base import Function


def function_range_estimation(function: Function, dims: int,
                              num_random_samples: int = 300000):
  rng = default_rng()

  value_at_min = function(
    np.repeat(function.domain.min, dims).astype(np.float32))
  value_at_max = function(
    np.repeat(function.domain.max, dims).astype(np.float32))

  values = []
  for _ in range(num_random_samples):
    values.append(function(rng.uniform(size=(dims,),
                                       low=function.domain.min,
                                       high=function.domain.max).astype(
      np.float32)))

  max_value = np.max(values)
  min_value = np.min(values)
  mean_value = np.mean(values)

  print('======= Information about function range =======')
  print('Objective value at [{0}, ..., {0}]: {1}'.format(function.domain.min,
                                                         value_at_min))
  print('Objective value at [{0}, ..., {0}]: {1}'.format(function.domain.max,
                                                         value_at_max))
  print(
    '------ Summary after {0} uniform random sampled positions ------'.format(
      num_random_samples))
  print('Maximum value:', max_value)
  print('Minimum value:', min_value)
  print('Mean value:', mean_value)
  print('=================================================')

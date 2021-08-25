"""Utility functions/classes for the functions module."""

import numpy as np

from src.functions import numpy_functions as functions_np
from src.functions import tensorflow_functions as functions_tf
from src.functions import core


def get_common_function(name, in_tensorflow=False) -> core.Function:

  if in_tensorflow:
    return next(
      (f for f in functions_tf.list_all_functions() if f.name == name), None)
  else:
    return next(
      (f for f in functions_np.list_all_functions() if f.name == name), None)


def test_functions():
  for func in functions_np.list_all_functions():
    min, max = func.domain
    pos = np.random.uniform(min, max, (2, 10))
    result = func(pos)
    expected_shape = (10,)
    if result.shape != expected_shape:
      raise ValueError(
        f'Function {func.name} is broken! '
        f'result shape is {result.shape} '
        f'instead of: {expected_shape}')
    else:
      print(f'Function {func.name}: Passed test!')

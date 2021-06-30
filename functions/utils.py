"""Utility functions/classes for the functions module."""

import functions.numpy_functions as functions_np
import functions.tensorflow_functions as functions_tf
from functions.function import Function


def get_common_function(name, in_tensorflow=False) -> Function:
  if in_tensorflow:
    function = next(
      (f for f in functions_tf.list_all_functions() if f.name == name), None)
  else:
    function = next(
      (f for f in functions_np.list_all_functions() if f.name == name), None)

  return function

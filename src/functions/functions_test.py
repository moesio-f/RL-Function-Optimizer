"""Function validation tests."""

import tensorflow as tf
import numpy as np
import unittest

from src.functions import numpy_functions as npf
from src.functions import tensorflow_functions as tff


class TestNumpyFunctions(unittest.TestCase):
  
  def setUp(self) -> None:
    self.array = np.array([1,2,3,4], dtype=np.float64)
    self.zero = np.array([0,0,0,0], dtype=np.float64)

  def tearDown(self) -> None:
    del self.array
    del self.zero

  # check dtypes between function's input and output
  def check_dtypes(self, function, input, output):
    self.assertEqual(input.dtype, output.dtype,
      f"{function.name} output is {output.dtype} when should be {input.dtype}.")

  def test_ackley(self):
    f = npf.Ackley()
    result = f(self.array)
    self.assertEqual(result, 8.43469444443746497)

    result = f(self.zero)
    self.assertEqual(result, 4.44089209850062616e-16)
    self.check_dtypes(f, self.array, result)
  
  def test_griewank(self):
    f = npf.Griewank()
    result = f(self.array)
    self.assertEqual(result, 1.00187037800320189)

    result = f(self.zero)
    self.assertEqual(result, 0.0)
    self.check_dtypes(f, self.array, result)
  
  def test_rastrigin(self):
    f = npf.Rastrigin()
    result = f(self.array)
    self.assertEqual(result, 30.0)

    result = f(self.zero)
    self.assertEqual(result, 0.0)
    self.check_dtypes(f, self.array, result)

  def test_levy(self):
    f = npf.Levy()
    result = f(self.array)
    self.assertEqual(result, 2.76397190019909811)

    result = f(self.zero)
    self.assertEqual(result, 0.897533662350923467)
    self.check_dtypes(f, self.array, result)

  def test_rosenbrock(self):
    f = npf.Rosenbrock()
    result = f(self.array)
    self.assertEqual(result, 2705.0)

    result = f(self.zero)
    self.assertEqual(result, 3.0)
    self.check_dtypes(f, self.array, result)

  def test_zakharov(self):
    f = npf.Zakharov()
    result = f(self.array)
    self.assertEqual(result, 50880.0)

    result = f(self.zero)
    self.assertEqual(result, 0.0)
    self.check_dtypes(f, self.array, result)

  def test_sum_squares(self):
    f = npf.SumSquares()
    result = f(self.array)
    self.assertEqual(result, 100.0)

    result = f(self.zero)
    self.assertEqual(result, 0.0)
    self.check_dtypes(f, self.array, result)
  
  def test_sphere(self):
    f = npf.Sphere()
    result = f(self.array)
    self.assertEqual(result, 30.0)

    result = f(self.zero)
    self.assertEqual(result, 0.0)
    self.check_dtypes(f, self.array, result)

  def test_rotated_hyper_ellipsoid(self):
    f = npf.RotatedHyperEllipsoid()
    result = f(self.array)
    self.assertEqual(result, 50.0)

    result = f(self.zero)
    self.assertEqual(result, 0.0)
    self.check_dtypes(f, self.array, result)

  def test_dixon_price(self):
    f = npf.DixonPrice()
    result = f(self.array)
    self.assertEqual(result, 4230.0)

    result = f(self.zero)
    self.assertEqual(result, 1.0)
    self.check_dtypes(f, self.array, result)


def test_random():
  list_tf_functions = tff.list_all_functions()
  list_np_functions = npf.list_all_functions()

  dims = 500
  random_pos_tf = tf.random.uniform((dims,), -1.0, 1.0, tf.float32)
  random_pos_np = random_pos_tf.numpy()

  print('random_pos_tf', random_pos_tf)
  print('random_pos_np', random_pos_np)

  for f_tf, f_np in zip(list_tf_functions, list_np_functions):
    print('----------------------------')
    tf_pos = random_pos_tf
    np_pos = random_pos_np

    if f_tf.name == 'Bohachevsky' and dims > 2:
      print('Bohachevsky: Considering only first 2 coordinates of the '
            'positions.')
      tf_pos = tf_pos[:2]
      np_pos = np_pos[:2]

    print(f_tf.name, f_tf(tf_pos))
    print(f_np.name, f_np(np_pos))
    print('----------------------------')

  print('Process finished.')
 
if __name__ == "__main__":
  unittest.main()

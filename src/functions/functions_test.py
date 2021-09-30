"""Function validation tests."""

import tensorflow as tf
import numpy as np
import unittest

import src.functions.tensorflow_functions as tff
import src.functions.numpy_functions as npf

class TestNumpyFunctions(unittest.TestCase):
  batch_size = 2 # batch size of array in multiple input testing
  @classmethod
  def setUpClass(cls) -> None:
    cls.array = np.array([1,2,3,4], dtype=np.float64)
    cls.batch = cls.array[None].repeat(cls.batch_size, axis=0)
    cls.zero = np.array([0,0,0,0], dtype=np.float64)

  @classmethod
  def tearDownClass(cls) -> None:
    del cls.array
    del cls.zero
    del cls.batch
  
  # check dtypes between function's input and output
  def check_dtypes(self, function, input, output):
    self.assertEqual(input.dtype, output.dtype,
      f"{function.name} output is {output.dtype} when should be {input.dtype}.")

  
  def test_ackley(self):
    f = npf.Ackley()
    array_result = 8.43469444443746497
    batch_result = np.array(array_result).repeat(self.batch_size)
    
    result = f(self.array)
    self.assertEqual(result, array_result)

    result = f(self.batch)
    self.assertTrue(np.array_equal(result, batch_result))

    result = f(self.zero)
    self.assertEqual(result, 4.44089209850062616e-16)
    self.check_dtypes(f, self.array, result)
  
  
  def test_griewank(self):
    f = npf.Griewank()
    array_result = 1.00187037800320189
    batch_result = np.array(array_result).repeat(self.batch_size)

    result = f(self.array)
    self.assertEqual(result, array_result)

    result = f(self.batch)
    self.assertTrue(np.array_equal(result, batch_result))

    result = f(self.zero)
    self.assertEqual(result, 0.0)
    self.check_dtypes(f, self.array, result)
  
  
  def test_rastrigin(self):
    f = npf.Rastrigin()
    array_result = 30.0
    batch_result = np.array(array_result).repeat(self.batch_size)

    result = f(self.array)
    self.assertEqual(result, array_result)

    result = f(self.batch)
    self.assertTrue(np.array_equal(result, batch_result))

    result = f(self.zero)
    self.assertEqual(result, 0.0)
    self.check_dtypes(f, self.array, result)

  
  def test_levy(self):
    f = npf.Levy()
    array_result = 2.76397190019909811
    batch_result = np.array(array_result).repeat(self.batch_size)

    result = f(self.array)
    self.assertEqual(result, array_result)

    result = f(self.batch)
    self.assertTrue(np.array_equal(result, batch_result))

    result = f(self.zero)
    self.assertEqual(result, 0.897533662350923467)
    self.check_dtypes(f, self.array, result)

  
  def test_rosenbrock(self):
    f = npf.Rosenbrock()
    array_result = 2705.0
    batch_result = np.array(array_result).repeat(self.batch_size)
    
    result = f(self.array)
    self.assertEqual(result, array_result)

    result = f(self.batch)
    self.assertTrue(np.array_equal(result, batch_result))

    result = f(self.zero)
    self.assertEqual(result, 3.0)
    self.check_dtypes(f, self.array, result)

  
  def test_zakharov(self):
    f = npf.Zakharov()
    array_result = 50880.0
    batch_result = np.array(array_result).repeat(self.batch_size)
    
    result = f(self.array)
    self.assertEqual(result, array_result)

    result = f(self.batch)
    self.assertTrue(np.array_equal(result, batch_result))

    result = f(self.zero)
    self.assertEqual(result, 0.0)
    self.check_dtypes(f, self.array, result)

  
  def test_sum_squares(self):
    f = npf.SumSquares()
    array_result = 100.0
    batch_result = np.array(array_result).repeat(self.batch_size)
    
    result = f(self.array)
    self.assertEqual(result, array_result)

    result = f(self.batch)
    self.assertTrue(np.array_equal(result, batch_result))

    result = f(self.zero)
    self.assertEqual(result, 0.0)
    self.check_dtypes(f, self.array, result)
  
  
  def test_sphere(self):
    f = npf.Sphere()
    array_result = 30.0
    batch_result = np.array(array_result).repeat(self.batch_size)
    
    result = f(self.array)
    self.assertEqual(result, array_result)

    result = f(self.batch)
    self.assertTrue(np.array_equal(result, batch_result))

    result = f(self.zero)
    self.assertEqual(result, 0.0)
    self.check_dtypes(f, self.array, result)

  
  def test_rotated_hyper_ellipsoid(self):
    f = npf.RotatedHyperEllipsoid()
    array_result = 50.0
    batch_result = np.array(array_result).repeat(self.batch_size)
    
    result = f(self.array)
    self.assertEqual(result, array_result)

    result = f(self.batch)
    self.assertTrue(np.array_equal(result, batch_result))

    result = f(self.zero)
    self.assertEqual(result, 0.0)
    self.check_dtypes(f, self.array, result)

  
  def test_dixon_price(self):
    f = npf.DixonPrice()
    array_result = 4230.0
    batch_result = np.array(array_result).repeat(self.batch_size)
    
    result = f(self.array)
    self.assertEqual(result, array_result)

    result = f(self.batch)
    self.assertTrue(np.array_equal(result, batch_result))

    result = f(self.zero)
    self.assertEqual(result, 1.0)
    self.check_dtypes(f, self.array, result)


class TestTensorflowFunctions(unittest.TestCase):
  
  batch_size = 10 # batch size of array in multiple input testing
  dtype = tf.float64

  @classmethod
  def setUpClass(cls) -> None:
    cls.array = tf.constant([1,2,3,4], dtype=cls.dtype)
    cls.batch = tf.repeat(cls.array[None], cls.batch_size, 0)
    cls.zero = tf.zeros((4,), dtype=cls.dtype)

  @classmethod
  def tearDownClass(cls) -> None:
    del cls.array
    del cls.zero
    del cls.batch
  
  # check dtypes between function's input and output
  def check_dtypes(self, function, input, output):
    self.assertEqual(input.dtype, output.dtype,
      f"{function} output is {output.dtype} when should be {input.dtype}.")
  

  # Get batch expected result from array expected result
  def batch_result(self, array_result):
    return tf.repeat(tf.expand_dims(array_result, 0), self.batch_size, 0)
  

  def check_shapes(self, output: tf.Tensor, expected_output: tf.Tensor):
    self.assertEqual(output.shape, expected_output.shape)


  def test_ackley(self):
    f = tff.Ackley()

    # Expected Values
    array_result = tf.constant(8.43469444443746497, self.dtype)
    zero_result = tf.constant(4.44089209850062616e-16, self.dtype)
    batch_result = self.batch_result(array_result)
    
    result = f(self.array)
    self.assertEqual(result, array_result)

    result = f(self.batch)
    self.assertTrue(tf.reduce_all(result == batch_result))
    self.check_shapes(result, batch_result)

    result = f(self.zero)
    self.assertEqual(result, zero_result)

    f = tf.function(f) # Testing Tracing

    result = f(self.array)
    self.assertEqual(result, array_result)

    self.check_shapes(result, zero_result)
    self.check_dtypes(f, self.array, result)
  
  
  def test_griewank(self):
    f = tff.Griewank()

    # Expected Values
    array_result = tf.constant(1.00187037800320189, self.dtype)
    zero_result = tf.constant(0.0, self.dtype)
    batch_result = self.batch_result(array_result)

    result = f(self.array)
    self.assertEqual(result, array_result)

    result = f(self.batch)
    self.assertTrue(tf.reduce_all(result == batch_result))
    self.check_shapes(result, batch_result)

    result = f(self.zero)
    self.assertEqual(result, zero_result)

    f = tf.function(f) # Testing Tracing

    result = f(self.array)
    self.assertEqual(result, array_result)

    self.check_shapes(result, zero_result)
    self.check_dtypes(f, self.array, result)
    
  
  def test_rastrigin(self):
    f = tff.Rastrigin()

    # Expected Values
    array_result = tf.constant(30.0, self.dtype)
    zero_result = tf.constant(0.0, self.dtype)
    batch_result = self.batch_result(array_result)

    result = f(self.array)
    self.assertEqual(result, array_result)

    result = f(self.batch)
    self.assertTrue(tf.reduce_all(result == batch_result))
    self.check_shapes(result, batch_result)

    result = f(self.zero)
    self.assertEqual(result, zero_result)

    f = tf.function(f) # Testing Tracing

    result = f(self.array)
    self.assertEqual(result, array_result)

    self.check_shapes(result, zero_result)
    self.check_dtypes(f, self.array, result)
    
  
  def test_levy(self):
    f = tff.Levy()

    # Expected Values
    array_result = tf.constant(2.76397190019909811, self.dtype)
    zero_result = tf.constant(0.897533662350923467, self.dtype)
    batch_result = self.batch_result(array_result)

    result = f(self.array)
    self.assertEqual(result, array_result)

    result = f(self.batch)
    self.assertTrue(tf.reduce_all(result == batch_result))
    self.check_shapes(result, batch_result)

    result = f(self.zero)
    self.assertEqual(result, zero_result)

    f = tf.function(f) # Testing Tracing

    result = f(self.array)
    self.assertEqual(result, array_result)

    self.check_shapes(result, zero_result)
    self.check_dtypes(f, self.array, result)
    
  
  def test_rosenbrock(self):
    f = tff.Rosenbrock()

    # Expected Values
    array_result = tf.constant(2705.0, self.dtype)
    zero_result = tf.constant(3.0, self.dtype)
    batch_result = self.batch_result(array_result)
    
    result = f(self.array)
    self.assertEqual(result, array_result)

    result = f(self.batch)
    self.assertTrue(tf.reduce_all(result == batch_result))
    self.check_shapes(result, batch_result)

    result = f(self.zero)
    self.assertEqual(result, zero_result)

    f = tf.function(f) # Testing Tracing

    result = f(self.array)
    self.assertEqual(result, array_result)

    self.check_shapes(result, zero_result)
    self.check_dtypes(f, self.array, result)
    
  
  def test_zakharov(self):
    f = tff.Zakharov()

    # Expected Values
    array_result = tf.constant(50880.0, self.dtype)
    zero_result = tf.constant(0.0, self.dtype)
    batch_result = self.batch_result(array_result)
    
    result = f(self.array)
    self.assertEqual(result, array_result)

    result = f(self.batch)
    self.assertTrue(tf.reduce_all(result == batch_result))
    self.check_shapes(result, batch_result)

    result = f(self.zero)
    self.assertEqual(result, zero_result)

    f = tf.function(f) # Testing Tracing

    result = f(self.array)
    self.assertEqual(result, array_result)

    self.check_shapes(result, zero_result)
    self.check_dtypes(f, self.array, result)
    
  
  def test_sum_squares(self):
    f = tff.SumSquares()

    # Expected Values
    array_result = tf.constant(100.0, self.dtype)
    zero_result = tf.constant(0.0, self.dtype)
    batch_result = self.batch_result(array_result)
    
    result = f(self.array)
    self.assertEqual(result, array_result)

    result = f(self.batch)
    self.assertTrue(tf.reduce_all(result == batch_result))
    self.check_shapes(result, batch_result)

    result = f(self.zero)
    self.assertEqual(result, zero_result)

    f = tf.function(f) # Testing Tracing

    result = f(self.array)
    self.assertEqual(result, array_result)

    self.check_shapes(result, zero_result)
    self.check_dtypes(f, self.array, result)
    
  
  def test_sphere(self):
    f = tff.Sphere()

    # Expected Values
    array_result = tf.constant(30.0, self.dtype)
    zero_result = tf.constant(0.0, self.dtype)
    batch_result = self.batch_result(array_result)
    
    result = f(self.array)
    self.assertEqual(result, array_result)

    result = f(self.batch)
    self.assertTrue(tf.reduce_all(result == batch_result))
    self.check_shapes(result, batch_result)

    result = f(self.zero)
    self.assertEqual(result, zero_result)

    f = tf.function(f) # Testing Tracing

    result = f(self.array)
    self.assertEqual(result, array_result)

    self.check_shapes(result, zero_result)
    self.check_dtypes(f, self.array, result)
  
  
  def test_rotated_hyper_ellipsoid(self):
    f = tff.RotatedHyperEllipsoid()

    # Expected Values
    array_result = tf.constant(50.0, self.dtype)
    zero_result = tf.constant(0.0, self.dtype)
    batch_result = self.batch_result(array_result)
    
    result = f(self.array)
    self.assertEqual(result, array_result)

    result = f(self.batch)
    self.assertTrue(tf.reduce_all(result == batch_result))
    self.check_shapes(result, batch_result)

    result = f(self.zero)
    self.assertEqual(result, zero_result)

    f = tf.function(f) # Testing Tracing

    result = f(self.array)
    self.assertEqual(result, array_result)

    self.check_shapes(result, zero_result)
    self.check_dtypes(f, self.array, result)
    

  def test_dixon_price(self):
    f = tff.DixonPrice()

    # Expected Values
    array_result = tf.constant(4230.0, self.dtype)
    zero_result = tf.constant(1.0, self.dtype)
    batch_result = self.batch_result(array_result)
    
    result = f(self.array)
    self.assertEqual(result, array_result)

    result = f(self.batch)
    self.assertTrue(tf.reduce_all(result == batch_result))
    self.check_shapes(result, batch_result)

    result = f(self.zero)
    self.assertEqual(result, zero_result)

    f = tf.function(f) # Testing Tracing

    result = f(self.array)
    self.assertEqual(result, array_result)

    self.check_shapes(result, zero_result)
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

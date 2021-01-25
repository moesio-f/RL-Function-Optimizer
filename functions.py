import tensorflow as tf
import numpy as np

# Some functions to test
def sphere(x):
  return tf.reduce_sum(x*x)

def ackley(x, a=20, b=0.2, c=2*np.math.pi):
  if isinstance(x, tf.Tensor):
    d = x.shape.as_list()[0]
  else:
    d = len(x)
  return -a * tf.exp(-b * tf.sqrt(tf.reduce_sum(x*x)/d)) - \
            tf.exp(tf.reduce_sum(tf.cos(c*x))/d) + a + np.math.e

def griewank(x):
  sum = x**2 / 4000
  prod = tf.cos(np.array([num/tf.sqrt(i) for i, num in enumerate(x, start=1)]))
  return tf.reduce_sum(sum) - tf.reduce_prod(prod) + 1

def rastrigin(x):
  if isinstance(x, tf.Tensor):
    d = x.shape.as_list()[0]
  else:
    d = len(x)
  return 10*d + tf.reduce_sum(x**2 - 10*tf.cos(x * 2 * np.math.pi))
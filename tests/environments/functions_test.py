"""Function validation tests."""

import tensorflow as tf

import functions.numpy_functions as np_functions
import functions.tensorflow_functions as tf_functions

list_tf_functions = tf_functions.list_all_functions()
list_np_functions = np_functions.list_all_functions()

dims = 2
random_pos_tf = tf.random.uniform((dims,), -1.0, 1.0, tf.float32)
random_pos_np = random_pos_tf.numpy()

print('random_pos_tf', random_pos_tf)
print('random_pos_np', random_pos_np)

for f_tf, f_np in zip(list_tf_functions, list_np_functions):
  print('----------------------------')
  print(f_tf.name, f_tf(random_pos_tf))
  print(f_np.name, f_np(random_pos_np))
  print('----------------------------')

print('done')

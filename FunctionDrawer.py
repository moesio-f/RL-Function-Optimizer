import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

class FunctionDrawer(object):
  def __init__(self, function, range, resolution=80):
    self.create_mesh(function, range, resolution)
    
  def create_mesh(self, function, range, resolution):
    self.function = function
    self.ax = plt.subplot(projection='3d')
    self.range = range
    self.res = resolution

    # creating mesh
    x = y = np.linspace(-self.range, self.range, self.res)
    X,Y = np.lib.meshgrid(x, y)
    
    zs = [self.function(np.array([x,y])) for x,y in zip(np.ravel(X), np.ravel(Y))]

    if isinstance(zs[0], tf.Tensor):
      zs = [x.numpy() for x in zs]
      
    zs = np.array(zs)
    Z = zs.reshape(X.shape)

    self.mesh = (X,Y,Z)

  def draw_mesh(self, **kwargs):
    self.ax.clear()
    self.ax.plot_surface(self.mesh[0],self.mesh[1],self.mesh[2],**kwargs)

  def draw(self, **kwargs):
    plt.pause(0.1)
    plt.draw()


# Some functions to test
D = tf.random.uniform((2,), -2.0, 2.0)
def sphere(x):
  x = x + D
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
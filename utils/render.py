"""Utility methods for function environments rendering."""

import matplotlib.pyplot as plt
import numpy as np

from functions.base import Function


# TODO: Atualizar classes para renderizar os ambientes

class FunctionDrawer(object):
  """ Draw Optimization functions implemented with tensorflow
  function:
      tensorflow function to draw
  domain:
      hypercube tuple with minimum and maximum values for the domain
      ex: (-10, 10)
          (-5.12, 5.12)
  resolution:
      integer representing quality of render
      note: bigger numbers make interactive window slow
  """

  def __init__(self, function: Function, resolution=80):
    self.set_mesh(function, resolution)
    self.points = []

  #  Set internal mesh to a new function
  def set_mesh(self, function: Function, resolution=80):
    self.function = function
    self.fig = plt.figure()
    self.ax = self.fig.add_subplot(projection='3d')
    self.quality = resolution

    # creating mesh
    x = y = np.linspace(function.domain.min, function.domain.max, self.quality)
    X, Y = np.lib.meshgrid(x, y)

    zs = np.array([np.ravel(X), np.ravel(Y)])
    zs = self.function(zs)

    Z = zs.reshape(X.shape)
    self.mesh = (X, Y, Z)

  def clear(self):
    self.ax.clear()
    self.points = []

  def draw_mesh(self, **kwargs):
    self.ax.plot_surface(self.mesh[0], self.mesh[1], self.mesh[2], **kwargs)
    self.draw()

  def update_scatter(self, new_position: np.ndarray):
    x, y = new_position[None].T
    z = self.function(new_position)
    self.points[0]._offsets3d = (x, y, z)

  def scatter(self, point, color='r', **kwargs):
    z = self.function(point)
    x, y = point
    self.points.append(self.ax.scatter(x, y, z, color=color, **kwargs))
    self.draw()

  def draw(self):
    plt.draw()

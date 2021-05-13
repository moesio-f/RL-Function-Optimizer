import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from functions.function import Function

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

    def __init__(self, function:Function, resolution=80):
        self.set_mesh(function, resolution)

    #  Set internal mesh to a new function
    def set_mesh(self, function: Function, resolution=80):

        self.function = function
        self.ax = plt.subplot(projection='3d')
        self.quality = resolution

        # creating mesh
        x = y = np.linspace(function.domain.min, function.domain.max, self.quality)
        X, Y = np.lib.meshgrid(x, y)

        zs = np.array([np.ravel(X), np.ravel(Y)])
        zs = tf.convert_to_tensor(zs, dtype=tf.float32)
        zs = self.function(zs).numpy()

        Z = zs.reshape(X.shape)
        self.mesh = (X, Y, Z)

    def draw_mesh(self, **kwargs):
        self.ax.clear()
        self.ax.plot_surface(self.mesh[0], self.mesh[1], self.mesh[2], **kwargs)
        self.draw()

    def draw_point(self, point, color='r'):
        z = self.function(point)
        if tf.is_tensor(z):
            z = z.numpy()
        x, y = point
        self.ax.scatter(x, y, z, color=color)
        self.draw()

    def draw(self, pause_time: float = None, **kwargs):
        if pause_time:
            plt.pause(pause_time)
        plt.draw()

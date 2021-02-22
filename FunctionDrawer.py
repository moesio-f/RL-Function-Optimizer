import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


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
        X, Y = np.lib.meshgrid(x, y)

        zs = [self.function(np.array([x, y])) for x, y in zip(np.ravel(X), np.ravel(Y))]

        if isinstance(zs[0], tf.Tensor):
            zs = [x.numpy() for x in zs]

        zs = np.array(zs)
        Z = zs.reshape(X.shape)

        self.mesh = (X, Y, Z)

    def draw_mesh(self, **kwargs):
        self.ax.clear()
        self.ax.plot_surface(self.mesh[0], self.mesh[1], self.mesh[2], **kwargs)

    def draw(self, **kwargs):
        plt.pause(0.1)
        plt.draw()

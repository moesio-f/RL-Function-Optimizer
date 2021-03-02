import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

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
    def __init__(self, function=None, domain=(-10, 10), resolution=80):
        if not function:
            def linear(x):
                return tf.reduce_sum(x, axis=0)
            function = linear
        self.set_mesh(function, domain, resolution)

    #  Set internal mesh to a new function
    def set_mesh(self, function, domain:tuple, resolution=80):

        self.function = function
        self.ax = plt.subplot(projection='3d')
        self.domain = domain
        self.quality = resolution

        # creating mesh
        x = y = np.linspace(domain[0], domain[1], self.quality)
        X, Y = np.lib.meshgrid(x, y)

        zs = [np.ravel(X), np.ravel(Y)]
        zs = tf.constant(zs)
        zs = self.function(zs).numpy()

        Z = zs.reshape(X.shape)
        self.mesh = (X, Y, Z)

    def draw_mesh(self, **kwargs):
        self.ax.clear()
        self.ax.plot_surface(self.mesh[0], self.mesh[1], self.mesh[2], **kwargs)
        self.draw()
    
    def draw_point(self, point, color='r'):
        z = self.function(point).numpy() # get 3d height
        x,y = point
        self.ax.scatter(x, y, z, color=color)
        self.draw()
    
    def draw(self, pause_time:float = None, **kwargs):
        if pause_time:
            plt.pause(pause_time)
        plt.draw()

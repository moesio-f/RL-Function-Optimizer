import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from functions.tensorflow_functions import Sphere


def sphereMesh():
    x_space = np.linspace(sphere.domain.min, sphere.domain.max, 30)
    y_space = np.linspace(sphere.domain.min, sphere.domain.max, 30)
    X, Y = np.lib.meshgrid(x_space, y_space)
    Z = X ** 2 + Y ** 2
    return X, Y, Z


sphere = Sphere()
ITERATIONS = 100
DIMENTION = 1000
learning_rate = 0.1

# figura da função em 3d
fig = plt.figure()
ax = plt.axes(projection='3d')
X, Y, Z = sphereMesh()
ax.plot_surface(X, Y, Z, alpha=0.5, cmap='viridis')

# algoritmo de otimização
agent = tf.random.uniform(minval=sphere.domain.min, maxval=sphere.domain.max, shape=(DIMENTION,))
done = False
while not done:
    with tf.GradientTape() as tape:
        tape.watch(agent)
        loss = sphere(agent)
        print(loss.numpy())
    gradients = tape.gradient(loss, agent)
    update = -learning_rate * gradients
    agent += update

    loss2d = sphere(tf.Variable([agent[0], agent[1]]))

    ax.scatter(agent[0], agent[1], loss2d, color='r')
    plt.pause(0.1)

    if loss < 0.05:
        done = True
plt.show()

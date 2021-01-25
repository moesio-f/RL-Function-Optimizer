import matplotlib as mpl
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

from functions import sphere, ackley, rastrigin
from FunctionDrawer import FunctionDrawer
from DDPG import Agent

mpl.use('TkAgg')

EPISODES = 10
STEPS = 50
DIMENTION = 2
INTERVAL = 5.12

agent = Agent.DDPGAgent([2 * DIMENTION + 1], 2, 5.12, -5.12)


def get_gradient(function, position):
    with tf.GradientTape() as tape:
        tape.watch(position)
        loss = function(position)
    return tape.gradient(loss, position), loss


def optimize(f, training=True, render=False):
    drawer = FunctionDrawer(f, INTERVAL)
    for ep in range(EPISODES):
        if render:
            drawer.draw_mesh(alpha=0.5)
        state = tf.random.uniform(minval=-INTERVAL, maxval=INTERVAL, shape=(DIMENTION,))
        for i in range(STEPS):
            gradients, loss = get_gradient(f, state)
            loss = tf.reshape(loss, [1])
            input = tf.concat([state, loss, gradients], 0)
            
            action = INTERVAL * agent.choose_action(input, training)
            new_state = state + action
            reward = -loss

            # Saber se saiu do domínio da função
            done = (tf.reduce_max(new_state) > INTERVAL) or (tf.reduce_min(new_state) < -INTERVAL)

            # action = -0.1 * gradients  # gradient descent
            # action = agent(state + loss + gradients)  # learned algorithm

            if training:
                agent.memorize(state, action, reward, new_state, done)
                agent.learn()

            if render:
                loss2d = f(tf.Variable([state[0], state[1]]))
                drawer.ax.scatter(state[0], state[1], loss2d, color='r')
                drawer.draw()

            state = new_state
            if done:
                break


if __name__ == '__main__':
    functions = [sphere, ackley, rastrigin]

    for function in functions:
        const = tf.random.uniform((DIMENTION,), -1, 1)

        def f(x):
            return function(x + const)
        optimize(f)

import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from FunctionDrawer import FunctionDrawer, sphere
from DDPG import Agent


EPISODES = 10
STEPS = 50
DIMENTION = 2
INTERVAL = 5.12

drawer = FunctionDrawer(sphere, INTERVAL)
agent = Agent.DDPGAgent([2*DIMENTION + 1], 2, 5.12, -5.12)

def optimize(f):

    for ep in range(EPISODES):
        drawer.draw_mesh(alpha=0.5)

        state = tf.random.uniform(minval=-INTERVAL, maxval=INTERVAL, shape=(DIMENTION,))
        done = False
        for i in range(STEPS):

            # calculate loss and gradients
            with tf.GradientTape() as tape:
                tape.watch(state)
                loss = f(state)
                print(loss.numpy())
            gradients = tape.gradient(loss, state)
            
            # action = -0.1*gradients                   # gradient descent
            # action = agent(state + loss + gradients)  # learned algorithm

            loss = tf.reshape(loss, [1])
            input = tf.concat([state, loss, gradients], 0)
            
            action = agent.choose_action(input)
            agent.learn()
            
            state += action
            
            # draw
            loss2d = sphere(tf.Variable([state[0], state[1]]))
            drawer.ax.scatter(state[0], state[1], loss2d, color='r')
            drawer.draw()

            done = loss < 0.05
            if done:
                break

optimize(sphere)
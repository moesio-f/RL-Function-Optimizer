import matplotlib as mpl
import tensorflow as tf
import numpy as np
from random import sample
from collections import deque

from functions import *
from FunctionDrawer import FunctionDrawer
from DDPG_Agent import DDPGAgent

mpl.use('TkAgg')


# Recebe um deque como argumento
# Retorna os gradientes e f(x)
def get_gradient(f, position):
    with tf.GradientTape() as tape:
        tape.watch(position)
        loss = f(position)
    grads = tape.gradient(loss, position)
    loss = tf.reshape(loss, [1])
    return grads, loss


# Recebe um deque como argumento
# Retorna a média dos gradientes normalizada (média não ponderada)
def get_gradients_mean_normalized(gradients_hist):
    gradients_hist_tensor = tf.convert_to_tensor([gradients_hist[i] for i in range(len(gradients_hist))])
    gradients_mean = tf.reduce_mean(gradients_hist_tensor, axis=0)
    return tf.math.l2_normalize(gradients_mean, axis=0)


# Recebe um deque como argumento
# Retorna se o agente 'parou' em uma determinada posição
def has_stopped(position_hist):
    if len(position_hist) < 5:
        return False
    position_hist_tensor = tf.convert_to_tensor([position_hist[i] for i in range(len(position_hist))])
    mean_pos = tf.reduce_mean(position_hist_tensor)
    max_pos = tf.reduce_max(position_hist_tensor)
    min_pos = tf.reduce_min(position_hist_tensor)
    dif = tf.abs(tf.sqrt(tf.reduce_sum((max_pos - mean_pos) ** 2)) - tf.sqrt(tf.reduce_sum((mean_pos - min_pos) ** 2)))
    return dif < 0.0001


def optimize(f, optimizer, f_name=None, training=True, render=False,
             BEST_SOLUTION=tf.float32.max,
             EPISODES=10, STEPS=50, LOW=-5.12, HIGH=5.12, DIMS=10):
    drawer = FunctionDrawer(f, HIGH)
    rewards = []
    for ep in range(EPISODES):
        if render:
            drawer.draw_mesh(alpha=0.5)

        episode_reward = 0.0

        position = tf.random.uniform(minval=LOW, maxval=HIGH, shape=(DIMS,))
        gradients, loss = get_gradient(f, position)

        state = position

        for i in range(STEPS):
            action = optimizer.choose_action(observation=state, current_position=position, training=training)

            new_position = position + action
            new_gradients, new_loss = get_gradient(f, new_position)

            new_state = new_position
            reward = -loss
            episode_reward += reward

            # Teste usando máximo iterações como único critério de parada
            done = (i >= STEPS - 1)

            if loss < BEST_SOLUTION:
                BEST_SOLUTION = loss

            if training:
                optimizer.memorize(state, action, reward, new_state, done)
                optimizer.learn()

            if render:
                loss2d = f(tf.Variable([position[0], position[1]]))
                drawer.ax.scatter(position[0], position[1], loss2d, color='r')
                drawer.draw()

            position = new_position
            gradients = new_gradients
            state = new_state
            loss = new_loss

            if done:
                break

        rewards.append(episode_reward)
        if ep % 10 == 0:
            print('function: %s episode: %d \n' % (f_name, ep),
                  '\tfinal objective value: %.2f\n' % loss,
                  '\tepisode reward: %.2f\n' % rewards[-1],
                  '\taverage reward (last 20 ep): %.2f\n\n' % np.mean(rewards[-20:]))
            optimizer.save_model(make_dir=True)

    return BEST_SOLUTION


if __name__ == '__main__':
    # Número de episódios para cada função
    episodes = 50
    # Quantidade de interações realizadas em cada episódio (2000)
    # OBS:. Durante treino podemos utilizar um número menor de steps
    steps = 2000
    # Dimensões das funções: 30 **(Testar para maiores dimensões)
    dims = 30
    # num_gradients_history = 20 Por hora não vai ser utilizado.
    # Quantidade total de treinos realizados em todas funções
    total_training = 50

    agent = DDPGAgent((dims,), dims)
    best_solution = tf.float32.max

    # No caso de uma única função, o agente treina em (episodes*total_training) episódios
    for ep in range(total_training):
        print('---current training: %d | best solution: %.2f---\n' % (ep, best_solution))
        function = sphere
        low, high = get_low_and_high(function)
        agent.set_low_and_high(low, high)
        function_name = get_function_name(function)
        best_solution = optimize(function, agent, f_name=function_name,
                                 BEST_SOLUTION=best_solution,
                                 EPISODES=episodes, STEPS=steps, LOW=low, HIGH=high, DIMS=dims)
        # Salvar o melhor fitness encontrado até o momento
        # Fitness x Iteração

    agent.save_model(make_dir=True)
    print('\n--best solution found: %.2f--\n' % best_solution)

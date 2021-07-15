"""TD3-IG agent test on FunctionEnvironment."""

import numpy as np
import tensorflow as tf
from tf_agents.agents.ddpg import critic_network as critic_net
from tf_agents.drivers import dynamic_step_driver as dy_sd
from tf_agents.environments import tf_py_environment
from tf_agents.environments import wrappers
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.train.utils import train_utils
from tf_agents.utils import common

from agents import td3_inverting_gradients as td3_ig
from environments import py_function_environment as py_fun_env
from functions import numpy_functions as npf
from networks import linear_actor_network as linear_actor_net
from utils import evaluation

if __name__ == '__main__':
  num_episodes = 2000
  initial_collect_episodes = 20
  collect_steps_per_iteration = 1

  # Hiperparametros da memória de replay
  buffer_size = 1000000
  batch_size = 256

  # Hiperparametros do Agente
  actor_lr = 1e-5
  critic_lr = 2e-4
  tau = 1e-4
  actor_update_period = 2
  target_update_period = 2

  discount = 0.99
  gradient_clipping_norm = 2.5

  exploration_noise_std = 0.5
  exploration_noise_std_end = 0.1
  exploration_noise_num_episodes = 1850
  target_policy_noise = 0.2
  target_policy_noise_clip = 0.5

  # Actor Network
  fc_layer_params = [256, 256]

  # Critic Network
  action_fc_layer_params = None
  observation_fc_layer_params = None
  joint_fc_layer_params = [256, 256]

  # Criando o Env.

  steps = 250
  steps_eval = 500

  dims = 2
  function = npf.Sphere()

  min_reward = -1e5
  max_reward = 1e5
  reward_scale = 1.0

  env_training = py_fun_env.PyFunctionEnvironment(function=function,
                                                  dims=dims,
                                                  clip_actions=False)
  env_training = wrappers.TimeLimit(env=env_training, duration=steps)

  env_eval = py_fun_env.PyFunctionEnvironment(function=function,
                                              dims=dims,
                                              clip_actions=False)
  env_eval = wrappers.TimeLimit(env=env_eval, duration=steps_eval)

  tf_env_training = tf_py_environment.TFPyEnvironment(environment=env_training)
  tf_env_eval = tf_py_environment.TFPyEnvironment(environment=env_eval)

  obs_spec = tf_env_training.observation_spec()
  act_spec = tf_env_training.action_spec()
  time_spec = tf_env_training.time_step_spec()

  # Atualizando a exploração para ser determinada com base nos steps
  exploration_noise_num_steps = round(exploration_noise_num_episodes * steps)

  # Criando as redes.
  actor_network = linear_actor_net.LinearActorNetwork(
    input_tensor_spec=obs_spec,
    output_tensor_spec=act_spec,
    fc_layer_params=fc_layer_params,
    activation_fn=tf.keras.activations.relu)
  critic_network = critic_net.CriticNetwork(
    input_tensor_spec=(obs_spec, act_spec),
    observation_fc_layer_params=observation_fc_layer_params,
    action_fc_layer_params=action_fc_layer_params,
    joint_fc_layer_params=joint_fc_layer_params,
    activation_fn=tf.keras.activations.relu,
    output_activation_fn=tf.keras.activations.linear)

  # Criando o agente.
  actor_optimizer = tf.keras.optimizers.Adam(learning_rate=actor_lr)
  critic_optimizer = tf.keras.optimizers.Adam(learning_rate=critic_lr)

  train_step = train_utils.create_train_step()

  agent = td3_ig.Td3AgentInvertingGradients(
    time_step_spec=time_spec,
    action_spec=act_spec,
    actor_network=actor_network,
    critic_network=critic_network,
    actor_optimizer=actor_optimizer,
    critic_optimizer=critic_optimizer,
    target_update_tau=tau,
    exp_noise_std=exploration_noise_std,
    exp_noise_std_end=exploration_noise_std_end,
    exp_noise_steps=exploration_noise_num_steps,
    target_policy_noise=target_policy_noise,
    target_policy_noise_clip=target_policy_noise_clip,
    actor_update_period=actor_update_period,
    target_update_period=target_update_period,
    train_step_counter=train_step,
    gradient_clipping=gradient_clipping_norm,
    gamma=discount)

  agent.initialize()

  # Replay buffer
  replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
    data_spec=agent.collect_data_spec,
    batch_size=tf_env_training.batch_size,
    max_length=buffer_size)

  # Criando o Driver.

  # Data Collection (Collect for initial episodes)
  driver = dy_sd.DynamicStepDriver(env=tf_env_training,
                                   policy=agent.collect_policy,
                                   observers=[replay_buffer.add_batch],
                                   num_steps=collect_steps_per_iteration)

  initial_collect_driver = dy_sd.DynamicStepDriver(
    env=tf_env_training,
    policy=agent.collect_policy,
    observers=[replay_buffer.add_batch],
    num_steps=collect_steps_per_iteration)

  # Convertendo principais funções para tf.function's (Graph Mode)
  initial_collect_driver.run = common.function(initial_collect_driver.run)
  driver.run = common.function(driver.run)
  agent.train = common.function(agent.train)

  # Realizando coleta inicial.
  for _ in range(initial_collect_episodes):
    done = False
    while not done:
      time_step, _ = initial_collect_driver.run()
      done = time_step.is_last()

  # Criando o dataset.
  dataset = replay_buffer.as_dataset(
    sample_batch_size=batch_size,
    num_steps=2).prefetch(64)

  iterator = iter(dataset)

  # Treinamento do Agente
  agent.train_step_counter.assign(0)

  for ep in range(num_episodes):
    done = False
    best_solution = np.finfo(np.float32).max
    ep_rew = 0.0
    while not done:
      time_step, _ = driver.run()
      experience, unused_info = next(iterator)
      agent.train(experience)

      # Acessando indíce 0 por conta da dimensão extra (batch)
      obj_value = driver.env.get_info().objective_value[0]

      if obj_value < best_solution:
        best_solution = obj_value

      ep_rew += time_step.reward
      done = time_step.is_last()

    print('episode = {0} '
          'Best solution on episode: {1} '
          'Return on episode: {2}'.format(ep, best_solution, ep_rew))

  # Realizando os testes do agente (Policy e Collect Policy) para 1 único
  # episódio.
  evaluation.evaluate_agent(tf_env_eval,
                            agent.policy,
                            function,
                            dims,
                            name_algorithm='TD3-IG',
                            save_to_file=True,
                            verbose=True)

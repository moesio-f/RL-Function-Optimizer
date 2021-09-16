"""DDPG para aprender um algoritmo de otimização."""

import os

import tensorflow as tf
from tf_agents.agents.ddpg import ddpg_agent
from tf_agents.agents.ddpg import critic_network as critic_net
from tf_agents.drivers import dynamic_step_driver as dy_sd
from tf_agents.environments import tf_py_environment
from tf_agents.environments import wrappers
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.train.utils import train_utils
from tf_agents.utils import common
from tf_agents.metrics import tf_metrics

from src.single_agent.environments import py_function_environment as py_fun_env
from src.single_agent.networks import linear_actor_network as lin_actor_net
from src.functions import numpy_functions as npf
from src import config

from experiments.evaluation import utils as eval_utils
from experiments.training import utils as training_utils

LOG_DIR = os.path.join(config.EXPERIMENTS_DIR, 'logs')
TRAIN_LOG_DIR = os.path.join(LOG_DIR, 'training')
EVAL_LOG_DIR = os.path.join(LOG_DIR, 'eval')

if __name__ == '__main__':
  num_episodes = 5000
  initial_collect_episodes = 10
  collect_steps_per_iteration = 1

  buffer_size = 1000000
  batch_size = 256

  actor_lr = 1e-3
  critic_lr = 1e-3
  tau = 1e-2
  target_update_period = 1

  discount = 0.95

  ou_stddev = 0.2
  ou_damping = 0.15

  actor_layer_params = [256, 256]

  critic_action_fc_layer_params = None
  critic_observation_fc_layer_params = None
  # Camadas e unidades para a 'critic network'.
  critic_fc_layer_params = [256, 256]

  steps = 50  # Quantidade de interações agente-ambiente para treino.
  steps_eval = 500  # Quantidade de interações agente-ambiente para avaliação.

  dims = 30  # Dimensões da função.
  function = npf.Sphere()

  '''
  # Criação dos SummaryWriter's
  train_summary_writer = tf.compat.v2.summary.create_file_writer(
    TRAIN_LOG_DIR, flush_millis=10 * 1000)
  train_summary_writer.set_as_default()

  eval_summary_writer = tf.compat.v2.summary.create_file_writer(
    EVAL_LOG_DIR, flush_millis=10 * 1000)
  
  # Criação das métricas
  train_metrics = [tf_metrics.AverageReturnMetric(),
                   tf_metrics.MaxReturnMetric()]

  eval_metrics = []
  '''

  # Criação do ambiente
  env_training = py_fun_env.PyFunctionEnvironment(function=function,
                                                  dims=dims)
  env_training = wrappers.TimeLimit(env=env_training, duration=(steps - 1))

  env_eval = py_fun_env.PyFunctionEnvironment(function=function,
                                              dims=dims)
  env_eval = wrappers.TimeLimit(env=env_eval, duration=(steps_eval - 1))

  tf_env_training = tf_py_environment.TFPyEnvironment(environment=env_training)
  tf_env_eval = tf_py_environment.TFPyEnvironment(environment=env_eval)

  obs_spec = tf_env_training.observation_spec()
  act_spec = tf_env_training.action_spec()
  time_spec = tf_env_training.time_step_spec()

  # Criação do agente, redes neurais, otimizadores
  actor_network = lin_actor_net.LinearActorNetwork(
    input_tensor_spec=obs_spec,
    output_tensor_spec=act_spec,
    fc_layer_params=actor_layer_params,
    activation_fn=tf.keras.activations.relu)

  critic_network = critic_net.CriticNetwork(
    input_tensor_spec=(obs_spec, act_spec),
    observation_fc_layer_params=critic_observation_fc_layer_params,
    action_fc_layer_params=critic_action_fc_layer_params,
    joint_fc_layer_params=critic_fc_layer_params,
    activation_fn=tf.keras.activations.relu,
    output_activation_fn=tf.keras.activations.linear)

  actor_optimizer = tf.keras.optimizers.Adam(learning_rate=actor_lr,
                                             clipnorm=0.5)
  critic_optimizer = tf.keras.optimizers.Adam(learning_rate=critic_lr,
                                              clipnorm=0.5)

  train_step = train_utils.create_train_step()

  agent = ddpg_agent.DdpgAgent(
    time_step_spec=time_spec,
    action_spec=act_spec,
    actor_network=actor_network,
    critic_network=critic_network,
    actor_optimizer=actor_optimizer,
    critic_optimizer=critic_optimizer,
    ou_stddev=ou_stddev,
    ou_damping=ou_damping,
    target_update_tau=tau,
    target_update_period=target_update_period,
    train_step_counter=train_step,
    gamma=discount)

  agent.initialize()

  # Criação do Replay Buffer e drivers
  replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
    data_spec=agent.collect_data_spec,
    batch_size=tf_env_training.batch_size,
    max_length=buffer_size)

  driver = dy_sd.DynamicStepDriver(env=tf_env_training,
                                   policy=agent.collect_policy,
                                   observers=[replay_buffer.add_batch],
                                   num_steps=collect_steps_per_iteration)

  initial_collect_driver = dy_sd.DynamicStepDriver(
    env=tf_env_training,
    policy=agent.collect_policy,
    observers=[replay_buffer.add_batch],
    num_steps=collect_steps_per_iteration)

  # Conversão das principais funções para tf.function's
  initial_collect_driver.run = common.function(initial_collect_driver.run)
  driver.run = common.function(driver.run)
  agent.train = common.function(agent.train)

  # Coleta inicial
  for _ in range(initial_collect_episodes):
    for _ in range(steps):
      initial_collect_driver.run()

  # Criação do dataset
  dataset = replay_buffer.as_dataset(
    num_parallel_calls=3,
    sample_batch_size=batch_size,
    num_steps=2).prefetch(3)

  iterator = iter(dataset)

  agent.train_step_counter.assign(0)


  @tf.function
  def train_phase():
    print('tracing')
    driver.run()
    experience, _ = next(iterator)
    agent.train(experience)


  # Treinamento
  for ep in range(num_episodes):
    for _ in range(steps):
      train_phase()
    print(ep)

  # Avaliação do algoritmo aprendido (policy) em 100 episódios distintos.
  # Produz um gráfico de convergência para o agente na função.
  eval_utils.evaluate_agent(tf_env_eval,
                            agent.policy,
                            function,
                            dims,
                            algorithm_name='DDPG',
                            save_to_file=False,
                            episodes=100)

  # Salvamento da policy aprendida.
  # Pasta de saída: output/DDPG-{dims}D-{function.name}/
  # OBS:. Caso já exista, a saída é sobrescrita.
  training_utils.save_policy('DDPG',
                             function,
                             dims,
                             agent.policy)

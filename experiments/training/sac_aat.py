"""SAC para aprender um algoritmo de otimização."""

import time

import numpy as np
import tensorflow as tf
from tf_agents.agents.ddpg import critic_network as critic_net
from tf_agents.agents.sac import sac_agent
from tf_agents.agents.sac import tanh_normal_projection_network as tanh_net
from tf_agents.drivers import dynamic_step_driver as dy_sd
from tf_agents.drivers import dynamic_episode_driver as dy_ed
from tf_agents.environments import tf_py_environment
from tf_agents.environments import wrappers
from tf_agents.networks import actor_distribution_network as actor_net
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.train.utils import train_utils
from tf_agents.utils import common
from tf_agents.metrics import tf_metrics

from src.single_agent.environments import py_function_environment as py_fun_env
from src.single_agent.metrics import tf_custom_metrics
from src.single_agent.typing.types import LayerParam
from src.functions import numpy_functions as npf
from src.functions import core as functions_core

from experiments.evaluation import utils as eval_utils
from experiments.training import utils as training_utils


def train_sac(function: functions_core.Function,
              dims: int,
              training_episodes: int = 2000,
              stop_threshold: float = None,
              env_steps: int = 250,
              env_eval_steps: int = 500,
              eval_interval: int = 100,
              eval_episodes: int = 10,
              initial_collect_episodes: int = 20,
              collect_steps_per_iteration: int = 1,
              buffer_size: int = 1000000,
              batch_size: int = 256,
              actor_lr: float = 3e-4,
              critic_lr: float = 3e-4,
              alpha_lr: float = 3e-4,
              tau: float = 5e-3,
              target_update_period: int = 2,
              discount: float = 0.99,
              actor_layers: LayerParam = None,
              critic_action_layers: LayerParam = None,
              critic_observation_layers: LayerParam = None,
              critic_joint_layers: LayerParam = None,
              summary_flush_secs: int = 10,
              debug_summaries: bool = False,
              summarize_grads_and_vars: bool = False):
  algorithm_name = 'SAC'

  # Criando o diretório do agente
  agent_dir = training_utils.create_agent_dir(algorithm_name,
                                              function,
                                              dims)

  # Obtendo função equivalente em TensorFlow (Utilizada no cálculo das métricas)
  tf_function = npf.get_tf_function(function)

  env_training = py_fun_env.PyFunctionEnv(function=function,
                                          dims=dims)
  env_training = wrappers.TimeLimit(env=env_training, duration=env_steps)

  env_eval = py_fun_env.PyFunctionEnv(function=function,
                                      dims=dims)
  env_eval = wrappers.TimeLimit(env=env_eval, duration=env_eval_steps)

  # Conversão para TFPyEnvironment's
  tf_env_training = tf_py_environment.TFPyEnvironment(environment=env_training)
  tf_env_eval = tf_py_environment.TFPyEnvironment(environment=env_eval)

  # Criação dos SummaryWriter's
  print('Creating logs directories.')
  log_dir, log_eval_dir, log_train_dir = training_utils.create_logs_dir(
    agent_dir)

  train_summary_writer = tf.compat.v2.summary.create_file_writer(
    log_train_dir, flush_millis=summary_flush_secs * 1000)
  train_summary_writer.set_as_default()

  eval_summary_writer = tf.compat.v2.summary.create_file_writer(
    log_eval_dir, flush_millis=summary_flush_secs * 1000)

  # Criação das métricas
  train_metrics = [tf_metrics.AverageReturnMetric(),
                   tf_metrics.MaxReturnMetric()]

  eval_metrics = [tf_metrics.AverageReturnMetric(buffer_size=eval_episodes),
                  tf_custom_metrics.AverageBestObjectiveValueMetric(
                    function=tf_function, buffer_size=eval_episodes)]

  # Criação do agente, redes neurais, otimizadores
  obs_spec = tf_env_training.observation_spec()
  act_spec = tf_env_training.action_spec()
  time_spec = tf_env_training.time_step_spec()

  if actor_layers is None:
    actor_layers = [256, 256]

  continuous_projection_net = tanh_net.TanhNormalProjectionNetwork

  actor_network = actor_net.ActorDistributionNetwork(
    input_tensor_spec=obs_spec,
    output_tensor_spec=act_spec,
    fc_layer_params=actor_layers,
    continuous_projection_net=continuous_projection_net)

  if critic_joint_layers is None:
    critic_joint_layers = [256, 256]

  critic_activation_fn = tf.keras.activations.relu
  critic_output_activation_fn = tf.keras.activations.linear

  critic_network = critic_net.CriticNetwork(
    input_tensor_spec=(obs_spec, act_spec),
    observation_fc_layer_params=critic_observation_layers,
    action_fc_layer_params=critic_action_layers,
    joint_fc_layer_params=critic_joint_layers,
    activation_fn=critic_activation_fn,
    output_activation_fn=critic_output_activation_fn,
    kernel_initializer='glorot_uniform',
    last_kernel_initializer='glorot_uniform')

  actor_optimizer = tf.keras.optimizers.Adam(learning_rate=actor_lr)
  critic_optimizer = tf.keras.optimizers.Adam(learning_rate=critic_lr)
  alpha_optimizer = tf.keras.optimizers.Adam(learning_rate=alpha_lr)

  train_step = train_utils.create_train_step()

  agent = sac_agent.SacAgent(time_step_spec=time_spec,
                             action_spec=act_spec,
                             critic_network=critic_network,
                             critic_optimizer=critic_optimizer,
                             actor_network=actor_network,
                             actor_optimizer=actor_optimizer,
                             alpha_optimizer=alpha_optimizer,
                             gamma=discount,
                             target_update_tau=tau,
                             target_update_period=target_update_period,
                             td_errors_loss_fn=tf.math.squared_difference,
                             train_step_counter=train_step,
                             debug_summaries=debug_summaries,
                             summarize_grads_and_vars=summarize_grads_and_vars)

  agent.initialize()

  # Criação do Replay Buffer e drivers
  replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
    data_spec=agent.collect_data_spec,
    batch_size=tf_env_training.batch_size,
    max_length=buffer_size)

  observers_train = [replay_buffer.add_batch] + train_metrics
  driver = dy_sd.DynamicStepDriver(
    env=tf_env_training,
    policy=agent.collect_policy,
    observers=observers_train,
    num_steps=collect_steps_per_iteration)

  initial_collect_driver = dy_ed.DynamicEpisodeDriver(
    env=tf_env_training,
    policy=agent.collect_policy,
    observers=[replay_buffer.add_batch],
    num_episodes=initial_collect_episodes)

  eval_driver = dy_ed.DynamicEpisodeDriver(env=tf_env_eval,
                                           policy=agent.policy,
                                           observers=eval_metrics,
                                           num_episodes=eval_episodes)

  # Conversão das principais funções para tf.function's
  initial_collect_driver.run = common.function(initial_collect_driver.run)
  driver.run = common.function(driver.run)
  eval_driver.run = common.function(eval_driver.run)
  agent.train = common.function(agent.train)

  print('Initializing replay buffer by collecting experience for {0} '
        'episodes with a collect policy.'.format(initial_collect_episodes))
  initial_collect_driver.run()

  # Criação do dataset
  dataset = replay_buffer.as_dataset(
    num_parallel_calls=3,
    sample_batch_size=batch_size,
    num_steps=2).prefetch(3)

  iterator = iter(dataset)

  # Criação da função para calcular as métricas
  def compute_eval_metrics():
    return eval_utils.eager_compute(eval_metrics,
                                    eval_driver,
                                    train_step=agent.train_step_counter,
                                    summary_writer=eval_summary_writer,
                                    summary_prefix='Metrics')

  agent.train_step_counter.assign(0)

  @tf.function
  def train_phase():
    print('tracing')
    driver.run()
    experience, _ = next(iterator)
    agent.train(experience)

  # Salvando hiperparâmetros antes de iniciar o treinamento
  hp_dict = {
    "discount": discount,
    "tau": tau,
    "target_update_period": target_update_period,
    "training_episodes": training_episodes,
    "buffer_size": buffer_size,
    "batch_size": batch_size,
    "stop_threshold": stop_threshold,
    "train_env": {
      "steps": env_steps,
      "function": function.name,
      "dims": dims,
      "domain": function.domain
    },
    "eval_env": {
      "steps": env_eval_steps,
      "function": function.name,
      "dims": dims,
      "domain": function.domain
    },
    "networks": {
      "actor_net": {
        "class": type(actor_network).__name__,
        "continuous_projection_net": continuous_projection_net.__name__,
        "actor_layers": actor_layers
      },
      "critic_net": {
        "class": type(critic_network).__name__,
        "activation_fn": critic_activation_fn.__name__,
        "output_activation_fn": critic_output_activation_fn.__name__,
        "critic_action_fc_layers": critic_action_layers,
        "critic_obs_fc_layers": critic_observation_layers,
        "critic_joint_layers": critic_joint_layers
      }
    },
    "optimizers": {
      "actor_optimizer": type(actor_optimizer).__name__,
      "actor_lr": actor_lr,
      "critic_optimizer": type(critic_optimizer).__name__,
      "critic_lr": critic_lr,
      "alpha_optimizer": type(alpha_optimizer).__name__,
      "alpha_lr": alpha_lr
    }
  }

  training_utils.save_specs(agent_dir, hp_dict)
  tf.summary.text("Hyperparameters",
                  training_utils.json_pretty_string(hp_dict),
                  step=0)

  for ep in range(training_episodes):
    start_time = time.time()
    for _ in range(env_steps):
      train_phase()

      for train_metric in train_metrics:
        train_metric.tf_summaries(train_step=agent.train_step_counter)

    if ep % eval_interval == 0:
      print('-------- Evaluation --------')
      start_eval = time.time()
      results = compute_eval_metrics()
      avg_return = results.get(eval_metrics[0].name)
      avg_best_value = results.get(eval_metrics[1].name)
      print('Average return: {0}'.format(avg_return))
      print('Average best value: {0}'.format(avg_best_value))
      print('Eval delta time: {0:.2f}'.format(time.time() - start_eval))
      print('---------------------------')
      if stop_threshold is not None and avg_best_value < stop_threshold:
        break

    delta_time = time.time() - start_time
    print('Finished episode {0}. '
          'Delta time since last episode: {1:.2f}'.format(ep, delta_time))

  # Computando métricas de avaliação uma última vez.
  compute_eval_metrics()

  # Avaliação do algoritmo aprendido (policy) em 100 episódios distintos.
  # Produz um gráfico de convergência para o agente na função.
  eval_utils.evaluate_agent(tf_env_eval,
                            agent.policy,
                            function,
                            dims,
                            env_eval_steps,
                            algorithm_name=algorithm_name,
                            save_to_file=True,
                            episodes=100,
                            save_dir=agent_dir)

  # Salvamento da policy aprendida.
  # Pasta de saída: output/SAC-{dims}D-{function.name}-{num}/policy
  training_utils.save_policy(agent_dir, agent.policy)


if __name__ == '__main__':
  train_sac(npf.Ackley(), 2,
            env_steps=50,
            eval_interval=10,
            env_eval_steps=100,
            stop_threshold=1e-2,
            training_episodes=500)

"""SAC agent test on FunctionEnvironment."""

import numpy as np
import tensorflow as tf
from tf_agents.agents import SacAgent
from tf_agents.agents.ddpg.critic_network import CriticNetwork
from tf_agents.agents.sac.tanh_normal_projection_network import \
  TanhNormalProjectionNetwork
from tf_agents.drivers import dynamic_step_driver
from tf_agents.environments import tf_py_environment
from tf_agents.environments.wrappers import TimeLimit
from tf_agents.networks.actor_distribution_network import \
  ActorDistributionNetwork
from tf_agents.replay_buffers.tf_uniform_replay_buffer import \
  TFUniformReplayBuffer
from tf_agents.train.utils import train_utils
from tf_agents.utils import common

from environments.py_function_environment import PyFunctionEnvironment
from functions.numpy_functions import Sphere
from utils.evaluation import evaluate_agent

# Hiperparametros de treino
num_episodes = 2000
initial_collect_episodes = 20
c_steps_per_it = 1
replay_buffer_capacity = 1000000
batch_size = 256
target_update_tau = 0.005
target_update_period = 1

# Hiperparametros do Agente
actor_lr = 3e-4
critic_lr = 3e-4
alpha_lr = 3e-4
discount = 0.99

# Networks
actor_layer_params = [256, 256]
critic_observation_layer_params = None
critic_action_layer_params = None
critic_joint_layer_params = [256, 256]

# Envs
steps = 250
steps_eval = 500

dims = 2
function = Sphere()

env = PyFunctionEnvironment(function=function, dims=dims, clip_actions=True)

env_training = TimeLimit(env=env, duration=steps)
env_eval = TimeLimit(env=env, duration=steps_eval)

tf_env_training = tf_py_environment.TFPyEnvironment(environment=env_training)
tf_env_eval = tf_py_environment.TFPyEnvironment(environment=env_eval)

obs_spec = tf_env_training.observation_spec()
act_spec = tf_env_training.action_spec()
time_spec = tf_env_training.time_step_spec()

# Creating networks
actor_network = ActorDistributionNetwork(input_tensor_spec=obs_spec,
                                         output_tensor_spec=act_spec,
                                         fc_layer_params=actor_layer_params,
                                         continuous_projection_net=
                                         TanhNormalProjectionNetwork)

critic_network = CriticNetwork(input_tensor_spec=(obs_spec, act_spec),
                               observation_fc_layer_params=
                               critic_observation_layer_params,
                               action_fc_layer_params=critic_action_layer_params,
                               joint_fc_layer_params=critic_joint_layer_params,
                               activation_fn=tf.keras.activations.relu,
                               output_activation_fn=tf.keras.activations.linear,
                               kernel_initializer='glorot_uniform',
                               last_kernel_initializer='glorot_uniform')

# Creating agent
actor_optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=actor_lr)
critic_optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=critic_lr)
alpha_optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=alpha_lr)

train_step = train_utils.create_train_step()

agent = SacAgent(time_step_spec=time_spec,
                 action_spec=act_spec,
                 critic_network=critic_network,
                 critic_optimizer=critic_optimizer,
                 actor_network=actor_network,
                 actor_optimizer=actor_optimizer,
                 alpha_optimizer=alpha_optimizer,
                 gamma=discount,
                 target_update_tau=target_update_tau,
                 target_update_period=target_update_period,
                 td_errors_loss_fn=tf.math.squared_difference,
                 train_step_counter=train_step)

agent.initialize()

# Data Collection and Replay Buffer
replay_buffer = TFUniformReplayBuffer(data_spec=agent.collect_data_spec,
                                      batch_size=tf_env_training.batch_size,
                                      max_length=replay_buffer_capacity)

# Creating a dataset
dataset = replay_buffer.as_dataset(
  sample_batch_size=batch_size,
  num_steps=2).prefetch(64)

iterator = iter(dataset)

initial_driver = dynamic_step_driver.DynamicStepDriver(env=tf_env_training,
                                                       policy=
                                                       agent.collect_policy,
                                                       observers=[
                                                         replay_buffer.add_batch],
                                                       num_steps=c_steps_per_it)

driver = dynamic_step_driver.DynamicStepDriver(env=tf_env_training,
                                               policy=agent.collect_policy,
                                               observers=[
                                                 replay_buffer.add_batch],
                                               num_steps=c_steps_per_it)

driver.run = common.function(driver.run)
initial_driver.run = common.function(initial_driver.run)

for _ in range(initial_collect_episodes):
  done = False
  while not done:
    time_step, _ = initial_driver.run()
    done = time_step.is_last()

# Training
agent.train = common.function(agent.train)
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

  print(
    'episode = {0} Best solution on episode: {1} Return on episode: {2}'.format(
      ep, best_solution, ep_rew))

evaluate_agent(tf_env_eval, agent.policy, function, dims, name_algorithm='SAC',
               save_to_file=True)

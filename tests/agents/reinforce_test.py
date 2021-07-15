"""REINFORCE agent test on FunctionEnvironment."""

import tensorflow as tf
from tf_agents.agents.reinforce import reinforce_agent
from tf_agents.drivers import dynamic_episode_driver as dy_ed
from tf_agents.environments import tf_py_environment
from tf_agents.environments import wrappers
from tf_agents.networks import actor_distribution_network as actor_net
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.train.utils import train_utils
from tf_agents.utils import common

from environments import py_function_environment as py_fun_env
from functions import numpy_functions as npf
from utils import evaluation

if __name__ == '__main__':
  # Hiperparametros de treino
  num_episodes = 2000

  # Hiperparametros do Agente
  actor_lr = 1e-3
  discount = 1.0

  # Actor Network
  fc_layer_params = [512, 512, 256]

  # Envs
  steps = 100
  steps_eval = 500

  dims = 2
  function = npf.Sphere()

  env_training = py_fun_env.PyFunctionEnvironment(function=function,
                                                  dims=dims,
                                                  clip_actions=True)
  env_training = wrappers.TimeLimit(env=env_training, duration=steps)

  env_eval = py_fun_env.PyFunctionEnvironment(function=function,
                                              dims=dims,
                                              clip_actions=True)
  env_eval = wrappers.TimeLimit(env=env_eval, duration=steps_eval)

  print(env_training.wrapped_env())
  print(env_eval.wrapped_env())

  tf_env_training = tf_py_environment.TFPyEnvironment(environment=env_training)
  tf_env_eval = tf_py_environment.TFPyEnvironment(environment=env_eval)

  obs_spec = tf_env_training.observation_spec()
  act_spec = tf_env_training.action_spec()
  time_spec = tf_env_training.time_step_spec()

  # Creating network and Distribution
  actor_network = actor_net.ActorDistributionNetwork(input_tensor_spec=obs_spec,
                                                     output_tensor_spec=
                                                     act_spec,
                                                     fc_layer_params=
                                                     fc_layer_params)

  # Creating agent
  actor_optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=actor_lr)

  train_step = train_utils.create_train_step()

  agent = reinforce_agent.ReinforceAgent(time_step_spec=time_spec,
                                         action_spec=act_spec,
                                         actor_network=actor_network,
                                         optimizer=actor_optimizer,
                                         gamma=discount,
                                         normalize_returns=True,
                                         train_step_counter=train_step)

  agent.initialize()

  # Data Collection and Replay Buffer
  replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
    data_spec=agent.collect_data_spec,
    batch_size=tf_env_training.batch_size,
    max_length=steps + 5)

  driver = dy_ed.DynamicEpisodeDriver(env=tf_env_training,
                                      policy=agent.collect_policy,
                                      observers=[replay_buffer.add_batch],
                                      num_episodes=1)

  driver.run = common.function(driver.run)
  agent.train = common.function(agent.train)

  # Training
  agent.train_step_counter.assign(0)

  for ep in range(num_episodes):
    driver.run()
    experience = replay_buffer.gather_all()
    agent.train(experience)

    observations = tf.unstack(experience.observation[0])
    rewards = tf.unstack(experience.reward[0])
    best_solution = min([function(x.numpy()) for x in observations])
    ep_rew = sum(rewards)
    print('episode = {0} '
          'Best solution on episode: '
          '{1} Return on episode: {2}'.format(ep, best_solution, ep_rew))

    replay_buffer.clear()

  evaluation.evaluate_agent(tf_env_eval, agent.policy, function, dims,
                            name_algorithm='REINFORCE',
                            save_to_file=True)

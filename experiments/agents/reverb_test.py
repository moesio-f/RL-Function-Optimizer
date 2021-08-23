"""Reverb TD3 agent test on FunctionEnvironment."""
import os
import sys
import tempfile

import numpy as np
import reverb
import tensorflow as tf
from tf_agents.agents.ddpg import critic_network as critic_net
from tf_agents.environments import wrappers
from tf_agents.policies import py_tf_eager_policy
from tf_agents.replay_buffers import reverb_replay_buffer
from tf_agents.replay_buffers import reverb_utils
from tf_agents.train import actor
from tf_agents.train import learner
from tf_agents.train import triggers
from tf_agents.train.utils import spec_utils
from tf_agents.train.utils import train_utils

from src.single_agent.agents import td3_ig_reverb_per as td3_ig_reverb
from src.single_agent.environments import py_function_environment as py_fun_env
from src.single_agent.environments import py_env_wrappers
from src.functions import numpy_functions as npf
from src.single_agent.networks import linear_actor_network as linear_actor_net
from src.single_agent.train import reverb_learner

sys.path.append('/content/RL-Function-Optimizer/')
tempdir = tempfile.gettempdir()

if __name__ == '__main__':
  # Hiperparametros de treino
  num_episodes = 800
  initial_collect_episodes = 20
  collect_steps_per_iteration = 1

  # Hiperparametros da memória de replay
  buffer_size = 10000000
  batch_size = 256
  prioritization_exponent = 0.8

  # Hiperparametros do learner
  is_weights_initial_exponent = 0.5
  is_weights_final_exponent = 1.0

  # Hiperparametros do Agente
  actor_lr = 1e-4
  critic_lr = 2e-3
  tau = 1e-4
  actor_update_period = 2
  target_update_period = 2

  discount = 0.99

  gradient_clipping_norm = 1.0

  exploration_noise_std = 0.6
  exploration_noise_std_end = 0.1
  exploration_noise_num_episodes = 700
  target_policy_noise = 0.2
  target_policy_noise_clip = 0.5

  policy_save_interval = 5000

  # Actor Network
  fc_layer_params = [400, 300]

  # Critic Network
  action_fc_layer_params = []
  observation_fc_layer_params = [400]
  joint_fc_layer_params = [300]

  # Envs
  steps = 500
  steps_eval = 2000
  dims = 20
  function = npf.Sphere()
  min_reward = -500
  max_reward = 500
  reward_scale = 1e-2

  env_training = py_fun_env.PyFunctionEnvironment(function=function,
                                                  dims=dims,
                                                  clip_actions=True)
  env_training = py_env_wrappers.RewardClip(env=env_training,
                                            min_reward=min_reward,
                                            max_reward=max_reward)
  env_training = py_env_wrappers.RewardScale(env=env_training,
                                             scale_factor=reward_scale)
  env_training = wrappers.TimeLimit(env=env_training, duration=steps)

  env_eval = py_fun_env.PyFunctionEnvironment(function=function,
                                              dims=dims,
                                              clip_actions=True)
  env_eval = py_env_wrappers.RewardClip(env=env_eval,
                                        min_reward=min_reward,
                                        max_reward=max_reward)
  env_eval = py_env_wrappers.RewardScale(env=env_eval,
                                         scale_factor=reward_scale)
  env_eval = wrappers.TimeLimit(env=env_eval, duration=steps_eval)

  obs_spec, act_spec, time_spec = (spec_utils.get_tensor_specs(env_training))

  exploration_noise_num_steps = round(exploration_noise_num_episodes * steps)
  is_weight_exponent_steps = round(0.85 * (num_episodes * steps))

  # Creating networks
  actor_network = linear_actor_net.LinearActorNetwork(
    input_tensor_spec=obs_spec,
    output_tensor_spec=act_spec,
    fc_layer_params=fc_layer_params,
    activation_fn=tf.keras.activations.relu)
  critic_network = critic_net.CriticNetwork(
    input_tensor_spec=(obs_spec, act_spec),
    observation_fc_layer_params=
    observation_fc_layer_params,
    action_fc_layer_params=action_fc_layer_params,
    joint_fc_layer_params=joint_fc_layer_params,
    activation_fn=tf.keras.activations.relu,
    output_activation_fn=tf.keras.activations.linear)

  # Creating agent
  actor_optimizer = tf.keras.optimizers.Adam(learning_rate=actor_lr)
  critic_optimizer = tf.keras.optimizers.Adam(learning_rate=critic_lr)

  train_step = train_utils.create_train_step()

  agent = td3_ig_reverb.Td3AgentReverb(
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
  table_name = 'per_table'
  reverb_table = reverb.Table(table_name,
                              max_size=buffer_size,
                              sampler=reverb.selectors.Prioritized(
                                prioritization_exponent),
                              remover=reverb.selectors.Fifo(),
                              rate_limiter=reverb.rate_limiters.MinSize(1))

  reverb_server = reverb.Server([reverb_table])

  replay_buffer = reverb_replay_buffer.ReverbReplayBuffer(
    data_spec=agent.collect_data_spec,
    table_name=table_name,
    sequence_length=2,
    local_server=reverb_server)

  rb_observer = reverb_utils.ReverbAddTrajectoryObserver(
    replay_buffer.py_client,
    table_name,
    sequence_length=2,
    stride_length=1)

  # Creating a dataset
  dataset = replay_buffer.as_dataset(
    sample_batch_size=batch_size,
    num_steps=2).prefetch(50)

  experience_dataset_fn = lambda: dataset

  eval_policy = py_tf_eager_policy.PyTFEagerPolicy(agent.policy,
                                                   use_tf_function=True)
  collect_policy = py_tf_eager_policy.PyTFEagerPolicy(agent.collect_policy,
                                                      use_tf_function=True)

  initial_collect_actor = actor.Actor(
    env_training,
    collect_policy,
    train_step,
    steps_per_run=initial_collect_episodes * steps,
    observers=[rb_observer])

  initial_collect_actor.run()

  collect_actor = actor.Actor(env_training,
                              collect_policy,
                              train_step,
                              steps_per_run=1,
                              observers=[rb_observer])

  saved_model_dir = os.path.join(tempdir, learner.POLICY_SAVED_MODEL_DIR)

  learning_triggers = [
    triggers.PolicySavedModelTrigger(saved_model_dir, agent, train_step,
                                     interval=policy_save_interval),
    triggers.StepPerSecondLogTrigger(train_step, interval=1000), ]

  agent_learner = reverb_learner.ReverbLearnerPER(tempdir,
                                                  train_step,
                                                  agent,
                                                  replay_buffer,
                                                  initial_is_weight_exp=
                                                  is_weights_initial_exponent,
                                                  final_is_weight_exp=
                                                  is_weights_final_exponent,
                                                  is_weight_exp_steps=
                                                  is_weight_exponent_steps,
                                                  experience_dataset_fn=
                                                  experience_dataset_fn,
                                                  triggers=learning_triggers)

  # Training
  agent.train_step_counter.assign(0)

  for ep in range(num_episodes):
    done = False
    best_solution = np.finfo(np.float32).max
    ep_rew = 0.0
    while not done:
      collect_actor.run()
      agent_learner.run(iterations=1)

      time_step = collect_actor._time_step
      obj_value = collect_actor._env.get_info().objective_value

      if obj_value < best_solution:
        best_solution = obj_value

      ep_rew += time_step.reward
      done = time_step.is_last()

    print(
      'episode = {0} '
      'Best solution on episode: {1} '
      'Return on episode: {2}'.format(ep, best_solution, ep_rew))

  rb_observer.close()
  reverb_server.stop()

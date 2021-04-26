import numpy as np
import tensorflow as tf

from tf_agents.agents.ppo.ppo_clip_agent import PPOClipAgent
from tf_agents.networks.value_network import ValueNetwork
from tf_agents.environments.wrappers import TimeLimit
from tf_agents.environments import tf_py_environment
from tf_agents.trajectories.trajectory import Trajectory
from tf_agents.drivers import dynamic_step_driver
from tf_agents.replay_buffers.tf_uniform_replay_buffer import TFUniformReplayBuffer
from tf_agents.networks.actor_distribution_network import ActorDistributionNetwork
from tf_agents.train.utils import train_utils
from tf_agents.utils import common

from functions.numpy_functions import *
from environments.py_function_environment import PyFunctionEnvironment
from utils.evaluation import evaluate_agent

# Hiperparametros de treino
num_episodes = 30000  # @param {type:"integer"}
collect_trajectories_per_training_iteration = 10  # @param {type:"integer"}

# Hiperparametros do Agente
lr = 3e-4  # @param {type:"number"}
discount = 0.99  # @param {type:"number"}
num_epochs = 25  # @param {type:"number"}
train_sequence_length = 250  # @param {type: "integer"}

# Networks
actor_layer_params = [256, 256]
value_layer_params = [256, 256]

# Envs
steps = 250  # @param {type:"integer"}
steps_eval = 500  # @param {type:"integer"}

dims = 2  # @param {type:"integer"}
function = Sphere()  # @param ["Sphere()", "Ackley()", "Griewank()", "Levy()", "Zakharov()", "RotatedHyperEllipsoid()", "Rosenbrock()"]{type: "raw"}

env = PyFunctionEnvironment(function=function, dims=dims)

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
                                         fc_layer_params=actor_layer_params)

value_network = ValueNetwork(input_tensor_spec=obs_spec,
                             fc_layer_params=value_layer_params)

# Creating agent
optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=lr)

train_step = train_utils.create_train_step()

agent = PPOClipAgent(time_step_spec=time_spec,
                     action_spec=act_spec,
                     actor_net=actor_network,
                     value_net=value_network,
                     optimizer=optimizer,
                     discount_factor=discount,
                     num_epochs=num_epochs,
                     train_step_counter=train_step)

agent.initialize()

# Data Collection and Replay Buffer
replay_buffer = TFUniformReplayBuffer(
    data_spec=agent.collect_data_spec,
    batch_size=tf_env_training.batch_size,
    max_length=train_sequence_length)

driver = dynamic_step_driver.DynamicStepDriver(env=tf_env_training,
                                               policy=agent.collect_policy,
                                               observers=[replay_buffer.add_batch],
                                               num_steps=train_sequence_length)

driver.run = common.function(driver.run)
agent.train = common.function(agent.train)

# Training
agent.train_step_counter.assign(0)

for ep in range(num_episodes):
    trajectories = []

    for _ in range(collect_trajectories_per_training_iteration):
        driver.run()
        trajectories.append(replay_buffer.gather_all())
        replay_buffer.clear()

    experiences = Trajectory(step_type=tf.concat([traj.step_type for traj in trajectories], axis=0),
                             observation=tf.concat([traj.observation for traj in trajectories], axis=0),
                             action=tf.concat([traj.action for traj in trajectories], axis=0),
                             policy_info={'dist_params': {'loc': tf.concat([traj.policy_info['dist_params']['loc']
                                                                            for traj in trajectories], axis=0),
                                                          'scale': tf.concat([traj.policy_info['dist_params']['scale']
                                                                              for traj in trajectories], axis=0)}},
                             next_step_type=tf.concat([traj.next_step_type for traj in trajectories], axis=0),
                             reward=tf.concat([traj.reward for traj in trajectories], axis=0),
                             discount=tf.concat([traj.discount for traj in trajectories], axis=0))

    agent.train(experiences)

    best_solutions = []
    ep_rewards = []
    for traj in trajectories:
        best_solutions.append(min([function(x) for x in tf.unstack(traj.observation[0])]))
        ep_rewards.append(sum(tf.unstack(traj.reward[0])))
    print('episode = {0} Average Best solution on episode: {1} Average Return on episode: {2}'.format(
        ep, np.mean(best_solutions), np.mean(ep_rewards)))

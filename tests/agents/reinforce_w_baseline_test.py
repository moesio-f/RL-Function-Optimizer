import tensorflow as tf

from tf_agents.agents import ReinforceAgent
from tf_agents.environments.wrappers import TimeLimit
from tf_agents.environments import tf_py_environment
from tf_agents.drivers import dynamic_episode_driver
from tf_agents.replay_buffers.tf_uniform_replay_buffer import TFUniformReplayBuffer
from tf_agents.networks.actor_distribution_network import ActorDistributionNetwork
from tf_agents.networks.value_network import ValueNetwork
from tf_agents.train.utils import train_utils
from tf_agents.utils import common

from functions.numpy_functions import *
from environments.py_function_environment import PyFunctionEnvironment
from utils.evaluation import evaluate_agent

# Hiperparametros de treino
num_episodes = 2000
collect_episodes_per_iteration = 1

# Hiperparametros do Agente
actor_lr = 5e-4
value_lr = 5e-4
discount = 1.0

# Networks
actor_layer_params = [256, 256]
value_layer_params = [256, 256]

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
                                         fc_layer_params=actor_layer_params)

value_network = ValueNetwork(input_tensor_spec=obs_spec,
                             fc_layer_params=value_layer_params)

# Creating agent
actor_optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=actor_lr)
value_optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=value_lr)

train_step = train_utils.create_train_step()

agent = ReinforceAgent(time_step_spec=time_spec,
                       action_spec=act_spec,
                       actor_network=actor_network,
                       value_network=value_network,
                       optimizer=actor_optimizer,
                       gamma=discount,
                       normalize_returns=True,
                       train_step_counter=train_step)

agent.initialize()

# Data Collection and Replay Buffer
replay_buffer = TFUniformReplayBuffer(
    data_spec=agent.collect_data_spec,
    batch_size=tf_env_training.batch_size,
    max_length=steps + 1)

driver = dynamic_episode_driver.DynamicEpisodeDriver(env=tf_env_training,
                                                     policy=agent.collect_policy,
                                                     observers=[replay_buffer.add_batch],
                                                     num_episodes=collect_episodes_per_iteration)

driver.run = common.function(driver.run)

# Training
agent.train = common.function(agent.train)
agent.train_step_counter.assign(0)

for ep in range(num_episodes):
    driver.run()
    experience = replay_buffer.gather_all()
    agent.train(experience)

    observations = tf.unstack(experience.observation[0])
    rewards = tf.unstack(experience.reward[0])
    best_solution = min([function(x.numpy()) for x in observations])
    ep_rew = sum(rewards)
    print('episode = {0} Best solution on episode: {1} Return on episode: {2}'.format(ep, best_solution, ep_rew))

    replay_buffer.clear()

evaluate_agent(tf_env_eval, agent.policy, function, dims, name_algorithm='REINFORCE-with-Baseline',
               save_to_file=False)
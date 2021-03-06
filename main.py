from functions.numpy_functions import *
from environments.py_function_environment import *
from utils.evaluation import evaluate_agent

import tensorflow as tf
from tf_agents.environments.wrappers import TimeLimit
from tf_agents.environments import tf_py_environment
from tf_agents.agents import Td3Agent
from tf_agents.policies.policy_saver import PolicySaver
from tf_agents.agents.ddpg.actor_network import ActorNetwork
from tf_agents.agents.ddpg.critic_network import CriticNetwork
from tf_agents.drivers import dynamic_step_driver
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.utils import common


# Hiperparametros de treino
num_episodes = 800  # @param {type:"integer"}
initial_collect_episodes = 10  # @param {type:"integer"}
collect_steps_per_iteration = 1  # @param {type:"integer"}

# Hiperparametros da memória de replay
buffer_size = 1000000  # @param {type:"integer"}
batch_size = 64  # @param {type:"number"}

# Hiperparametros do Agente
actor_lr = 1e-4  # @param {type:"number"}
critic_lr = 2e-4  # @param {type:"number"}
tau = 5e-4  # @param {type:"number"}
discount = 0.99  # @param {type:"number"}
exploration_noise_std = 0.2  # @param {type:"number"}
target_policy_noise = 0.2  # @param {type:"number"}
target_policy_noise_clip = 0.5  # @param {type:"number"}
actor_update_period = 2  # @param {type:"integer"}
target_update_period = 2  # @param {type:"integer"}
reward_scale_factor = 0.75  # @param {type:"number"}

# --- Arquitetura da rede ---
# Actor
fc_layer_params = [400, 300]  # FNN's do Actor
# Critic
observation_fc_layer_params = [400]  # FNN's apenas para observações
joint_fc_layer_params = [300]  # FNN's depois de concatenar (observação, ação)

# Envs
steps = 500  # @param {type:"integer"}
steps_eval = 2000  # @param {type:"integer"}
dims = 2  # @param {type:"integer"}
function = Ackley()

env = PyFunctionEnvironment(function=function, dims=dims)

env_training = TimeLimit(env=env, duration=steps)
env_eval = TimeLimit(env=env, duration=steps_eval)
tf_env_training = tf_py_environment.TFPyEnvironment(environment=env_training)
tf_env_eval = tf_py_environment.TFPyEnvironment(environment=env_eval)

obs_spec = tf_env_training.observation_spec()
act_spec = tf_env_training.action_spec()
time_spec = tf_env_training.time_step_spec()

# Creating networks
actor_network = ActorNetwork(input_tensor_spec=obs_spec,
                             output_tensor_spec=act_spec,
                             fc_layer_params=fc_layer_params,
                             activation_fn=tf.keras.activations.relu)
critic_network = CriticNetwork(input_tensor_spec=(obs_spec, act_spec),
                               observation_fc_layer_params=observation_fc_layer_params,
                               joint_fc_layer_params=joint_fc_layer_params,
                               activation_fn=tf.keras.activations.relu,
                               output_activation_fn=tf.keras.activations.linear)

# Creating agent
actor_optimizer = tf.keras.optimizers.Adam(learning_rate=actor_lr)
critic_optimizer = tf.keras.optimizers.Adam(learning_rate=critic_lr)

agent = Td3Agent(
    time_step_spec=time_spec,
    action_spec=act_spec,
    actor_network=actor_network,
    critic_network=critic_network,
    actor_optimizer=actor_optimizer,
    critic_optimizer=critic_optimizer,
    target_update_tau=tau,
    exploration_noise_std=exploration_noise_std,
    target_policy_noise=target_policy_noise,
    target_policy_noise_clip=target_policy_noise_clip,
    actor_update_period=actor_update_period,
    target_update_period=target_update_period,
    reward_scale_factor=reward_scale_factor,
    train_step_counter=tf.Variable(0),
    gamma=discount)

agent.initialize()

# Replay buffer
replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(data_spec=agent.collect_data_spec,
                                                               batch_size=tf_env_training.batch_size,
                                                               max_length=buffer_size)

# Data Collection (Collect for initial episodes)
driver = dynamic_step_driver.DynamicStepDriver(env=tf_env_training,
                                               policy=agent.collect_policy,
                                               observers=[replay_buffer.add_batch],
                                               num_steps=collect_steps_per_iteration)
driver.run = common.function(driver.run)

initial_collect_driver = dynamic_step_driver.DynamicStepDriver(env=tf_env_training,
                                                               policy=agent.collect_policy,
                                                               observers=[replay_buffer.add_batch],
                                                               num_steps=collect_steps_per_iteration)

initial_collect_driver.run = common.function(initial_collect_driver.run)

for _ in range(initial_collect_episodes):
    done = False
    while not done:
        time_step, _ = initial_collect_driver.run()
        done = time_step.is_last()

# Creating a dataset
dataset = replay_buffer.as_dataset(
    sample_batch_size=batch_size,
    num_steps=2)

iterator = iter(dataset)

# Training
agent.train = common.function(agent.train)
agent.train_step_counter.assign(0)

for ep in range(num_episodes):
    done = False
    best_solution = tf.float32.max
    ep_rew = 0.0
    while not done:
        time_step, _ = driver.run()
        experience, unused_info = next(iterator)
        agent.train(experience)

        obj_value = -time_step.reward.numpy()[0]

        if obj_value < best_solution and not time_step.is_first():
            best_solution = obj_value

        ep_rew += -obj_value
        done = time_step.is_last()

    print('episode = {0} Best solution on episode: {1} Return on episode: {2}'.format(ep, best_solution, ep_rew))

evaluate_agent(tf_env_eval, agent.policy, function, dims, name_algorithm='TD3', save_to_file=True)
evaluate_agent(tf_env_eval, agent.collect_policy, function, dims, name_algorithm='TD3', save_to_file=True)

tf_policy_saver = PolicySaver(agent.policy)
tf_policy_collect_saver = PolicySaver(agent.collect_policy)

tf_policy_saver.save('policy')
tf_policy_collect_saver.save('policy_collect')

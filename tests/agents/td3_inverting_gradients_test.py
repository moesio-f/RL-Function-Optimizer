"""Importando as funções e o ambiente."""

from functions.numpy_functions import *
from environments.py_function_environment_unbounded import PyFunctionEnvironmentUnbounded

"""Imports para Main (Agente, Redes, etc)"""

import tensorflow as tf

from tf_agents.environments.wrappers import TimeLimit
from tf_agents.environments import tf_py_environment
from tf_agents.agents.ddpg.critic_network import CriticNetwork
from tf_agents.drivers import dynamic_step_driver
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.train.utils import train_utils
from tf_agents.utils import common

from environments.py_env_wrappers import RewardClip, RewardScale
from networks.custom_actor_network import CustomActorNetwork
from agents.td3_inverting_gradients import Td3AgentInvertingGradients
from utils.evaluation import evaluate_agent

"""Hiperparâmetros"""

# Hiperparametros de treino
num_episodes = 1000  # @param {type:"integer"}
initial_collect_episodes = 15  # @param {type:"integer"}
collect_steps_per_iteration = 1  # @param {type:"integer"}

# Hiperparametros da memória de replay
buffer_size = 1000000  # @param {type:"integer"}
batch_size = 256  # @param {type:"number"}

# Hiperparametros do Agente
actor_lr = 1e-4  # @param {type:"number"}
critic_lr = 2e-3  # @param {type:"number"}
tau = 1e-3  # @param {type:"number"}
actor_update_period = 2  # @param {type:"integer"}
target_update_period = 2  # @param {type:"integer"}

discount = 0.99  # @param {type:"number"}
gradient_clipping_norm = 2.5  # @param {type:"number"}

exploration_noise_std = 0.55  # @param {type:"number"}
exploration_noise_std_end = 0.15  # @param {type: "number"}
exploration_noise_num_episodes = 700  # @param {type: "number"}
target_policy_noise = 0.2  # @param {type:"number"}
target_policy_noise_clip = 0.5  # @param {type:"number"}

# Actor Network
fc_layer_params = [512, 256, 128]  # FNN's do Actor

# Critic Network
action_fc_layer_params = []  # FNN's apenas para ações
observation_fc_layer_params = [512]  # FNN's apenas para observações
joint_fc_layer_params = [256, 128]  # FNN's depois de concatenar (observação, ação)

"""Criando o Env"""

# Envs
steps = 500  # @param {type:"integer"}
steps_eval = 2000  # @param {type:"integer"}

dims = 20  # @param {type:"integer"}
function = Sphere()  # @param ["Sphere()", "Ackley()", "Griewank()", "Levy()", "Zakharov()", "RotatedHyperEllipsoid()", "Rosenbrock()"]{type: "raw"}

min_reward = -1e5  # @param {type:"number"}
max_reward = 1e5  # @param {type:"number"}
reward_scale = 1.0  # @param {type:"number"}

env = PyFunctionEnvironmentUnbounded(function=function, dims=dims)
env = RewardScale(env=env, scale_factor=reward_scale)
env = RewardClip(env=env, min_reward=min_reward, max_reward=max_reward)

env_training = TimeLimit(env=env, duration=steps)
env_eval = TimeLimit(env=env, duration=steps_eval)

tf_env_training = tf_py_environment.TFPyEnvironment(environment=env_training)
tf_env_eval = tf_py_environment.TFPyEnvironment(environment=env_eval)

obs_spec = tf_env_training.observation_spec()
act_spec = tf_env_training.action_spec()
time_spec = tf_env_training.time_step_spec()

# Atualizando a exploração para ser determinada com base nos steps
exploration_noise_num_steps = round(exploration_noise_num_episodes * steps)

"""Criando as redes"""

# Creating networks
actor_network = CustomActorNetwork(input_tensor_spec=obs_spec,
                                   output_tensor_spec=act_spec,
                                   fc_layer_params=fc_layer_params,
                                   activation_fn=tf.keras.activations.relu,
                                   activation_action_fn=tf.keras.activations.linear)
critic_network = CriticNetwork(input_tensor_spec=(obs_spec, act_spec),
                               observation_fc_layer_params=observation_fc_layer_params,
                               action_fc_layer_params=action_fc_layer_params,
                               joint_fc_layer_params=joint_fc_layer_params,
                               activation_fn=tf.keras.activations.relu,
                               output_activation_fn=tf.keras.activations.linear)

"""Criando o agente"""

# Creating agent
actor_optimizer = tf.keras.optimizers.Adam(learning_rate=actor_lr)
critic_optimizer = tf.keras.optimizers.Adam(learning_rate=critic_lr)

train_step = train_utils.create_train_step()

agent = Td3AgentInvertingGradients(
    time_step_spec=time_spec,
    action_spec=act_spec,
    actor_network=actor_network,
    critic_network=critic_network,
    actor_optimizer=actor_optimizer,
    critic_optimizer=critic_optimizer,
    target_update_tau=tau,
    exploration_noise_std=exploration_noise_std,
    exploration_noise_std_end=exploration_noise_std_end,
    exploration_noise_num_steps=exploration_noise_num_steps,
    target_policy_noise=target_policy_noise,
    target_policy_noise_clip=target_policy_noise_clip,
    actor_update_period=actor_update_period,
    target_update_period=target_update_period,
    train_step_counter=train_step,
    gradient_clipping=gradient_clipping_norm,
    gamma=discount)

agent.initialize()

"""Replay Buffer"""

# Replay buffer
replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(data_spec=agent.collect_data_spec,
                                                               batch_size=tf_env_training.batch_size,
                                                               max_length=buffer_size)

"""Criando o Driver"""

# Data Collection (Collect for initial episodes)
driver = dynamic_step_driver.DynamicStepDriver(env=tf_env_training,
                                               policy=agent.collect_policy,
                                               observers=[replay_buffer.add_batch],
                                               num_steps=collect_steps_per_iteration)

initial_collect_driver = dynamic_step_driver.DynamicStepDriver(env=tf_env_training,
                                                               policy=agent.collect_policy,
                                                               observers=[replay_buffer.add_batch],
                                                               num_steps=collect_steps_per_iteration)

"""Convertendo principais funções para tf.function's (Graph Mode)"""

initial_collect_driver.run = common.function(initial_collect_driver.run)
driver.run = common.function(driver.run)
agent.train = common.function(agent.train)

"""Realizando coleta inicial"""
for _ in range(initial_collect_episodes):
    done = False
    while not done:
        time_step, _ = initial_collect_driver.run()
        done = time_step.is_last()

"""Criando o dataset"""

# Creating a dataset
dataset = replay_buffer.as_dataset(
    sample_batch_size=batch_size,
    num_steps=2).prefetch(64)

iterator = iter(dataset)

"""Treinamento do Agente"""

# Training
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

    print('episode = {0} Best solution on episode: {1} Return on episode: {2}'.format(ep, best_solution, ep_rew))

"""Realizando os testes do agente depois que sendo chamado"""

evaluate_agent(tf_env_eval, agent.policy, function, dims, name_algorithm='TD3-IG',
               save_to_file=True)

evaluate_agent(tf_env_eval, agent.collect_policy, function, dims, name_algorithm='TD3-IG',
               save_to_file=True)

"""Salvando ambas policies e agente"""

from tf_agents.policies.policy_saver import PolicySaver

tf_policy_saver = PolicySaver(agent.policy)
tf_policy_collect_saver = PolicySaver(agent.collect_policy)

tf_policy_saver.save('policy')
tf_policy_collect_saver.save('policy_collect')
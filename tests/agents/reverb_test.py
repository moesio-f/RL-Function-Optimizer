"""Importando as funções e o ambiente."""

import sys

sys.path.append('/content/RL-Function-Optimizer/')
import os

from functions.numpy_functions import *
from environments.py_function_environment import PyFunctionEnvironment

"""Imports para Main (Agente, Redes, etc)"""

import tensorflow as tf
import reverb
import tempfile

from tf_agents.agents.ddpg.critic_network import CriticNetwork

from tf_agents.replay_buffers import reverb_replay_buffer
from tf_agents.replay_buffers import reverb_utils

from tf_agents.environments.wrappers import TimeLimit

from tf_agents.train.utils import spec_utils
from tf_agents.train.utils import train_utils

from tf_agents.policies import py_tf_eager_policy

from tf_agents.train import actor
from tf_agents.train import learner
from tf_agents.train import triggers

from environments.py_env_wrappers import RewardClip, RewardScale
from networks.custom_actor_network import CustomActorNetwork
from agents.td3_ig_reverb_per import Td3AgentReverb
from train.reverb_learner import ReverbLearnerPER

tempdir = tempfile.gettempdir()

"""Hiperparâmetros"""

# Hiperparametros de treino
num_episodes = 800  # @param {type:"integer"}
initial_collect_episodes = 20  # @param {type:"integer"}
collect_steps_per_iteration = 1  # @param {type:"integer"}

# Hiperparametros da memória de replay
buffer_size = 10000000  # @param {type:"integer"}
batch_size = 256  # @param {type:"number"}
prioritization_exponent = 0.8  # @param {type:"number"}

# Hiperparametros do learner
is_weights_initial_exponent = 0.5  # @param {type:"number"}
is_weights_final_exponent = 1.0  # @param {type:"number"}

# Hiperparametros do Agente
actor_lr = 1e-4  # @param {type:"number"}
critic_lr = 2e-3  # @param {type:"number"}
tau = 1e-4  # @param {type:"number"}
actor_update_period = 2  # @param {type:"integer"}
target_update_period = 2  # @param {type:"integer"}

discount = 0.99  # @param {type:"number"}

gradient_clipping_norm = 1.0  # @param {type:"number"}

exploration_noise_std = 0.6  # @param {type:"number"}
exploration_noise_std_end = 0.1  # @param {type: "number"}
exploration_noise_num_episodes = 700  # @param {type: "integer"}
target_policy_noise = 0.2  # @param {type:"number"}
target_policy_noise_clip = 0.5  # @param {type:"number"}

policy_save_interval = 5000  # @param {type:"integer"}

# Actor Network
fc_layer_params = [400, 300]  # FNN's do Actor

# Critic Network
action_fc_layer_params = []  # FNN's apenas para ações
observation_fc_layer_params = [400]  # FNN's apenas para observações
joint_fc_layer_params = [300]  # FNN's depois de concatenar (observação, ação)

"""Criando o Env"""

# Envs
steps = 500  # @param {type:"integer"}
steps_eval = 2000  # @param {type:"integer"}
dims = 20  # @param {type:"integer"}
function = Sphere()  # @param ["Sphere()", "Ackley()", "Griewank()", "Levy()", "Zakharov()", "RotatedHyperEllipsoid()", "Rosenbrock()"]{type: "raw"}
min_reward = -500  # @param {type:"number"}
max_reward = 500 # @param {type:"number"}
reward_scale = 1e-2  # @param {type:"number"}

env = PyFunctionEnvironment(function=function, dims=dims)
env = RewardClip(env=env, min_reward=min_reward, max_reward=max_reward)
env = RewardScale(env=env, scale_factor=reward_scale)

env_training = TimeLimit(env=env, duration=steps)
env_eval = TimeLimit(env=env, duration=steps_eval)

obs_spec, act_spec, time_spec = (spec_utils.get_tensor_specs(env_training))

""" Atualizando alguns hiperparametros """

exploration_noise_num_steps = round(exploration_noise_num_episodes * steps)
is_weight_exponent_steps = round(0.85 * (num_episodes * steps))

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

agent = Td3AgentReverb(
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
table_name = 'per_table'
reverb_table = reverb.Table(table_name,
                            max_size=buffer_size,
                            sampler=reverb.selectors.Prioritized(prioritization_exponent),
                            remover=reverb.selectors.Fifo(),
                            rate_limiter=reverb.rate_limiters.MinSize(1))

reverb_server = reverb.Server([reverb_table])

replay_buffer = reverb_replay_buffer.ReverbReplayBuffer(data_spec=agent.collect_data_spec,
                                                        table_name=table_name,
                                                        sequence_length=2,
                                                        local_server=reverb_server)

rb_observer = reverb_utils.ReverbAddTrajectoryObserver(replay_buffer.py_client,
                                                       table_name,
                                                       sequence_length=2,
                                                       stride_length=1)

# Creating a dataset
dataset = replay_buffer.as_dataset(
    sample_batch_size=batch_size,
    num_steps=2).prefetch(50)

experience_dataset_fn = lambda: dataset

"""Actor"""

eval_policy = py_tf_eager_policy.PyTFEagerPolicy(agent.policy, use_tf_function=True)
collect_policy = py_tf_eager_policy.PyTFEagerPolicy(agent.collect_policy, use_tf_function=True)

initial_collect_actor = actor.Actor(env_training,
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

"""Learner"""

saved_model_dir = os.path.join(tempdir, learner.POLICY_SAVED_MODEL_DIR)

learning_triggers = [
    triggers.PolicySavedModelTrigger(saved_model_dir, agent, train_step, interval=policy_save_interval),
    triggers.StepPerSecondLogTrigger(train_step, interval=1000), ]

agent_learner = ReverbLearnerPER(tempdir,
                                 train_step,
                                 agent,
                                 replay_buffer,
                                 initial_is_weight_exp=is_weights_initial_exponent,
                                 final_is_weight_exp=is_weights_final_exponent,
                                 is_weight_exp_steps=is_weight_exponent_steps,
                                 experience_dataset_fn=experience_dataset_fn,
                                 triggers=learning_triggers)

"""Treinamento do Agente"""

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

    print('episode = {0} Best solution on episode: {1} Return on episode: {2}'.format(ep, best_solution, ep_rew))

rb_observer.close()
reverb_server.stop()

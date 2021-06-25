import tensorflow as tf

from tf_agents.agents.ppo import ppo_clip_agent
from tf_agents.drivers import dynamic_episode_driver
from tf_agents.environments import parallel_py_environment
from tf_agents.environments import tf_py_environment
from tf_agents.networks import actor_distribution_network
from tf_agents.networks import value_network
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.system import system_multiprocessing as multiprocessing
from tf_agents.train.utils import train_utils
from tf_agents.utils import common

from tf_agents.environments.wrappers import TimeLimit
from functions.numpy_functions import *
from environments.py_function_environment import *
from utils.evaluation import evaluate_agent

""" Jupyter Notebook fix for absl
import sys
from absl import app

# Addresses `UnrecognizedFlagError: Unknown command line flag 'f'`
sys.argv = sys.argv[:1]

# `app.run` calls `sys.exit`
try:
  app.run(lambda argv: None)
except:
  pass
"""


def main(_):
    num_episodes = 2000
    collect_full_episodes_per_episode = 1
    replay_buffer_capacity = 251  # Per-environment
    num_epochs = 25
    num_parallel_environments = 15

    learning_rate = 3e-4
    actor_fc_layers = (256, 256)
    value_fc_layers = (256, 256)

    num_eval_episodes = 25
    eval_interval = 100
    steps = 250

    dims = 2
    function = Ackley()

    tf_env_eval = tf_py_environment.TFPyEnvironment(
        TimeLimit(env=PyFunctionEnvironment(function=function, dims=dims, clip_actions=True), duration=steps))

    def evaluate_current_policy(policy, num_episodes=num_eval_episodes):
        total_return = tf.Variable([0.])
        best_solutions = tf.Variable([0.])

        obj_value = tf.Variable([tf.float32.max])
        best_solution = tf.Variable([tf.float32.max])
        episode_return = tf.Variable([0.])

        def run_policy():
            for _ in range(num_episodes):
                time_step = tf_env_eval.reset()
                episode_return.assign([0.])
                best_solution.assign([tf.float32.max])

                while not time_step.is_last():
                    action_step = policy.action(time_step)
                    time_step = tf_env_eval.step(action_step.action)

                    episode_return.assign_add(time_step.reward)
                    obj_value.assign(tf_env_eval.get_info().objective_value)

                    if tf.math.less(obj_value, best_solution):
                        best_solution.assign(obj_value)

                best_solutions.assign_add(best_solution)
                total_return.assign_add(episode_return)

        run_policy()
        avg_return = tf.math.divide(total_return, num_episodes)
        avg_best_solution = tf.math.divide(best_solutions, num_episodes)
        return avg_best_solution, avg_return

    global_step = train_utils.create_train_step()

    tf_env_training = tf_py_environment.TFPyEnvironment(parallel_py_environment.ParallelPyEnvironment(
        [lambda: TimeLimit(env=PyFunctionEnvironment(function=function, dims=dims, clip_actions=True),
                           duration=steps)] * num_parallel_environments))

    actor_net = actor_distribution_network.ActorDistributionNetwork(tf_env_training.observation_spec(),
                                                                    tf_env_training.action_spec(),
                                                                    fc_layer_params=actor_fc_layers)
    value_net = value_network.ValueNetwork(tf_env_training.observation_spec(),
                                           fc_layer_params=value_fc_layers,
                                           activation_fn=tf.keras.activations.relu)

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    agent = ppo_clip_agent.PPOClipAgent(
        tf_env_training.time_step_spec(),
        tf_env_training.action_spec(),
        optimizer,
        actor_net=actor_net,
        value_net=value_net,
        entropy_regularization=0.0,
        importance_ratio_clipping=0.2,
        normalize_observations=False,
        normalize_rewards=False,
        use_gae=True,
        num_epochs=num_epochs,
        train_step_counter=global_step)

    agent.initialize()

    replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(agent.collect_data_spec,
                                                                   batch_size=num_parallel_environments,
                                                                   max_length=replay_buffer_capacity)

    collect_driver = dynamic_episode_driver.DynamicEpisodeDriver(tf_env_training,
                                                                 agent.collect_policy,
                                                                 observers=[replay_buffer.add_batch],
                                                                 num_episodes=collect_full_episodes_per_episode)

    collect_driver.run = common.function(collect_driver.run, autograph=False)
    agent.train = common.function(agent.train, autograph=False)

    for ep in range(num_episodes):
        collect_driver.run()

        experience = replay_buffer.gather_all()
        agent.train(experience=experience)
        replay_buffer.clear()

        print('episode {0}'.format(ep))
        if ep % eval_interval == 0:
            avg_best_sol, avg_return = evaluate_current_policy(policy=agent.policy)
            print('avg_best_solution: {0} avg_return: {1}'.format(avg_best_sol.numpy()[0], avg_return.numpy()[0]))

    print('---- Training finished ----')
    avg_best_sol, avg_return = evaluate_current_policy(policy=agent.policy, num_episodes=100)
    print('avg_best_solution: {0} avg_return: {1}'.format(avg_best_sol.numpy()[0], avg_return.numpy()[0]))
    evaluate_agent(tf_env_eval, agent.policy, function, dims, name_algorithm='PPO-Clip',
                   save_to_file=False)


if __name__ == '__main__':
    multiprocessing.enable_interactive_mode()
    multiprocessing.handle_main(main)

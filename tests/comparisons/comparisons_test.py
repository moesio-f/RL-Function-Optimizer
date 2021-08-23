"""Comparisons between trained agents."""

import os
from typing import List, NamedTuple

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import pandas as pd

from tf_agents.environments import tf_environment
from tf_agents.environments import tf_py_environment
from tf_agents.environments import wrappers
from tf_agents.policies import tf_policy, random_tf_policy

from environments import py_function_environment as py_fun_env
from functions import base as base_functions
from functions import numpy_functions as npf

MODELS_DIR = '../models'

class TrajectoryInfo(NamedTuple):
  algoritm: str
  improvement: np.ndarray
  best_position: np.ndarray
  best_iteration: int

def plot_trajectories(trajectories: [TrajectoryInfo],
                      function: base_functions.Function,
                      dims: int,
                      save_to_file: bool = True):
  # pylint: disable=missing-docstring
  _, ax = plt.subplots(figsize=(18.0, 10.0,))

  for traj in trajectories:
    ax.plot(traj.improvement,
            label='{0} | Best value: {1}'.format(traj.algoritm,
                                                 traj.improvement[-1]))

  ax.set(xlabel="Iterations",
         ylabel="Best objective value",
         title="{0} ({1}D)".format(function.name, str(dims)))

  ax.set_xscale('symlog', base=10)
  ax.set_yscale('log', base=10)
  ax.set_xlim(left=0)

  ax.legend(loc='upper right')
  ax.grid()

  plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
           rotation_mode="anchor")

  if save_to_file:
    plt.savefig(fname='{0}-{1}dims-comparison.png'.format(function.name, dims),
                bbox_inches='tight')
  else:
    plt.show()


def write_to_csv(trajectories: List[TrajectoryInfo],
                 function: base_functions.Function,
                 dims: int):
  file_name = f'{function.name}_{dims}_convergence.csv'
  data = pd.DataFrame({
    traj.algoritm: traj.improvement for traj in trajectories})
  data.to_csv(file_name, index_label='iteration')

def run_episode(tf_eval_env: tf_environment.TFEnvironment,
                policy: tf_policy.TFPolicy,
                trajectory_name: str,
                function: base_functions.Function) -> TrajectoryInfo:
  # pylint: disable=missing-docstring
  time_step = tf_eval_env.current_time_step()
  done = False

  best_pos = time_step.observation.numpy()[0]
  best_solution = function(best_pos)
  best_it = 0
  best_values_at_it = [best_solution]

  it = 0

  while not done:
    it += 1
    action_step = policy.action(time_step)
    time_step = tf_eval_env.step(action_step.action)

    pos = time_step.observation.numpy()[0]
    obj_value = function(pos)

    if obj_value < best_solution:
      best_solution = obj_value
      best_pos = pos
      best_it = it

    best_values_at_it.append(best_solution)
    done = time_step.is_last()

  return TrajectoryInfo(
    algoritm = trajectory_name,
    improvement = best_values_at_it,
    best_iteration = best_it,
    best_position = best_pos)


def run_rl_agent(policy: tf_policy.TFPolicy,
                 trajectory_name: str,
                 num_steps: int,
                 function: base_functions.Function,
                 dims: int,
                 initial_time_step,
                 clip_actions=False) -> TrajectoryInfo:
  # pylint: disable=missing-docstring
  # Como as policies já estão treinadas, não tem problema remover o clip da
  # ações. Relembrar que o ambiente não realiza checagem nas ações, apenas os
  # specs que são diferentes.
  env = py_fun_env.PyFunctionEnvironment(function, dims, clip_actions=clip_actions)
  env = wrappers.TimeLimit(env, duration=num_steps)

  tf_eval_env = tf_py_environment.TFPyEnvironment(environment=env)
  tf_eval_env._time_step = initial_time_step

  return run_episode(tf_eval_env=tf_eval_env,
                     policy=policy,
                     trajectory_name=trajectory_name,
                     function=function)


def get_average_trajectory(training_results: List[TrajectoryInfo]):
  """Return the average of results for one algoritm"""
  
  improvement = [train.improvement for train in training_results]
  improvement = np.mean(np.array(improvement, np.float32), axis=0)

  best_iter = [train.best_iteration for train in training_results]
  best_iter = np.mean(np.array(best_iter), axis=0).astype(np.int32)

  best_pos = [train.best_position for train in training_results]
  best_pos = np.mean(np.array(best_pos, np.float32), axis=0)

  return TrajectoryInfo(
    algoritm = training_results[0].algoritm,
    improvement = improvement,
    best_iteration = best_iter,
    best_position = best_pos)


if __name__ == '__main__':
  DIMS = 30
  STEPS = 500
  EPISODES = 100

  for FUNCTION in [npf.Sphere(), npf.Rosenbrock(), npf.SumSquares(), npf.Griewank(npf.base.Domain(-10, 10)),
                   npf.Ackley(), npf.Levy(), npf.Rastrigin(), npf.RotatedHyperEllipsoid(npf.base.Domain(-10,10))]:
    # Como as policies já estão treinadas, não tem problema remover o clip da
    # ações. Relembrar que o ambiente não realiza checagem nas ações, apenas
    # os specs que são diferentes.
    ENV = py_fun_env.PyFunctionEnvironment(FUNCTION, DIMS, clip_actions=True)
    ENV = wrappers.TimeLimit(ENV, duration=STEPS)
    TF_ENV = tf_py_environment.TFPyEnvironment(environment=ENV)

    reinforce_policy = tf.compat.v2.saved_model.load(
      os.path.join(MODELS_DIR, 'ReinforceAgent', str(DIMS)+'D', FUNCTION.name))
    reinforce_trajectories: [TrajectoryInfo] = []

    sac_policy = tf.compat.v2.saved_model.load(
      os.path.join(MODELS_DIR, 'SacAgent', str(DIMS)+'D', FUNCTION.name))
    sac_trajectories: [TrajectoryInfo] = []

    td3_policy = tf.compat.v2.saved_model.load(
      os.path.join(MODELS_DIR, 'Td3Agent', str(DIMS)+'D', FUNCTION.name))
    td3_trajectories: [TrajectoryInfo] = []

    ppo_policy = tf.compat.v2.saved_model.load(
      os.path.join(MODELS_DIR, 'PPOClipAgent', str(DIMS)+'D', FUNCTION.name))
    ppo_trajectories: [TrajectoryInfo] = []

    rand_policy = random_tf_policy.RandomTFPolicy(TF_ENV.time_step_spec(), TF_ENV.action_spec())
    rand_trajectories = []

    for _ in range(EPISODES):
      initial_ts = TF_ENV.reset()

      reinforce_trajectories.append(run_rl_agent(policy=reinforce_policy,
                                                 trajectory_name='REINFORCE',
                                                 num_steps=STEPS,
                                                 function=FUNCTION,
                                                 dims=DIMS,
                                                 initial_time_step=initial_ts))

      sac_trajectories.append(run_rl_agent(policy=sac_policy,
                                           trajectory_name='SAC',
                                           num_steps=STEPS,
                                           function=FUNCTION,
                                           dims=DIMS,
                                           initial_time_step=initial_ts))

      td3_trajectories.append(run_rl_agent(policy=td3_policy,
                                           trajectory_name='TD3',
                                           num_steps=STEPS,
                                           function=FUNCTION,
                                           dims=DIMS,
                                           initial_time_step=initial_ts))

      ppo_trajectories.append(run_rl_agent(policy=ppo_policy,
                                           trajectory_name='PPO',
                                           num_steps=STEPS,
                                           function=FUNCTION,
                                           dims=DIMS,
                                           initial_time_step=initial_ts))

      rand_trajectories.append(run_rl_agent(policy=rand_policy,
                                           trajectory_name='RANDOM',
                                           num_steps=STEPS,
                                           function=FUNCTION,
                                           dims=DIMS,
                                           initial_time_step=initial_ts,
                                           clip_actions=True))

    avg_reinforce = get_average_trajectory(reinforce_trajectories)
    avg_sac = get_average_trajectory(sac_trajectories)
    avg_td3 = get_average_trajectory(td3_trajectories)
    avg_ppo = get_average_trajectory(ppo_trajectories)
    avg_rand = get_average_trajectory(rand_trajectories)

    plot_trajectories([avg_reinforce, avg_sac, avg_td3, avg_ppo, avg_rand],
                      function=FUNCTION, dims=DIMS)
    write_to_csv([avg_reinforce, avg_sac, avg_td3, avg_ppo, avg_rand],
                 function=FUNCTION, dims=DIMS)

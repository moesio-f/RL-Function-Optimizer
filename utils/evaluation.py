"""Utility methods for agent evaluation."""

import matplotlib.pyplot as plt
import numpy as np
from tf_agents.environments.tf_environment import TFEnvironment
from tf_agents.policies.tf_policy import TFPolicy

from functions.base import Function


def evaluate_agent(eval_env: TFEnvironment, policy_eval: TFPolicy,
                   function: Function,
                   dims, name_algorithm, name_policy=None,
                   save_to_file=False, verbose=False,
                   show_all_trajectories=False,
                   episodes=50):
  if name_policy is None:
    name_policy = policy_eval.__class__.__name__

  trajectories = []
  best_trajectory = [np.finfo(np.float32).max]
  best_it = None
  best_pos = None

  for _ in range(episodes):
    time_step = eval_env.reset()
    info = eval_env.get_info()

    best_solution_ep = info.objective_value[0]
    best_it_ep = 0
    best_pos_ep = info.position[0]

    trajectory = [best_solution_ep]
    it = 0

    while not time_step.is_last():
      it += 1
      action_step = policy_eval.action(time_step)
      time_step = eval_env.step(action_step.action)
      info = eval_env.get_info()

      obj_value = info.objective_value[0]

      if obj_value < best_solution_ep:
        best_solution_ep = obj_value
        best_it_ep = it
        best_pos_ep = info.position[0]

      trajectory.append(best_solution_ep)

    if trajectory[-1] < best_trajectory[-1]:
      best_trajectory = trajectory
      best_pos = best_pos_ep
      best_it = best_it_ep

    trajectories.append(trajectory)

  mean = np.mean(trajectories, axis=0)

  _, ax = plt.subplots(figsize=(18.0, 10.0,))

  if show_all_trajectories:
    for traj in trajectories:
      ax.plot(traj, '--c', alpha=0.4)

  ax.plot(mean, 'r', label='Best mean value: {0}'.format(mean[-1]))
  ax.plot(best_trajectory, 'g',
          label='Best value: {0}'.format(best_trajectory[-1]))

  ax.set(xlabel="Iterations\nBest solution at: {0}".format(best_pos),
         ylabel="Best objective value",
         title="{0} on {1} ({2} Dims) [{3}]".format(name_algorithm,
                                                    function.name,
                                                    dims,
                                                    name_policy))

  ax.set_xscale('symlog', base=2)
  ax.set_xlim(left=0)

  ax.legend()
  ax.grid()

  plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
           rotation_mode="anchor")
  if save_to_file:
    plt.savefig(fname='{0}-{1}dims-{2}.png'.format(function.name,
                                                   dims,
                                                   name_policy),
                bbox_inches='tight')
  plt.show()

  if verbose:
    print('best_solution: ', best_trajectory[-1])
    print('found at it: ', best_it)
    print('at position: ', best_pos)

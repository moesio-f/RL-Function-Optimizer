import matplotlib.pyplot as plt
import numpy as np
from functions.function import Function
from tf_agents.policies.tf_policy import TFPolicy
from tf_agents.environments.tf_environment import TFEnvironment
from utils.render.function_drawer import FunctionDrawer


def evaluate_agent(eval_env: TFEnvironment, policy_eval: TFPolicy, function: Function,
                   dims, name_algorithm, name_policy=None,
                   save_to_file=False, verbose=False, episodes=50):
    if name_policy is None:
        name_policy = policy_eval.__class__.__name__
    
    trajectories = []
    best_trajectory = [np.finfo(np.float32).max]

    for ep in range(episodes):
        time_step = eval_env.reset()

        pos = time_step.observation.numpy()[0]
        best_solution = function(pos)

        trajectory = [best_solution]
        best_it = 0
        it = 0

        while not time_step.is_last():
            it += 1
            action_step = policy_eval.action(time_step)
            time_step = eval_env.step(action_step.action)

            obj_value = -time_step.reward.numpy()[0]

            if obj_value < best_solution:
                best_solution = obj_value
                pos = time_step.observation.numpy()[0]
                best_it = it

            trajectory.append(best_solution)
        
        if trajectory[-1] < best_trajectory[-1]:
            best_trajectory = trajectory

        trajectories.append(trajectory)

    mean = np.mean(trajectories, axis=0)

    fig, ax = plt.subplots(figsize=(18.0, 10.0,))
    for traj in trajectories:
        ax.plot(traj, '--c', alpha=0.4)
    ax.plot(mean, 'r', label='Best mean value: {0}'.format(mean[-1]))
    ax.plot(best_trajectory, 'g', label='Best value: {0}'.format(best_trajectory[-1]))

    ax.set(xlabel="Iterations\nBest solution at: {0}".format(pos),
           ylabel="Best objective value",
           title="{0} on {1} ({2} Dims) [{3}]".format(name_algorithm,
                                                      function.name,
                                                      dims,
                                                      name_policy))

    ax.set_xscale('symlog', base=2)
    ax.set_xlim(left=0)

    ax.legend()
    ax.grid()

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    if save_to_file:
        plt.savefig(fname='{0}-{1}dims-{2}.png'.format(function.name,
                                                       dims,
                                                       name_policy),
                    bbox_inches='tight')
    plt.show()

    if verbose:
        print('best_solution: ', best_solution)
        print('found at it: ', best_it)
        print('at position: ', pos)

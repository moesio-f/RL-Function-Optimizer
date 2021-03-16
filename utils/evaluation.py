import matplotlib.pyplot as plt
import numpy as np
from functions.function import Function
from tf_agents.policies.tf_policy import TFPolicy
from tf_agents.environments.tf_environment import TFEnvironment


def evaluate_agent(eval_env: TFEnvironment, policy_eval: TFPolicy, function: Function,
                   dims, name_algorithm, name_policy=None,
                   save_to_file=False, verbose=False):
    if name_policy is None:
        name_policy = policy_eval.__class__.__name__

    time_step = eval_env.reset()

    pos = time_step.observation.numpy()[0]
    best_solution = function(pos)

    best_solution_at_it = [best_solution]
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

        best_solution_at_it.append(best_solution)

    fig, ax = plt.subplots(figsize=(18.0, 10.0,))
    ax.plot(range(len(best_solution_at_it)), best_solution_at_it,
            label='Best value found: {0}'.format(best_solution))
    ax.set(xlabel="Iterations\nBest solution at: {0}".format(pos),
           ylabel="Best objective value",
           title="{0} on {1} ({2} Dims) [{3}]".format(name_algorithm,
                                                      function.name,
                                                      dims,
                                                      name_policy))

    x_ticks = np.arange(0, len(best_solution_at_it), step=50.0)
    x_labels = ['{:.0f}'.format(val) for val in x_ticks]

    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_labels)
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

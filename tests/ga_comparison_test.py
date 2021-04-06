import os
import tensorflow as tf
from collections import namedtuple
import matplotlib.pyplot as plt

from tf_agents.environments.tf_py_environment import TFPyEnvironment
from tf_agents.environments.wrappers import TimeLimit
from tf_agents.environments.tf_environment import TFEnvironment
from tf_agents.policies.tf_policy import TFPolicy
from tf_agents.policies.random_tf_policy import RandomTFPolicy

from environments.py_function_environment import PyFunctionEnvironment
from environments.py_function_environment_unbounded import PyFunctionEnvironmentUnbounded

from deap import base
from deap import creator
from deap import tools
from deap import algorithms

from functions.numpy_functions import *


class Trajectory(namedtuple('Trajectory', ('list_best_values', 'name', 'best_iteration', 'best_position'))):
    pass


def plot_trajectories(trajectories: [Trajectory], save_to_file: bool = True):
    fig, ax = plt.subplots(figsize=(18.0, 10.0,))

    for traj in trajectories:
        ax.plot(traj.list_best_values, label='{0} | Best value: {1}'.format(traj.name, traj.list_best_values[-1]))

    ax.set(xlabel="Iterations/Generations",
           ylabel="Best objective value",
           title="{0} ({1}D): eaSimple vs TD3-IG vs Random".format(function.name, dims))

    ax.set_xscale('symlog', base=2)
    ax.set_xlim(left=0)

    ax.legend(loc='upper right')
    ax.grid()

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    if save_to_file:
        plt.savefig(fname='{0}-{1}dims-comparison.png'.format(function.name, dims),
                    bbox_inches='tight')
    plt.show()


def run_episode(tf_eval_env: TFEnvironment, policy: TFPolicy, trajectory_name: str) -> Trajectory:
    time_step = tf_eval_env.reset()
    info = tf_eval_env.get_info()
    done = False

    best_solution = info.objective_value[0]
    best_pos = info.position[0]
    best_it = 0
    best_values_at_it = [best_solution]

    it = 0

    while not done:
        it += 1
        action_step = policy.action(time_step)
        time_step = tf_eval_env.step(action_step.action)
        info = tf_eval_env.get_info()

        obj_value = info.objective_value[0]

        if obj_value < best_solution:
            best_solution = obj_value
            best_pos = info.position[0]
            best_it = it

        best_values_at_it.append(best_solution)
        done = time_step.is_last()

    return Trajectory(list_best_values=best_values_at_it,
                      name=trajectory_name,
                      best_iteration=best_it,
                      best_position=best_pos)


def create_ga_env() -> base.Toolbox:
    def evalFunction(individual):
        return function(np.array(individual, dtype=np.float32)),

    creator.create("FitnessMinimum", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMinimum)

    toolbox_ = base.Toolbox()

    toolbox_.register("attr_position", np.random.uniform, function.domain.min, function.domain.max)

    toolbox_.register("individual", tools.initRepeat, creator.Individual, toolbox_.attr_position, dims)
    toolbox_.register("population", tools.initRepeat, list, toolbox_.individual)

    toolbox_.register("evaluate", evalFunction)
    toolbox_.register("mate", tools.cxTwoPoint)
    toolbox_.register("mutate", tools.mutFlipBit, indpb=0.05)
    toolbox_.register("select", tools.selTournament, tournsize=3)

    return toolbox_


def run_td3_ig(num_steps=2000, policy_dir_name: str = 'policy') -> Trajectory:
    policy_dir = os.path.join(ROOT_DIR, policy_dir_name)
    policy = tf.compat.v2.saved_model.load(policy_dir)

    env = PyFunctionEnvironmentUnbounded(function, dims)
    env = TimeLimit(env, duration=num_steps)
    tf_eval_env = TFPyEnvironment(environment=env)

    return run_episode(tf_eval_env=tf_eval_env, policy=policy,
                       trajectory_name='TD3-IG')


def run_random_policy(num_steps=2000) -> Trajectory:
    env = PyFunctionEnvironmentUnbounded(function, dims)
    env = TimeLimit(env, duration=num_steps)
    tf_eval_env = TFPyEnvironment(environment=env)
    policy = RandomTFPolicy(time_step_spec=tf_eval_env.time_step_spec(),
                            action_spec=tf_eval_env.action_spec())

    return run_episode(tf_eval_env=tf_eval_env, policy=policy,
                       trajectory_name='Random')


def run_random_policy_bounded_actions(num_steps=2000) -> Trajectory:
    env = PyFunctionEnvironment(function, dims)
    env = TimeLimit(env, duration=num_steps)
    tf_eval_env = TFPyEnvironment(environment=env)
    policy = RandomTFPolicy(time_step_spec=tf_eval_env.time_step_spec(),
                            action_spec=tf_eval_env.action_spec())

    return run_episode(tf_eval_env=tf_eval_env, policy=policy,
                       trajectory_name='Random-Bounded-Actions')


def run_ea_simple(_toolbox: base.Toolbox, num_ind=300, num_gens=2000) -> Trajectory:
    pop = _toolbox.population(n=num_ind)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("best", np.min)

    pop, log = algorithms.eaSimple(pop, _toolbox, cxpb=0.5, mutpb=0.2, ngen=num_gens,
                                   stats=stats, halloffame=hof, verbose=False)

    objective_values = log.select('best')

    best_val = objective_values[0]
    best_values_at_it = [best_val]
    best_gen = 0

    i = 0
    for val in objective_values[1:]:
        i += 1
        if val < best_val:
            best_val = val
            best_gen = i
        best_values_at_it.append(best_val)

    return Trajectory(list_best_values=best_values_at_it,
                      name='eaSimple-{0}-individuals'.format(num_ind),
                      best_iteration=best_gen,
                      best_position=None)


function = Ackley()
dims = 20
steps = 2000

ROOT_DIR = os.getcwd()
tb = create_ga_env()

plot_trajectories([run_td3_ig(num_steps=steps),
                   run_random_policy(num_steps=steps),
                   run_random_policy_bounded_actions(num_steps=steps),
                   run_ea_simple(_toolbox=tb, num_gens=steps, num_ind=300),
                   run_ea_simple(_toolbox=tb, num_gens=steps, num_ind=1)])

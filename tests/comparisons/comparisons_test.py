import os
import tensorflow as tf
from collections import namedtuple
import csv
import matplotlib.pyplot as plt
from tf_agents.environments.tf_py_environment import TFPyEnvironment
from tf_agents.environments.wrappers import TimeLimit
from tf_agents.environments.tf_environment import TFEnvironment
from tf_agents.policies.tf_policy import TFPolicy
from environments.py_function_environment import PyFunctionEnvironment
from functions.numpy_functions import *
from functions.function import Function

MODELS_DIR = '../models'


class Trajectory(namedtuple('Trajectory', ('list_best_values', 'name', 'best_iteration', 'best_position'))):
    pass


def plot_trajectories(trajectories: [Trajectory], function: Function, dims: int, save_to_file: bool = True):
    fig, ax = plt.subplots(figsize=(18.0, 10.0,))

    for traj in trajectories:
        ax.plot(traj.list_best_values, label='{0} | Best value: {1}'.format(traj.name, traj.list_best_values[-1]))

    ax.set(xlabel="Iterations",
           ylabel="Best objective value",
           title="{0} ({1}D)".format(function.name, str(dims)))

    ax.set_xscale('symlog', base=10)
    ax.set_yscale('log', base=10)
    ax.set_xlim(left=0)

    ax.legend(loc='upper right')
    ax.grid()

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    if save_to_file:
        plt.savefig(fname='{0}-{1}dims-comparison.png'.format(function.name, dims),
                    bbox_inches='tight')
    plt.show()


def write_to_csv(trajectories: [Trajectory], function: Function, dims: int):
    for trajectory in trajectories:
        with open(f'{function.name}_{str(dims)}_{trajectory.name}_convergence.csv', 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['iteration', 'mean_objective_value'])
            for it, mean_obj_value in enumerate(trajectory.list_best_values):
                writer.writerow([it, mean_obj_value])


def run_episode(tf_eval_env: TFEnvironment,
                policy: TFPolicy,
                trajectory_name: str,
                function: Function) -> Trajectory:
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

    return Trajectory(list_best_values=best_values_at_it,
                      name=trajectory_name,
                      best_iteration=best_it,
                      best_position=best_pos)


def run_rl_agent(policy: TFPolicy,
                 trajectory_name: str,
                 num_steps: int,
                 function: Function,
                 dims: int,
                 initial_time_step) -> Trajectory:
    # Como as policies já estão treinadas, não tem problema remover o clip da ações.
    # Relembrar que o ambiente não realiza checagem nas ações, apenas os specs que são diferentes.
    env = PyFunctionEnvironment(function, dims, clip_actions=False)
    env = TimeLimit(env, duration=num_steps)

    tf_eval_env = TFPyEnvironment(environment=env)
    tf_eval_env._time_step = initial_time_step

    return run_episode(tf_eval_env=tf_eval_env,
                       policy=policy,
                       trajectory_name=trajectory_name,
                       function=function)


def get_average_trajectory(trajectories: [Trajectory]):
    best_values = []
    best_iterations = []
    best_positions = []
    name = trajectories[0].name

    for traj in trajectories:
        best_values.append(traj.list_best_values)
        best_iterations.append(traj.best_iteration)
        best_positions.append(traj.best_position)

    return Trajectory(list_best_values=np.mean(np.array(best_values, dtype=np.float32), axis=0),
                      best_iteration=np.mean(np.array(best_iterations, dtype=np.int32), axis=0).astype(np.int32),
                      best_position=np.mean(np.array(best_positions, dtype=np.float32), axis=0),
                      name=name)


if __name__ == '__main__':
    DIMS = 30
    STEPS = 500
    EPISODES = 100

    for FUNCTION in [Sphere(), Ackley()]:
        # Como as policies já estão treinadas, não tem problema remover o clip da ações.
        # Relembrar que o ambiente não realiza checagem nas ações, apenas os specs que são diferentes.
        ENV = PyFunctionEnvironment(FUNCTION, DIMS, clip_actions=False)
        ENV = TimeLimit(ENV, duration=STEPS)
        TF_ENV = TFPyEnvironment(environment=ENV)

        reinforce_policy = tf.compat.v2.saved_model.load(
            os.path.join(MODELS_DIR, 'REINFORCE-BL/' + f'{DIMS}D/' + FUNCTION.name))
        reinforce_trajectories: [Trajectory] = []

        sac_policy = tf.compat.v2.saved_model.load(os.path.join(MODELS_DIR, 'SAC-AAT/' + f'{DIMS}D/' + FUNCTION.name))
        sac_trajectories: [Trajectory] = []

        td3_policy = tf.compat.v2.saved_model.load(os.path.join(MODELS_DIR, 'TD3/' + f'{DIMS}D/' + FUNCTION.name))
        td3_trajectories: [Trajectory] = []

        td3_ig_policy = tf.compat.v2.saved_model.load(os.path.join(MODELS_DIR, 'TD3-IG/' + f'{DIMS}D/' + FUNCTION.name))
        td3_ig_trajectories: [Trajectory] = []

        ppo_policy = tf.compat.v2.saved_model.load(os.path.join(MODELS_DIR, 'PPO-CLIP/' + f'{DIMS}D/' + FUNCTION.name))
        ppo_trajectories: [Trajectory] = []

        for _ in range(EPISODES):
            initial_time_step = TF_ENV.reset()

            reinforce_trajectories.append(run_rl_agent(policy=reinforce_policy,
                                                       trajectory_name='REINFORCE',
                                                       num_steps=STEPS,
                                                       function=FUNCTION,
                                                       dims=DIMS,
                                                       initial_time_step=initial_time_step))

            sac_trajectories.append(run_rl_agent(policy=sac_policy,
                                                 trajectory_name='SAC',
                                                 num_steps=STEPS,
                                                 function=FUNCTION,
                                                 dims=DIMS,
                                                 initial_time_step=initial_time_step))

            td3_trajectories.append(run_rl_agent(policy=td3_policy,
                                                 trajectory_name='TD3',
                                                 num_steps=STEPS,
                                                 function=FUNCTION,
                                                 dims=DIMS,
                                                 initial_time_step=initial_time_step))

            td3_ig_trajectories.append(run_rl_agent(policy=td3_ig_policy,
                                                    trajectory_name='TD3-IG',
                                                    num_steps=STEPS,
                                                    function=FUNCTION,
                                                    dims=DIMS,
                                                    initial_time_step=initial_time_step))

            ppo_trajectories.append(run_rl_agent(policy=td3_ig_policy,
                                                 trajectory_name='PPO',
                                                 num_steps=STEPS,
                                                 function=FUNCTION,
                                                 dims=DIMS,
                                                 initial_time_step=initial_time_step))

        avg_reinforce = get_average_trajectory(reinforce_trajectories)
        avg_sac = get_average_trajectory(sac_trajectories)
        avg_td3 = get_average_trajectory(td3_trajectories)
        avg_td3_ig = get_average_trajectory(td3_ig_trajectories)
        avg_ppo = get_average_trajectory(ppo_trajectories)

        plot_trajectories([avg_reinforce, avg_sac, avg_td3, avg_td3_ig, avg_ppo], function=FUNCTION, dims=DIMS)
        write_to_csv([avg_reinforce, avg_sac, avg_td3, avg_td3_ig, avg_ppo], function=FUNCTION, dims=DIMS)

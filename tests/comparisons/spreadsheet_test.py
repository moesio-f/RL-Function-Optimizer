import os
import csv
import tensorflow as tf
import numpy as np
from collections import namedtuple

from tf_agents.environments.tf_py_environment import TFPyEnvironment
from tf_agents.environments.wrappers import TimeLimit

from environments.py_function_environment_unbounded import PyFunctionEnvironmentUnbounded
import functions.numpy_functions as np_functions


# Representa uma função descrita em sua totalidade
class FunctionDescription(namedtuple('FunctionDescription',
                                     ('function',
                                      'dims',
                                      'global_minimum'))):
    pass


# Representa um par de uma função totalmente descrita e uma
# policy que deve minimizar essa função
class PolicyFunctionPair(namedtuple('PolicyFunctionPair',
                                    ('policy',
                                     'function_description',
                                     'num_learning_episodes'))):
    pass


# Representa os dados obtidos depois de testar
# uma dada policy numa dada função (Ou seja, um PolicyFunctionPair).
class PolicyEvaluationData(namedtuple('PolicyEvaluationData',
                                      ('policy_function_pair_evaluated',
                                       'average_best_solution',
                                       'average_best_value_time',
                                       'final_best_values_stddev'))):
    pass


def get_all_functions_descriptions(dims: int) -> [FunctionDescription]:
    functions = np_functions.list_all_functions()
    functions_descriptions: [FunctionDescription] = []

    # Ackley 0
    # Rastrigin 0
    # Griewank 0
    # Levy 0
    # Sphere 0
    # SumSquares 0
    # Rotated 0
    # Rosenbrock 0
    # DixonPrice 0
    # Zakharov 0

    for function in functions:
        functions_descriptions.append(FunctionDescription(function=function,
                                                          dims=dims,
                                                          global_minimum=0.0))

    return functions_descriptions


def load_policies_and_functions(functions_descriptions: [FunctionDescription], algorithm: str, dims: int,
                                num_learning_episodes: dict) -> [PolicyFunctionPair]:
    root_dir = os.path.join(os.getcwd(), "../models")
    root_dir = os.path.join(root_dir, f'{algorithm}')
    root_dir = os.path.join(root_dir, f'{str(dims)}D')

    pairs: [PolicyFunctionPair] = []

    for function_desc in functions_descriptions:
        policy_dir = os.path.join(root_dir, function_desc.function.name)
        if os.path.exists(policy_dir):
            policy = tf.compat.v2.saved_model.load(policy_dir)
            pairs.append(PolicyFunctionPair(policy=policy,
                                            function_description=function_desc,
                                            num_learning_episodes=num_learning_episodes[function_desc.function.name]))
        else:
            print('Não foi possível encontrar uma policy para a função {0}.'.format(function_desc.function.name))
            print('{0} não foi incluído na lista.'.format(function_desc.function.name))

    return pairs


def evaluate_policies(policies_functions_pair: [PolicyFunctionPair],
                      steps=500, episodes=100, create_log=False) -> [PolicyEvaluationData]:
    policies_evaluation_data: [PolicyEvaluationData] = []

    def evaluate_single_policy(policy_function_pair: PolicyFunctionPair) -> PolicyEvaluationData:
        nonlocal steps
        nonlocal episodes

        env = PyFunctionEnvironmentUnbounded(function=policy_function_pair.function_description.function,
                                             dims=policy_function_pair.function_description.dims)
        env = TimeLimit(env=env, duration=steps)
        tf_env = TFPyEnvironment(environment=env)

        policy = policy_function_pair.policy

        best_solutions: [np.float32] = []
        best_solutions_iterations: [int] = []

        for ep in range(episodes):
            time_step = tf_env.reset()
            info = tf_env.get_info()
            it = 0

            best_solution_ep = info.objective_value[0]
            best_it_ep = it

            while not time_step.is_last():
                it += 1
                action_step = policy.action(time_step)
                time_step = tf_env.step(action_step.action)
                info = tf_env.get_info()

                obj_value = info.objective_value[0]

                if obj_value < best_solution_ep:
                    best_solution_ep = obj_value
                    best_it_ep = it

            best_solutions.append(best_solution_ep)
            best_solutions_iterations.append(best_it_ep)

        avg_best_solution = np.mean(best_solutions)
        avg_best_solution_time = np.rint(np.mean(best_solutions_iterations)).astype(np.int32)
        stddev_best_solutions = np.std(best_solutions)

        return PolicyEvaluationData(policy_function_pair_evaluated=policy_function_pair,
                                    average_best_solution=avg_best_solution,
                                    average_best_value_time=avg_best_solution_time,
                                    final_best_values_stddev=stddev_best_solutions)

    for pair in policies_functions_pair:
        policies_evaluation_data.append(evaluate_single_policy(pair))

    if create_log:
        with open('log.txt', 'w') as file:
            file.write('\n\nInformações gerais sobre os testes realizados para obter os dados:\n')
            file.write('\tCada função foi executada por {0} episódios.\n'.format(episodes))
            file.write('\tCada episódios permitiu o agente tomar {0} passos (steps).\n'.format(steps))
            file.write('Informações sobre cada função executada e testada:\n')

            for pol_func_pair in policies_functions_pair:
                function_desc = pol_func_pair.function_description
                file.write('\tFunção: {0} - Dimensões: {1} - Mínimo Global: {2}\n'.format(function_desc.function.name,
                                                                                          function_desc.dims,
                                                                                          function_desc.global_minimum))

    return policies_evaluation_data


def write_to_csv(policies_evaluation_data: [PolicyEvaluationData],
                 file_name: str):
    with open(file_name, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Function',
                         'Dimensions',
                         'Global Minimum',
                         'Number Learning Episodes',
                         'Average Best Solution',
                         'Average Best Solution Time (Iterations)',
                         'Stddev of best solutions'])
        for pol_eval_data in policies_evaluation_data:
            pol_function_pair = pol_eval_data.policy_function_pair_evaluated
            function_desc = pol_function_pair.function_description
            writer.writerow([function_desc.function.name,
                             function_desc.dims,
                             function_desc.global_minimum,
                             pol_function_pair.num_learning_episodes,
                             pol_eval_data.average_best_solution,
                             pol_eval_data.average_best_value_time,
                             pol_eval_data.final_best_values_stddev])


if __name__ == "__main__":
    dims = 30
    episodes = 100

    functions_descriptions = get_all_functions_descriptions(dims=dims)
    pol_func_pairs = load_policies_and_functions(functions_descriptions, algorithm='REINFORCE-BL', dims=dims,
                                                 num_learning_episodes={'Ackley': 2000,
                                                                        'Sphere': 2000})
    pol_eval_data = evaluate_policies(pol_func_pairs, episodes=episodes, create_log=True)
    write_to_csv(pol_eval_data, file_name='reinforce_data.csv')

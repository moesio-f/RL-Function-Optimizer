"""Policies' statistics test."""

import csv
import os
from collections import namedtuple

import numpy as np
import tensorflow as tf
from tf_agents.environments import tf_py_environment
from tf_agents.environments import wrappers

from src.single_agent.environments import py_function_environment as py_fun_env
from src.functions import numpy_functions as npf

ROOT_DIR = '../models/after-update'  # Alterar para a pasta com os modelos.


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
                                      ('policy_function_pair',
                                       'average_best_solution',
                                       'average_best_solution_time',
                                       'best_solutions_stddev'))):
  pass


def get_all_functions_descriptions(dims: int) -> [FunctionDescription]:
  # pylint: disable=missing-docstring
  functions = npf.list_all_functions()
  functions_desc: [FunctionDescription] = []

  # Ackley 0, Rastrigin 0, Griewank 0, Levy 0, Sphere 0, SumSquares 0
  # Rotated 0, Rosenbrock 0, DixonPrice 0, Zakharov 0

  for function in functions:
    functions_desc.append(FunctionDescription(function=function,
                                              dims=dims,
                                              global_minimum=0.0))

  return functions_desc


def load_policies_and_functions(functions_desc: [FunctionDescription],
                                algorithm: str,
                                dims: int,
                                num_learning_episodes: dict) -> \
      [PolicyFunctionPair]:
  # pylint: disable=missing-docstring
  root_dir = os.path.join(ROOT_DIR, f'{algorithm}')
  root_dir = os.path.join(root_dir, f'{str(dims)}D')

  pairs: [PolicyFunctionPair] = []

  for function_desc in functions_desc:
    policy_dir = os.path.join(root_dir, function_desc.function.name)
    if os.path.exists(policy_dir) \
          and function_desc.function.name in num_learning_episodes:
      policy = tf.compat.v2.saved_model.load(policy_dir)
      pairs.append(PolicyFunctionPair(policy=policy,
                                      function_description=function_desc,
                                      num_learning_episodes=
                                      num_learning_episodes[
                                        function_desc.function.name]))
    else:
      print('{0} não foi incluído na lista.'.format(
        function_desc.function.name))

  return pairs


def evaluate_policies(policies_functions_pair: [PolicyFunctionPair],
                      steps=500,
                      episodes=100) -> [PolicyEvaluationData]:
  # pylint: disable=missing-docstring
  policies_evaluation_data: [PolicyEvaluationData] = []

  def evaluate_single_policy(
        policy_function_pair: PolicyFunctionPair) -> PolicyEvaluationData:
    nonlocal steps
    nonlocal episodes

    # Como as policies já estão treinadas, não tem problema remover o clip da
    # ações. Relembrar que o ambiente não realiza checagem nas ações, apenas
    # os specs que são diferentes.
    env = py_fun_env.PyFunctionEnvironment(
      function=policy_function_pair.function_description.function,
      dims=policy_function_pair.function_description.dims, clip_actions=False)
    env = wrappers.TimeLimit(env=env, duration=steps)
    tf_env = tf_py_environment.TFPyEnvironment(environment=env)

    policy = policy_function_pair.policy

    best_solutions: [np.float32] = []
    best_solutions_iterations: [int] = []

    for _ in range(episodes):
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
    avg_best_solution_time = np.rint(np.mean(best_solutions_iterations)).astype(
      np.int32)
    stddev_best_solutions = np.std(best_solutions)

    return PolicyEvaluationData(
      policy_function_pair=policy_function_pair,
      average_best_solution=avg_best_solution,
      average_best_solution_time=avg_best_solution_time,
      best_solutions_stddev=stddev_best_solutions)

  for pair in policies_functions_pair:
    policies_evaluation_data.append(evaluate_single_policy(pair))

  return policies_evaluation_data


def write_to_csv(policies_evaluation_data: [PolicyEvaluationData],
                 file_name: str):
  # pylint: disable=missing-docstring
  with open(file_name, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Function',
                     'Dimensions',
                     'Global Minimum',
                     'Number Learning Episodes',
                     'Average Best Solution',
                     'Average Best Solution Time (Iterations)',
                     'Stddev of best solutions'])
    for pol_data in policies_evaluation_data:
      pol_function_pair = pol_data.policy_function_pair
      function_desc = pol_function_pair.function_description
      writer.writerow([function_desc.function.name,
                       function_desc.dims,
                       function_desc.global_minimum,
                       pol_function_pair.num_learning_episodes,
                       pol_data.average_best_solution,
                       pol_data.average_best_solution_time,
                       pol_data.best_solutions_stddev])


if __name__ == "__main__":
  DIMS = 30
  EPISODES = 100

  functions_descriptions = get_all_functions_descriptions(dims=DIMS)
  pol_func_pairs = load_policies_and_functions(functions_descriptions,
                                               algorithm='TD3',
                                               dims=DIMS,
                                               num_learning_episodes={
                                                 'Ackley': 2000,
                                                 'Griewank': 2000,
                                                 'Levy': 2000,
                                                 'Rastrigin': 2000,
                                                 'Rosenbrock': 2000,
                                                 'RotatedHyperEllipsoid': 2000,
                                                 'Sphere': 2000,
                                                 'SumSquares': 2000})
  pol_eval_data = evaluate_policies(pol_func_pairs,
                                    episodes=EPISODES)
  write_to_csv(pol_eval_data,
               file_name='td3_data.csv')

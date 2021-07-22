"""MORELA (MOdified REinforcement Learning Algorithm) tests."""

import copy

import numpy as np
from numpy import random as rand

import functions.numpy_functions as npf


class TableRow:
  """Represents a row in the environment table."""

  def __init__(self, array: np.ndarray, fitness: np.float32):
    self._array = array
    self._fitness = fitness

  @property
  def array(self):
    return self._array

  @array.setter
  def array(self, value):
    self._array = value

  @property
  def fitness(self):
    return self._fitness

  @fitness.setter
  def fitness(self, value):
    self._fitness = value


if __name__ == '__main__':
  rng = rand.default_rng()

  function = npf.Ackley()
  dims = 30
  max_iterations = 500
  learning_rate = 0.8
  discount = 0.2
  env_size = 20
  beta = rng.uniform(low=function.domain.min, high=function.domain.max,
                     size=dims).astype(np.float32)

  original_environment: [TableRow] = []
  last_step: TableRow
  best_solution: TableRow = TableRow(array=beta.copy(),
                                     fitness=np.finfo(np.float32).max)

  # Initializing original environment
  for _ in range(env_size):
    random_vector = rng.uniform(low=function.domain.min,
                                high=function.domain.max,
                                size=dims).astype(np.float32)
    original_environment.append(TableRow(array=random_vector,
                                         fitness=function(random_vector)))

  # Finding the best solution
  best_solution = min(original_environment, key=lambda x: x.fitness)

  for t in range(2, max_iterations + 1):
    sub_envs: [TableRow] = []

    # Generating sub-environment
    for _ in range(env_size):
      sub_env_arr = rng.uniform(low=best_solution.array - beta,
                                high=best_solution.array + beta)
      sub_env_fit = function(sub_env_arr)
      sub_envs.append(TableRow(array=sub_env_arr, fitness=sub_env_fit))

    # At this point, we have: original_env, best_sol (Last it), new sub_env
    # In other words, original_env_{t}, best_sol_{t-1}, new_sub_env_{t}

    # Thus, we must sort original_env_{t} and new_sub_env_{t} Then,
    # we compare both and if worst(original_env_{t}) > best(new_sub_env_{t})
    # then we delete worst(original_env_{t}) from original_env_{t} and add
    # best(new_sub_env_{t}) to original_env_{t}

    original_environment = sorted(original_environment, key=lambda x: x.fitness)
    sub_envs = sorted(sub_envs, key=lambda x: x.fitness)

    if sub_envs[0].fitness < original_environment[-1].fitness:
      original_environment[-1] = copy.deepcopy(sub_envs[0])

    # Update best_sol_{t-1} to best_sol_{t}
    best_solution = min(original_environment, key=lambda x: x.fitness)

    # Update original environment
    for row in original_environment:
      reward = np.nan_to_num((best_solution.array - row.array) / row.array)
      row.array = (1 - learning_rate) * row.array + learning_rate * (
            reward + discount * best_solution.array)
      row.fitness = function(row.array)

    beta = beta * 0.99
    print('iteration:', t, 'Best solution:', best_solution.fitness)

  print('Position:', best_solution.array)

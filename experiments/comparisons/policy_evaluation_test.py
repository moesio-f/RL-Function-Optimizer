"""Policy evaluation tests."""

import os

import tensorflow as tf
from tf_agents.environments import tf_py_environment
from tf_agents.environments import wrappers

from src.single_agent.environments import py_function_environment as py_fun_env
from src.functions import numpy_functions as npf
from src.single_agent.utils import evaluation

ROOT_DIR = '../models'

if __name__ == '__main__':
  function = npf.Sphere()
  dims = 30
  steps = 500

  policy_dir = os.path.join(ROOT_DIR,
                            'TD3-IG/' + f'{dims}D/' + function.name)
  saved_pol = tf.compat.v2.saved_model.load(policy_dir)

  env = py_fun_env.PyFunctionEnvironment(function, dims)
  env = wrappers.TimeLimit(env, duration=steps)

  tf_eval_env = tf_py_environment.TFPyEnvironment(environment=env)

  evaluation.evaluate_agent(tf_eval_env, saved_pol, function, dims,
                            name_policy='TD3-IG',
                            name_algorithm='TD3-IG',
                            save_to_file=True, episodes=10)

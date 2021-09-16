"""Policy evaluation tests."""

import os

import tensorflow as tf
from tf_agents.environments import tf_py_environment
from tf_agents.environments import wrappers

from src.single_agent.environments import py_function_environment as py_fun_env
from src.functions import numpy_functions as npf
from src import config

from experiments.evaluation import utils as eval_utils

POLICIES_DIR = config.POLICIES_DIR

if __name__ == '__main__':
  function = npf.Sphere()
  dims = 30
  steps = 500

  policy_dir = os.path.join(POLICIES_DIR, 'Td3Agent')
  policy_dir = os.path.join(policy_dir, f'{dims}D')
  policy_dir = os.path.join(policy_dir, function.name)

  saved_pol = tf.compat.v2.saved_model.load(policy_dir)

  env = py_fun_env.PyFunctionEnv(function, dims)
  env = wrappers.TimeLimit(env, duration=steps)

  tf_eval_env = tf_py_environment.TFPyEnvironment(environment=env)

  eval_utils.evaluate_agent(tf_eval_env,
                            saved_pol,
                            function,
                            dims,
                            algorithm_name='TD3',
                            save_to_file=False,
                            episodes=10)

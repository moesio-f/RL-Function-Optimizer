import os
import tensorflow as tf
from tf_agents.environments.tf_py_environment import TFPyEnvironment
from tf_agents.environments.wrappers import TimeLimit
from environments.py_function_environment import PyFunctionEnvironment
from environments.py_function_environment_unbounded import PyFunctionEnvironmentUnbounded
from functions.numpy_functions import *
from utils.evaluation import evaluate_agent

ROOT_DIR = os.path.dirname(os.getcwd())

policy_dir = os.path.join(ROOT_DIR, "policy")
policy_collect_dir = os.path.join(ROOT_DIR, "policy_collect")

saved_pol = tf.compat.v2.saved_model.load(policy_dir)
saved_pol_col = tf.compat.v2.saved_model.load(policy_collect_dir)

function = Ackley()
dims = 20
steps = 2000

env = PyFunctionEnvironment(function, dims)
env = TimeLimit(env, duration=steps)

tf_eval_env = TFPyEnvironment(environment=env)

evaluate_agent(tf_eval_env, saved_pol, function, dims,
               name_policy='ActorPolicy',
               name_algorithm='TD3',
               save_to_file=True)
evaluate_agent(tf_eval_env, saved_pol_col, function, dims,
               name_policy='GaussianPolicy',
               name_algorithm='TD3',
               save_to_file=True)

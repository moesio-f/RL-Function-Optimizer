"""PyFunctionEnvironment validation tests."""

from tf_agents.environments import utils
from tf_agents.environments import wrappers

from src.single_agent.environments import py_function_environment as py_fun_env
from src.single_agent.environments import py_env_wrappers
from src.functions import numpy_functions as npf

if __name__ == '__main__':
  function = npf.Sphere()
  dims = 30

  env = py_fun_env.PyFunctionEnvironment(function=function,
                                         dims=dims,
                                         clip_actions=True)
  env = py_env_wrappers.RewardClip(env=env,
                                   min_reward=-400.0,
                                   max_reward=400.0)
  env = py_env_wrappers.RewardScale(env=env,
                                    scale_factor=0.2)
  env = wrappers.TimeLimit(env=env,
                           duration=500)

  utils.validate_py_environment(env,
                                episodes=50)

  unbounded_env = py_fun_env.PyFunctionEnvironment(function=function,
                                                   dims=dims)
  unbounded_env = py_env_wrappers.RewardClip(env=unbounded_env,
                                             min_reward=-400.0,
                                             max_reward=400.0)
  unbounded_env = py_env_wrappers.RewardScale(env=unbounded_env,
                                              scale_factor=0.2)
  unbounded_env = wrappers.TimeLimit(env=unbounded_env,
                                     duration=500)

  utils.validate_py_environment(unbounded_env,
                                episodes=50)

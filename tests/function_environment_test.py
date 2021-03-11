from environments.py_function_environment import PyFunctionEnvironment
from environments.py_env_wrappers import RewardClip, RewardScale
from functions.numpy_functions import Sphere
from tf_agents.environments.utils import validate_py_environment
from tf_agents.environments.wrappers import TimeLimit

env = PyFunctionEnvironment(function=Sphere(), dims=30)
env = RewardClip(env=env, min_reward=-400.0, max_reward=400.0)
env = RewardScale(env=env, scale_factor=0.2)
env = TimeLimit(env=env, duration=500)

validate_py_environment(env, episodes=50)

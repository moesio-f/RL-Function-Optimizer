"""TFFunctionEnvironment validation tests."""

from tf_agents.policies import random_tf_policy

from environments import tf_function_environment as tf_function_env
from functions import tensorflow_functions as tff

dims = 20
function = tff.Sphere()

tf_env = tf_function_env.TFFunctionEnvironment(function=function, dims=dims)
policy = random_tf_policy.RandomTFPolicy(time_step_spec=tf_env.time_step_spec(),
                                         action_spec=tf_env.action_spec())

time_step = tf_env.reset()
i = 0
while not time_step.is_last():
  i += 1
  action_step = policy.action(time_step)
  time_step = tf_env.step(action_step.action)
  print(i)

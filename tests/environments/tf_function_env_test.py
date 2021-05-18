from tf_agents.policies.random_tf_policy import RandomTFPolicy
from environments.tf_function_environment import TFFunctionEnvironment
from functions.tensorflow_functions import *

dims = 20
function = Sphere()

tf_env = TFFunctionEnvironment(function=function, dims=dims)
policy = RandomTFPolicy(time_step_spec=tf_env.time_step_spec(),
                        action_spec=tf_env.action_spec())

time_step = tf_env.reset()
i = 0
while not time_step.is_last():
    i += 1
    action_step = policy.action(time_step)
    time_step = tf_env.step(action_step.action)
    print(i)

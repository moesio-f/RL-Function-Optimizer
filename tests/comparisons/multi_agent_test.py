"""Multiple trained agents tests."""

import os

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tf_agents.environments.tf_py_environment import TFPyEnvironment
from tf_agents.environments.wrappers import TimeLimit

from environments.py_function_environment import PyFunctionEnvironment
from functions.numpy_functions import Sphere

num_agents = 300
num_steps = 2000
function = Sphere()
dims = 20

envs = []

# ---- Creating envs ----
for _ in range(num_agents):
  envs.append(TFPyEnvironment(TimeLimit(
    PyFunctionEnvironment(function=function, dims=dims, clip_actions=False),
    duration=num_steps)))

# ---- Loading policies ----
ROOT_DIR = os.path.dirname(os.getcwd())

policy_dir = os.path.join(ROOT_DIR, "policy")
policy_collect_dir = os.path.join(ROOT_DIR, "policy_collect")

saved_pol = tf.compat.v2.saved_model.load(policy_dir)
saved_pol_col = tf.compat.v2.saved_model.load(policy_collect_dir)

name_algorithm = 'TD3'
name_policy = 'ActorPolicy'

# ---- Evaluating ----
best_solution_at_it = []
best_solution = tf.float32.max
best_solution_pos = np.zeros(shape=(dims,), dtype=np.float32)
policy = saved_pol

time_steps = []

for env in envs:
  time_steps.append(env.reset())
  pos = time_steps[-1].observation.numpy()[0]
  obj_value = function(pos)
  if obj_value < best_solution:
    best_solution = obj_value
    best_solution_pos = pos

best_solution_at_it.append(best_solution)

all_done = False
step = 0
while not all_done:
  dones = []
  for i, env in enumerate(envs):
    if not time_steps[i].is_last():
      action_step = policy.action(time_steps[i])
      time_steps[i] = env.step(action_step.action)
      obj_value = -time_steps[i].reward.numpy()[0]
      if obj_value < best_solution:
        best_solution = obj_value
        best_solution_pos = time_steps[i].observation.numpy()[0]
    dones.append(time_steps[i].is_last())
  all_done = np.all(dones)
  best_solution_at_it.append(best_solution)
  step += 1
  print(step)

fig, ax = plt.subplots(figsize=(18.0, 10.0,))
ax.plot(range(len(best_solution_at_it)), best_solution_at_it,
        label='Best value found: {0}'.format(best_solution))
ax.set(xlabel="Iterations\nBest solution at: {0}".format(best_solution_pos),
       ylabel="Best objective value",
       title="{0} on {1} ({2} Dims) [{3}]".format(name_algorithm,
                                                  function.name,
                                                  dims,
                                                  name_policy))

x_ticks = np.arange(0, len(best_solution_at_it), step=50.0)
x_labels = ['{:.0f}'.format(val) for val in x_ticks]

ax.set_xticks(x_ticks)
ax.set_xticklabels(x_labels)
ax.set_xscale('symlog', base=2)
ax.set_xlim(left=0)

ax.legend()
ax.grid()

plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
plt.savefig(
  fname='{0}-{1}dims-{2}.png'.format(function.name, dims, name_policy),
  bbox_inches='tight')
plt.show()

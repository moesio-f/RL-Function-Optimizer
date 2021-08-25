import numpy as np
import gym
import json
import os

from matplotlib import pyplot as plt

from src.multi_agent.maddpg import MADDPG
from src.multi_agent.multi_agent_env import MultiAgentFunctionEnv
from src.multi_agent.replay_buffer import MultiAgentReplayBuffer
from src.functions.numpy_functions import Sphere


def get_result_dir(folder_name='results', index=None):
  """Get result directory, if doesn't exist create one
    Args:
      index: if None create the next index folder to hold results
    directory structure:
      results/1/models/
      results/2/models/
      ...
    Returns: a tuple with root and models directory
    """

  def get_latest_index_dir(dir):
    index = 1
    while os.path.exists(os.path.join(dir, str(index))):
      index += 1
    return index

  if index:
    root_dir = os.path.join(folder_name, str(index))
  else:
    index = get_latest_index_dir(folder_name)
    root_dir = os.path.join(folder_name, str(index))
  models_dir = os.path.join(root_dir, 'models')
  os.makedirs(models_dir, exist_ok=True)
  return root_dir, models_dir


# Multi Agent Run
def run(env: gym.Env, agents: MADDPG, memory: MultiAgentReplayBuffer,
        batch_size=256, episodes=60_000,
        steps=25, update_rate=100, display=False, save_interval=200,
        save_index=None):
  root_dir, models_dir = get_result_dir(index=save_index)
  with open(os.path.join(root_dir, 'specs.json'), 'w') as specs_file:
    specs = {'batch_size': batch_size, 'episodes': episodes, 'steps': steps,
             'update_rate': update_rate, 'num_agents': agents.num_agents}
    json.dump(specs, specs_file, indent=True)

  total_steps = 0
  best_score = -np.inf

  if save_index is not None:
    agents.load(models_dir)

  for ep in range(episodes):
    states = env.reset()
    current_best = -np.inf

    for step in range(steps):
      if display:
        plt.pause(0.05)
        env.render()

      actions = agents.action(states)
      next_states, rewards, dones, infos = env.step(actions)
      memory.add(states, actions, rewards, next_states, dones)
      states = next_states

      if len(
            memory) > batch_size * steps and total_steps % update_rate == 0 and not display:
        experience = memory.sample(batch_size)
        losses = agents.train(experience)

      current_best = max(current_best, np.max(rewards))
      best_score = max(best_score, current_best)

      total_steps += 1
    print(
      f'episode {ep} | best score: {best_score:.5f} | current best: {current_best:.5f}')

    if not display and ep % save_interval == 0:
      agents.save(models_dir)
  return best_score


class ScriptedAgent:
  def __init__(self) -> None:
    self.num_agents = 1

  def __len__(self):
    return self.num_agents

  def action(self, obs):
    print(obs)
    return [0.1 * o for o in obs]

  def initialize(self):
    pass

  def load(self):
    pass

  def save(self):
    pass

  def train(self):
    pass


def test_env(evaluate=False, save_index=None):
  env = MultiAgentFunctionEnv(Sphere(), 2, 1)
  # env = SimpleMultiAgentEnv(2)
  agents = MADDPG(env.observation_space, env.action_space)
  # agents = ScriptedAgent()
  memory = MultiAgentReplayBuffer(1_000_000, len(agents))
  agents.initialize()
  if evaluate:
    best = run(env, agents, memory, episodes=10, display=True,
               save_index=save_index)
  else:
    best = run(env, agents, memory, steps=25, episodes=5_000, update_rate=64,
               save_index=save_index)
  print('Best result:', best)


if __name__ == '__main__':
  test_env()

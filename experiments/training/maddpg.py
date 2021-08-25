import matplotlib
import numpy as np
import gym
import json
import os

from matplotlib import pyplot as plt

from src.multi_agent.maddpg import MADDPG
from src.multi_agent.gym_env import MultiAgentFunctionEnv
from src.multi_agent.replay_buffer import MultiAgentReplayBuffer
from src.functions.numpy_functions import Sphere


def make_results_dir(directory='results'):
  """Get result directory, if doesn't exist create one
    Args:
      index: if None create the next index folder to hold results
    directory structure:
      results/1/models/
      results/2/models/
      ...
    Returns: a tuple with root and models directory
    """

  index = 1
  while os.path.exists(os.path.join(directory, str(index))):
    index += 1
  directory = os.path.join(directory, str(index), 'models')
  os.makedirs(directory, exist_ok=True)
  return os.path.dirname(directory)

def eval(env: gym.Env, agents: MADDPG, episodes=10, steps=50, index=None, directory = 'results'):
  
  if index is None:
    index = 0
    while os.path.exists(os.path.join(directory, str(index + 1))):
      index += 1

  agents.load(os.path.join(directory, str(index), 'models'))
  best_score = -np.inf

  for ep in range(episodes):
    states = env.reset()
    best_agent = -np.inf

    for step in range(steps):
      plt.pause(0.05)
      env.render()

      actions = agents.action(states)
      next_states, rewards, _, _ = env.step(actions)
      states = next_states

      best_agent_idx = np.argmax(rewards)
      best_agent = max(best_agent, rewards[best_agent_idx])
      best_score = max(best_score, best_agent)

    print(f'episode {ep} | best score: {best_score:.5f} | current best: {best_agent:.5f} by agent {best_agent_idx}')
  return best_score

# Multi Agent Run
def train(env: gym.Env, agents: MADDPG, memory: MultiAgentReplayBuffer,
        batch_size=256, episodes=60_000,
        steps=25, update_rate=100, save_interval=200):

  save_directory = make_results_dir()
  models_dir = os.path.join(save_directory, 'models')

  with open(os.path.join(save_directory, 'specs.json'), 'w') as specs_file:
    json.dump({
      'batch_size': batch_size, 
      'episodes': episodes, 
      'steps': steps,
      'update_rate': update_rate,
      'num_agents': agents.num_agents
    }, specs_file, indent=True)

  total_steps = 0
  best_score = -np.inf

  for ep in range(episodes):
    states = env.reset()
    best_agent = -np.inf

    for step in range(steps):
      actions = agents.action(states)
      next_states, rewards, dones, infos = env.step(actions)
      memory.add(states, actions, rewards, next_states, dones)
      states = next_states

      if len(memory) > batch_size * steps and total_steps % update_rate == 0:
        experience = memory.sample(batch_size)
        losses = agents.train(experience)

      best_agent_idx = np.argmax(rewards)
      best_agent = max(best_agent, rewards[best_agent_idx])
      best_score = max(best_score, best_agent)

      total_steps += 1
    print(f'episode {ep} | best score: {best_score:.5f} | current best: {best_agent:.5f} by agent {best_agent_idx}')

    if ep % save_interval == 0:
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
  env = MultiAgentFunctionEnv(Sphere(), dims=2, n_agents=1)
  
  # env = SimpleMultiAgentEnv(2)
  agents = MADDPG(env.observation_space, env.action_space)
  agents.initialize()
  # agents = ScriptedAgent()
  if evaluate:
    best = eval(env, agents, index=save_index)
  else:
    memory = MultiAgentReplayBuffer(1_000_000, len(agents))
    best = train(env, agents, memory, steps=25, episodes=5_000, update_rate=64)
  print('Best result:', best)

if __name__ == '__main__':
  test_env(evaluate=False)


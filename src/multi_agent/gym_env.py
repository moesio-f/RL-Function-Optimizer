from typing import List

from src.functions.core import Domain, Function
from src.single_agent.utils.render import FunctionDrawer
import matplotlib.pyplot as plt
import numpy as np
import gym
from gym.spaces import Box


class MultiAgentFunctionEnv(gym.Env):
  def __init__(self, function: Function, dims: int, n_agents: int, clip=False):
    state: list[np.ndarray]
    reseted: bool
    self.func = function
    self.drawer = FunctionDrawer(function)
    self.dims = dims
    self.n_agents = n_agents
    
    self.observation_space = [Box(*function.domain, (dims,)) for _ in range(n_agents)]
    
    if clip:
      self.action_space = self.observation_space
    else:
      self.action_space = [Box(-np.inf, np.inf, (dims,)) for _ in range(n_agents)]

  def step(self, actions: List[np.ndarray]):
    min, max = self.func.domain
    for i, action in enumerate(actions):
      self.state[i] = np.clip(self.state[i] + action, min, max)

    rewards = [-self.func(s) for s in self.state]
    dones = [False] * self.n_agents
    return self.state, rewards, dones, None

  def reset(self):
    self.reseted = True
    self.state = [space.sample().astype(np.float32)
                  for space in self.observation_space]
    return self.state

  def render(self, mode='human'):
    if self.reseted:
      self.reseted = False
      self.drawer.clear()
      self.drawer.draw_mesh(alpha=0.4, cmap='coolwarm')

      self.drawer.scatter(self.state[0][:2])
    self.drawer.update_scatter(self.state[0][:2])
  def __repr__(self) -> str:
    return f'{type(self).__name__}(function={self.func})'


class SimpleMultiAgentEnv(gym.Env):
  def __init__(self, objective: np.ndarray, dims: int, n_agents: int = 1,
               domain = Domain(-1.0, 1.0), clip=False):
    self.objective = objective.astype(np.float32)
    self.dims = dims
    self.n_agents = n_agents
    self.domain = domain

    self.observation_space = [Box(*self.domain, (dims,)) 
                              for _ in range(n_agents)]
    self.action_space = [Box(*self.domain, (dims,)) for _ in range(n_agents)]
    self.fig = None

  def init_viewer(self):
    self.fig, self.ax = plt.subplots()
    self.agent_axes = [self.ax.scatter(0, 0, color='b') for _ in range(self.n_agents)]
    self.objective_ax = self.ax.scatter(*self.objective, color='r')
    self.ax.set_xlim(self.domain)
    self.ax.set_ylim(self.domain)

  def step(self, actions: List[np.ndarray]):
    min, max = self.domain
    for i, action in enumerate(actions):
      self.states[i] = np.clip(self.states[i] + action, min, max)

    # get distance between each agent and the objective
    rewards = [-np.linalg.norm(s - self.objective) for s in self.states]
    dones = [False] * self.n_agents
    return self.states, rewards, dones, None

  def reset(self):
    min, max = self.domain
    shape = (self.dims,)
    self.states = [np.random.uniform(min, max, shape).astype(np.float32)
                   for _ in range(self.n_agents)]
    return np.concatenate(self.states)[None]

  def render(self, mode='human'):
    if self.fig is None:
      self.init_viewer()

    for agent_pos, ax in zip(self.states, self.agent_axes):
      ax.set_offsets(agent_pos[:2])
  
  def __repr__(self) -> str:
    return f'{type(self).__name__}(objective={self.objective})'

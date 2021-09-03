from typing import List

from src.functions.core import Domain, Function
from src.single_agent.utils.render import FunctionDrawer
import matplotlib.pyplot as plt
import numpy as np
import gym
from gym.spaces import Box


class MultiAgentFunctionEnv(gym.Env):
  def __init__(self, function: Function, dims: int, n_agents: int, clip_actions=False):
    states: list[np.ndarray] = None
    reseted: bool = False
    self.func = function
    self.drawer = FunctionDrawer(function)
    self.dims = dims
    self.n_agents = n_agents
    self.should_clip = clip_actions
    
    self.action_space = self.observation_space =\
      [Box(*function.domain, (dims,)) for _ in range(n_agents)]

  def step(self, actions: List[np.ndarray]):
    self.states = [s + a for s,a in zip(self.states, actions)]

    if self.should_clip:
      min, max = self.func.domain
      self.states = [np.clip(x, min, max) for x in self.states]

    rewards = [-self.func(s) for s in self.states]

    dones = [not obs_space.contains(state)
      for obs_space, state in zip(self.observation_space, self.states)]
    
    return self.states, rewards, dones, None

  def reset(self):
    self.reseted = True
    self.states = [space.sample().astype(np.float32)
                  for space in self.observation_space]
    return self.states

  def render(self, mode='human'):
    if self.reseted:
      self.reseted = False
      self.drawer.clear()
      self.drawer.draw_mesh(alpha=0.4, cmap='coolwarm')
      for state in self.states:
        self.drawer.scatter(state[:2])
    
    for i, state in enumerate(self.states):
        self.drawer.update_scatter(state[:2], i)
    
  def __repr__(self) -> str:
    return f'{type(self).__name__}(function={self.func})'


class SimpleMultiAgentEnv(gym.Env):
  def __init__(self, objective: np.ndarray, dims: int, n_agents: int = 1,
               domain = Domain(-1.0, 1.0)):
    self.fig = None
    self.objective = objective.astype(np.float32)
    self.dims = dims
    self.n_agents = n_agents
    self.domain = domain
    self.action_space = self.observation_space =\
      [Box(*self.domain, (dims,)) for _ in range(n_agents)]
      
  def init_viewer(self):
    self.fig, self.ax = plt.subplots()
    self.agent_axes = [self.ax.scatter(0, 0, color='b') for _ in range(self.n_agents)]
    self.objective_ax = self.ax.scatter(*self.objective, color='r')
    self.ax.set_xlim(self.domain)
    self.ax.set_ylim(self.domain)

  def step(self, actions: List[np.ndarray]):
    self.states = [s + a for s,a in zip(self.states, actions)]
    rewards = [-np.linalg.norm(s - self.objective) for s in self.states]
    dones = [not self.observation_space.contains(s) for s in self.state]
    return self.states, rewards, dones, None

  def reset(self):
    self.state = [space.sample().astype(np.float32)
                  for space in self.observation_space]
    return np.concatenate(self.states)[None]

  def render(self, mode='human'):
    if self.fig is None:
      self.init_viewer()

    for agent_pos, ax in zip(self.states, self.agent_axes):
      ax.set_offsets(agent_pos[:2])
  
  def __repr__(self) -> str:
    return f'{type(self).__name__}(objective={self.objective})'

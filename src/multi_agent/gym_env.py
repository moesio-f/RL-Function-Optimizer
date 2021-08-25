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
    self.n = n_agents
    self.observation_space = [
      Box(function.domain.min, function.domain.max, (dims,)) for _ in
      range(n_agents)]
    if clip:
      self.action_space = self.observation_space.copy()
    else:
      self.action_space = [Box(-np.inf, np.inf, (dims,)) for _ in
                           range(n_agents)]

  def step(self, actions: List[np.ndarray]):
    min, max = self.func.domain
    for i in range(len(actions)):
      self.state[i] += actions[i]

      if not self.observation_space[i].contains(self.state):
        self.state[i] = np.clip(self.state[i], min, max)

    rewards = -self.func(np.array(self.state).T)
    dones = [
              False] * self.n  # currently this environment does not know the final pos

    return self.state, rewards, dones, None

  def reset(self):
    min, max = self.func.domain
    self.state = [np.random.uniform(min, max, space.shape).astype(np.float32)
                  for space in self.observation_space]
    self.reseted = True
    return self.state

  def render(self, mode='human'):
    if self.reseted:
      self.reseted = False
      self.drawer.clear()
      self.drawer.draw_mesh(alpha=0.4, cmap='coolwarm')

      self.drawer.scatter(self.state[0][:2])
    self.drawer.update_scatter(self.state[0][:2])
    # if self.reseted:
    #     self.drawer.draw_mesh()
    # self.drawer.scatter(self.state.T, pause_time=0.1)


class SimpleMultiAgentEnv(gym.Env):
  def __init__(self, dims: int, n_agents: int = 1, n_landmarks: int = 1,
               clip=False):
    self.dims = dims
    self.n_agents = n_agents
    self.n_landmarks = n_landmarks
    self.domain = Domain(-1.0, 1.0)

    self.observation_space = [Box(*self.domain, (dims * n_landmarks,)) for _ in
                              range(n_agents)]
    self.action_space = [Box(*self.domain, (dims,)) for _ in range(n_agents)]
    self.fig = None

  def init_viewer(self):
    self.fig, self.ax = plt.subplots()
    self.agent_p = self.ax.scatter(0, 0, color='b')
    self.land_p = self.ax.scatter(0, 0, color='r')
    self.ax.set_xlim(self.domain)
    self.ax.set_ylim(self.domain)

  def step(self, actions: List[np.ndarray]):
    min, max = self.domain
    for i in range(len(actions)):
      self.states[i] += actions[i]

      if not self.observation_space[i].contains(self.states):
        self.states[i] = np.clip(self.states[i], min, max)

    rewards = [-np.sum((s - l) ** 2) for s, l in
               zip(self.states, self.landmarks)]
    dones = [
              False] * self.n_agents  # currently this environment does not know the final pos
    observation = self._observation()

    return observation, rewards, dones, None

  def reset(self):
    min, max = self.domain
    shape = (1, self.dims)

    self.states = [np.random.uniform(min, max, shape).astype(np.float32)
                   for _ in range(self.n_agents)]

    self.landmarks = [np.random.uniform(min, max, shape).astype(np.float32)
                      for _ in range(self.n_landmarks)]

    self.render()

    return self._observation()

  def _observation(self):
    observations = []
    for state in self.states:
      entity_pos = [entity - state for entity in self.landmarks]
      observations.append(np.concatenate(entity_pos))
    return observations

  def render(self, mode='human'):
    if self.fig is None:
      self.init_viewer()
    agent_p = self.states[0][0]
    land_p = self.landmarks[0][0]

    self.agent_p.set_offsets(agent_p)
    self.land_p.set_offsets(land_p)

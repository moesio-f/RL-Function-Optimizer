from functions.function import Function
from utils.render.function_drawer import FunctionDrawer
import numpy as np
import gym
from gym.spaces import Box

class MultiAgentFunctionEnv(gym.Env):
    state: np.ndarray
    reseted: bool

    def __init__(self, function: Function, dims, n_agents, clip=False):
        self.func = function
        self.drawer = FunctionDrawer(function)
        self.dims = dims
        self.n = n_agents
        self.observation_space = Box(function.domain.min, function.domain.max, (n_agents, dims))
        self.action_space = self.observation_space if clip else Box(-np.inf, np.inf, (n_agents, dims))

    def step(self, action: np.ndarray):
        self.state += action
        if not self.observation_space.contains(self.state):
            self.state = np.clip(self.state, self.func.domain.min, self.func.domain.max)
        
        rewards = -self.func(self.state.T)
        dones = [False] * self.n  # currently this environment does not know the final pos

        return self.state, rewards, dones, None
        
    def reset(self):
        self.state = np.random.uniform(self.func.domain.min, self.func.domain.max, 
                                       self.observation_space.shape)
        self.state = self.state.astype(np.float32)
        self.reseted = True
        return self.state
    
    def render(self, mode='human'):
        if self.reseted:
            self.drawer.draw_mesh()
            self.reseted = False
        self.drawer.draw_point(self.state.T, pause_time=0.1)

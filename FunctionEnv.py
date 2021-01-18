import numpy as np
import gym
from gym import spaces


class FunctionEnv(gym.Env):
    def __init__(self, dims, function, minimize):
        self.function = function
        self.dims = dims
        self.pos = np.random.uniform()
        self.action_space = spaces.Box()
        self.observation_space = spaces.Box()

    def step(self, action):
        # TODO
        return

    def reset(self):
        # TODO
        return
    
    def render(self, mode):
        # TODO
        return

from typing import List
import numpy as np

class MultiAgentReplayBuffer(object):
    def __init__(self, size: int, n_agents: int):
        """Create Prioritized Replay buffer.

        each data added in this replay buffer is a tuple: 
        (states, actions, rewards, next_states, dones)
        """
        self.n = n_agents
        self.agents = np.zeros((n_agents, size), dtype=tuple)
        self.used = 0
        self.maxsize = int(size)
        self.next_idx = 0

    def __len__(self):
        return self.used

    def clear(self):
        self.agents = np.zeros((self.n, self.maxsize), dtype=tuple)
        self.used = 0
        self.next_idx = 0

    def add(self, 
            obs: List[np.ndarray],
            act: List[np.ndarray],
            rew: List[np.float32],
            next_obs: List[np.ndarray], 
            dones: List[np.bool]):

        assert self.n == len(obs) == len(act) == len(rew) == len(next_obs) == len(dones)
        data = list(zip(obs, act, rew, next_obs, dones))

        for i in range(self.n):
            self.agents[i, self.next_idx] = data[i]
        
        self.next_idx = (self.next_idx + 1) % self.maxsize

        if self.used < self.maxsize:
            self.used += 1

    def sample_index(self, indexes: List[int]):
        # per agent info
        states, actions, rewards, next_states, dones = [], [], [], [], []

        for a in range(self.n):
            tuples = self.agents[a, indexes]
            
            trajectory = ([], [], [], [], [])
            for t in tuples:
                for i, info in enumerate(t):
                    trajectory[i].append(info)

            obs, act, rew, new_obs, done = trajectory
            states.append(np.array(obs, copy=False))
            actions.append(np.array(act, copy=False))
            rewards.append(np.array(rew, copy=False))
            next_states.append(np.array(new_obs, copy=False))
            dones.append(np.array(done, copy=False))

        return states, actions, rewards, next_states, dones

    def make_index(self, batch_size):
        return [np.random.randint(0, self.used) for _ in range(batch_size)]

    def sample(self, batch_size):
        if batch_size > 0:
            idxes = self.make_index(batch_size)
        else:
            idxes = range(0, self.used)
        states, actions, next_states, rewards, dones = self.sample_index(idxes)
        
        return states, actions, next_states, rewards, dones

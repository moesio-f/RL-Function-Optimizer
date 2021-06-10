from typing import List, Tuple

import tensorflow as tf

import numpy as np
from tf_agents.agents.ddpg.ddpg_agent import DdpgAgent

# from tf_agents.agents.tf_agent import LossInfo, TFAgent
# from tf_agents.policies.ou_noise_policy import OUNoisePolicy
# from tf_agents.policies.tf_policy import TFPolicy
# from tf_agents.trajectories import time_step as ts
# from tf_agents.trajectories import policy_step
# from tf_agents.typing import types
# from tf_agents.replay_buffers.tf_uniform_replay_buffer import TFUniformReplayBuffer
# from tf_agents.trajectories.trajectory import Transition

from tf_agents.policies.tf_policy import TFPolicy
from tf_agents.policies.policy_saver import PolicySaver
from tf_agents.networks import Sequential
from tf_agents.utils import common

from tf_agents.policies.actor_policy import ActorPolicy


class MultiAgentReplayBuffer(object):
    def __init__(self, size, n_agents):
        """Create Prioritized Replay buffer.

        each data added in this replay buffer is a tuple: 
        (states, actions, rewards, next_states, dones)

        where len(states) == len(actions) == ... == len(dones) == n_agents
        so, data has shape (5, n_agents)

        and self._storage has shape (size, 5, n_agents)

        Parameters
        ----------
        size: int
            Max number of transitions to store in the buffer. When the buffer
            overflows the old memories are dropped.
        """
        self._n_agents = n_agents
        self._storage = []
        self._maxsize = int(size)
        self._next_idx = 0

    def __len__(self):
        return len(self._storage)

    def clear(self):
        self._storage = []
        self._next_idx = 0

    def add(self, observations, actions, rewards, next_observations, dones):
        data = (observations, actions, rewards, next_observations, dones)

        if self._next_idx >= len(self._storage):
            self._storage.append(data)
        else:
            self._storage[self._next_idx] = data
        self._next_idx = (self._next_idx + 1) % self._maxsize

    # each return has shape (batch, n_agents, dimentions)
    def sample_index(self, indexes: List[int]):
        states, actions, rewards, next_states, dones = [], [], [], [], []
        for i in indexes:
            data = self._storage[i] 
            observations, action, reward, obs_tp1, done = data
            states.append(np.array(observations, copy=False))
            actions.append(np.array(action, copy=False))
            rewards.append(reward)
            next_states.append(np.array(obs_tp1, copy=False))
            dones.append(done)
        return np.array(states), np.array(actions), np.array(rewards), np.array(next_states), np.array(dones)

    def make_index(self, batch_size):
        return [np.random.randint(0, len(self._storage) - 1) for _ in range(batch_size)]

    # get latest batch_size indexes added and mix them
    def make_latest_index(self, batch_size):
        idx = [(self._next_idx - 1 - i) % self._maxsize for i in range(batch_size)]
        np.random.shuffle(idx)
        return idx

    def sample(self, batch_size):
        """Sample a batch of experiences.

        Parameters
        ----------
        batch_size: int
            How many transitions to sample.

        Returns
        -------
        obs_batch: np.array
            batch of agents observations
        act_batch: np.array
            batch of agents actions executed given obs_batch
        rew_batch: np.array
            agents rewards received as results of executing act_batch
        next_obs_batch: np.array
            next set of agents observations seen after executing act_batch
        done_mask: np.array
            done_mask[i] = 1 if executing act_batch[i] resulted in
            the end of an episode and 0 otherwise.
        """
        if batch_size > 0:
            idxes = self.make_index(batch_size)
        else:
            idxes = range(0, len(self._storage))
        return self.sample_index(idxes)

    def collect(self):
        return self.sample(-1)

class Agent:
    def __init__(self, actor_dims: int, critic_dims: int, n_agents: int,
                 n_actions: int, agent_idx: int, gamma=0.95, tau=0.01, 
                 actor_fc_params=(256,256), critic_fc_params=(256,256),
                 chckpt_dir=None):
        """)
        actor_fc_params stands for "fully_connected"
        """
        
        self.gamma = gamma
        self.tau = tau
        self.n_actions = n_actions
        self.n_agents = n_agents
        self.name = f'agent_{agent_idx}'
        self.index = agent_idx
        self.chckpt_dir = chckpt_dir

        # concatenate layer to pass input to critic network
        self.concatenate = tf.keras.layers.Concatenate()
        
        # config actor network
        actor_layers = [tf.keras.layers.Dense(units, 'relu') for units in actor_fc_params]
        actor_layers.append(tf.keras.layers.Dense(n_actions, 'linear'))

        input_spec = tf.TensorSpec((actor_dims,))
        self.actor = Sequential(actor_layers, input_spec, self.name+'_actor')
        
        # config critic network
        critic_layers = [tf.keras.layers.Dense(units, 'relu') for units in critic_fc_params]
        critic_layers.append(tf.keras.layers.Dense(1, 'linear'))

        input_spec = tf.TensorSpec((critic_dims + n_agents*n_actions,))
        self.critic = Sequential(critic_layers, input_spec, self.name+'_critic')

        self.target_actor = self.actor.copy(name=self.name+'_target_actor')
        self.target_critic = self.critic.copy(name=self.name+'_target_critic')

    def initiate(self):
        self.actor.create_variables()
        self.critic.create_variables()
        self.target_actor.create_variables()
        self.target_critic.create_variables()
        
    def update_targets(self):
        common.soft_variables_update(self.actor.variables,
            self.target_actor.variables, self.tau)
        common.soft_variables_update(self.critic.variables,
            self.target_critic.variables, self.tau)

    def action(self, obs: tf.Tensor) -> tf.Tensor:
        action = self.actor(obs)[0]
        return action

    # input shape (batch, n_agents, dims) -> (batch, n_agents*dims)
    def flatten(self, states, actions):
        states = tf.reshape(states, (states.shape[0], -1))
        actions = tf.reshape(actions, (actions.shape[0], -1))
        return states, actions
    
    # input shape = (batch, n_agents, dims)
    def evaluate(self, states: tf.Tensor, actions: tf.Tensor):
        input = self.flatten(states, actions)
        input = self.concatenate(input)
        return self.critic(input)[0]
    
    # input shape = (batch, n_agents, dims)
    def evaluate_target(self, states: tf.Tensor, actions: tf.Tensor):
        input = self.flatten(states, actions)
        input = self.concatenate(input)
        return self.target_critic(input)[0]

    # train critic based on all states and actions of environment
    def train_critic(self, states: tf.Tensor, actions: tf.Tensor, target_q: tf.Tensor,
                     optimizer: tf.keras.optimizers.Optimizer):
        with tf.GradientTape() as tape:
            critic_value = self.evaluate(states, actions)
            critic_loss = tf.keras.losses.mse(target_q, critic_value)
        optimizer.minimize(critic_loss, self.critic.trainable_variables, tape=tape)


    def train_actor(self, states: tf.Tensor, actions: tf.Tensor, optimizer: tf.keras.optimizers.Optimizer):
        observations = states[:, self.index] # batch of observations of the current agent

        batches, n_agents, _ = actions.shape
        # changing shape to (agents, batches, dims)
        agents_actions = [actions[:, i] for i in range(n_agents)] 

        with tf.GradientTape() as tape:
            new_actions = self.action(observations)

            # update the new batch of action in the joint actions
            agents_actions[self.index] = new_actions
            agents_actions = tf.stack(agents_actions)

            # returning to original shape
            updated_actions = tf.stack([agents_actions[:, i] for i in range(batches)])

            loss = self.evaluate(states, updated_actions)
            # regularization = tf.reduce_mean(new_actions)
            loss = -tf.reduce_mean(loss) # + 1e-3 * regularization
        optimizer.minimize(loss, self.actor.trainable_variables, tape=tape)
    
    def save_checkpoint(self):
        pass

class MADDPG:
    def __init__(self, actor_dims: List[int], critic_dims: int, 
            n_agents, n_actions, alpha=0.01, beta=0.01, chkpt_dir='results/maddpg/'):
        """
        Arguments:
          actor_dims: a list of integers with the size of each actor observation
          critic_dims: a integers with the size of each actor observation
        """
        self.agents: List[Agent] = []
        self.n_agents = n_agents
        self.n_actions = n_actions
        self.actor_optimizer = tf.keras.optimizers.Adam(alpha, clipnorm=0.5)
        self.critic_optimizer = tf.keras.optimizers.Adam(beta, clipnorm=0.5)

        # all critic nets receive all observations of actors
        assert sum(actor_dims) == critic_dims

        for index, dims in enumerate(actor_dims):
            self.agents.append(Agent(dims, critic_dims, n_agents, n_actions, index))

    def initialize(self):
        for agent in self.agents:
            agent.initiate()

    # raw obs: tensor with shape = (n_agents, dimentions)
    def action(self, state) -> np.ndarray:
        state = tf.convert_to_tensor(state)
        actions = []
        for agent, obs in zip(self.agents, state):
            action = agent.action(obs[None])
            actions.append(action)
        return tf.concat(actions, 0).numpy()

    def train(self, experience: Tuple[np.ndarray, ...]):
        """
        where transition is a 5-tuple with:
            (observations, actions, rewards, next_observations, dones)
        and each observation, actions, etc.. has shape (batch, n_agents, dims)
        """
        # TODO: 
        # for states, actions, rewards, new_states, dones in zip(experience):

        states, actions, rewards, new_states, dones = experience
        

        states = tf.convert_to_tensor(states, dtype=tf.float32)
        actions = tf.convert_to_tensor(actions, dtype=tf.float32)
        rewards = tf.convert_to_tensor(rewards, dtype=tf.float32)
        new_states = tf.convert_to_tensor(new_states, dtype=tf.float32)
        dones = tf.convert_to_tensor(dones)

        batch_size = states.shape[0]

        # pass to each agent's target actor is own batch of observations
        # obs: shape change (batches, n_agents, dims) -> (n_agents, batches, dims)
        new_actions = [agent.target_actor(new_states[:, i])[0] 
            for i, agent in enumerate(self.agents)]
        new_actions = tf.stack(new_actions)
        
        # shape change: (n_agents, batches, dims) -> (batches, n_agents, dims)
        new_actions = [new_actions[:, i] for i in range(batch_size)]
        new_actions = tf.stack(new_actions)

        for i, agent in enumerate(self.agents):
            # Take critic evaluation from new state and taking new actions
            new_critic_value = agent.evaluate_target(new_states, new_actions)
            target = rewards[:, i, None] + agent.gamma*new_critic_value

            agent.train_critic(states, actions, target, self.critic_optimizer)
            agent.train_actor(states, actions, self.actor_optimizer)

            agent.update_targets()
    def save(self):
        for agent in self.agents:
            agent.save_checkpoint()

from functions.numpy_functions import *
from environments.gym_env import MultiAgentFunctionEnv

if __name__ == '__main__':
    BATCH_SIZE = 32
    N_EPISODES = 2048
    N_STEPS = 25
    N_AGENTS = 1
    DIMS = 2
    EVALUATE = False

    FUNCTION = Sphere()
    N_ACTIONS = DIMS

    # Observation dims for actor and critic
    ACTOR_DIMS = [DIMS] * N_AGENTS  # each actor will see a observation which have length of 'dims'
    CRITIC_DIMS = sum(ACTOR_DIMS)   # every critic will see observation for every agent and

    env = MultiAgentFunctionEnv(FUNCTION, DIMS, N_AGENTS, True)

    maddpg_agents = MADDPG(ACTOR_DIMS, CRITIC_DIMS, N_AGENTS, N_ACTIONS)
    maddpg_agents.initialize()

    memory = MultiAgentReplayBuffer(100_000, N_AGENTS)
    
    # collecting data
    # for e in range(32):
    #     state = env.reset()
    #     for s in range(32):
    #         action = np.random.uniform(0., 2., state.shape).astype(np.float32)
    #         next_state, reward, done, _ = env.step(action)
    #         memory.add(state, action, reward, next_state, done)
    #         state = next_state

    PRINT_INTERVAL = 10
    total_steps = 0
    score_history = []
    best_score = 0

    if EVALUATE:
        # maddpg_agents.load
        pass

    checkpointer = common.Checkpointer("tmp/maddpg", agent=maddpg_agents.agents[0].actor)
    tf.compat.v1.train.get_or_create_global_step()

    for ep in range(N_EPISODES):
        obs = env.reset()
        score = [0.0] * N_AGENTS
        done = [False] * N_AGENTS

        for step in range(N_STEPS):
            actions = maddpg_agents.action(obs)
            next_obs, reward, done, info = env.step(actions)

            if step == N_STEPS -1:
                done = [True] * N_AGENTS
            
            memory.add(obs, actions, reward, next_obs, done)

            if total_steps % 100 == 64 and not EVALUATE:
                maddpg_agents.train(memory.sample(BATCH_SIZE))

            obs = next_obs

            for i in range(len(score)):
                score[i] += float(reward[i])
            
            total_steps += 1
        score_history.append(score)
        avg_score = np.mean(score_history[-100:])

        if not EVALUATE:
            if avg_score > best_score:
                maddpg_agents.save_checkpoint()
                best_score = avg_score
        if ep % PRINT_INTERVAL == 0 and ep > 0:
            print('episode', ep, 'average score {:.1f}'.format(avg_score))
            checkpointer.save()
    
    for e in range(4):
        state = env.reset()
        for s in range(32):
            env.render()
            action = maddpg_agents.action(state)
            next_state, reward, done, _ = env.step(action)
            state = next_state

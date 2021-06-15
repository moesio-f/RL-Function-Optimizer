from typing import List, Tuple

import numpy as np
import tensorflow as tf

from tf_agents.networks import Sequential
from tf_agents.utils import common

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

    # each argument must have shape (n_agents, dimentions)
    def add(self, obs: list, act: list, rew: list, next_obs: list, dones: List[np.bool]):
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

    # get latest batch_size indexes added and mix them
    def make_latest_index(self, batch_size):
        idx = [(self.next_idx - 1 - i) % self.maxsize for i in range(batch_size)]
        np.random.shuffle(idx)
        return idx

    def sample(self, batch_size):
        if batch_size > 0:
            idxes = self.make_index(batch_size)
        else:
            idxes = range(0, self.used)
        states, actions, next_states, rewards, dones = self.sample_index(idxes)
        
        return states, actions, next_states, rewards, dones

    def collect(self):
        return self.sample(-1)

class Agent:
    def __init__(self, actor_dims: int, critic_dims: int, n_agents: int,
                 n_actions: int, agent_idx: int, gamma=0.95, tau=0.01, 
                 actor_fc_params=(256,256), critic_fc_params=(256,256),
                 chckpt_dir=None):
        
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
        return self.actor(obs)[0]

    # input shape for each arg: (n_agents, batch_size, dims) 
    # output shape: (batch, n_agents*states + n_agents*actions)
    def preprocess(self, states: list, actions: list):
        return tf.concat(states + actions, -1)
    
    def evaluate(self, states, actions):
        input = self.preprocess(states, actions)
        return self.critic(input)[0]
    
    def evaluate_target(self, states, actions):
        input = self.preprocess(states, actions)
        return self.target_critic(input)[0]

    # train critic based on all states and actions of environment
    def train_critic(self, states: tf.Tensor, actions: tf.Tensor, target_q: tf.Tensor,
                     optimizer: tf.keras.optimizers.Optimizer):
        with tf.GradientTape() as tape:
            critic_value = self.evaluate(states, actions)
            critic_loss = tf.square(target_q - critic_value)
            loss = tf.reduce_mean(critic_loss)
        optimizer.minimize(loss, self.critic.trainable_variables, tape=tape)
        return critic_loss


    # input shape for each arg: (n_agents, batch_size, dims) 
    def train_actor(self, states: tf.Tensor, actions: tf.Tensor, optimizer: tf.keras.optimizers.Optimizer):
        observations = states[self.index]
        actions = tf.unstack(actions)
        with tf.GradientTape() as tape:
            new_actions = self.action(observations)

            # update the new batch of action in the joint actions
            actions[self.index] = new_actions

            q_value = self.evaluate(states, actions)
            regularization = tf.reduce_mean(tf.square(new_actions))
            loss = -tf.reduce_mean(q_value)  + 1e-3 * regularization
        optimizer.minimize(loss, self.actor.trainable_variables, tape=tape)
        return loss

    def save_checkpoint(self):
        pass

class MADDPG:
    def __init__(self, actor_dims: List[int], 
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
        critic_dims = sum(actor_dims)

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
        and each observation, actions, etc.. has shape (batch_size, n_agents, dims)
        """
        # TODO: 
        # for states, actions, rewards, new_states, dones in zip(experience):

        states, actions, rewards, new_states, dones = experience
        
        dones = [tf.convert_to_tensor(done) for done in dones]
        states = [tf.convert_to_tensor(state) for state in states]
        actions = [tf.convert_to_tensor(action) for action in actions]
        rewards = [tf.convert_to_tensor(reward) for reward in rewards]
        new_states = [tf.convert_to_tensor(new_state) for new_state in new_states]

        # pass to each agent's target actor is own batch of observations
        new_actions = [agent.target_actor(new_states[i])[0] for i, agent in enumerate(self.agents)]
        
        for i, agent in enumerate(self.agents):
            # Take critic evaluation from new state and taking new actions
            new_critic_value = agent.evaluate_target(new_states, new_actions)
            target = rewards[i] + agent.gamma*new_critic_value

            agent.train_critic(states, actions, target, self.critic_optimizer)
            agent.train_actor(states, actions, self.actor_optimizer)

            agent.update_targets()

    def save(self):
        for agent in self.agents:
            agent.save_checkpoint()


from environments.gym_env import MultiAgentFunctionEnv
from functions.numpy_functions import *

if __name__ == '__main__':
    BATCH_SIZE = 32
    N_EPISODES = 10_000
    N_STEPS = 25
    N_AGENTS = 4
    DIMS = 3
    display = False

    FUNCTION = Sphere()
    N_ACTIONS = DIMS

    # Observation dims for actor and critic
    ACTOR_DIMS = [DIMS] * N_AGENTS  # each actor will see a observation which have length of 'dims'
    CRITIC_DIMS = sum(ACTOR_DIMS)   # every critic will see observation for every agent and

    env = MultiAgentFunctionEnv(FUNCTION, DIMS, N_AGENTS, True)

    maddpg_agents = MADDPG(ACTOR_DIMS, N_AGENTS, N_ACTIONS)
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
    SAVE_INTERVAL = 128
    UPDATE_RATE = 10
    total_steps = 0
    best_score = float('-inf')

    # global_step = tf.compat.v1.train.get_or_create_global_step()
    # checkpointer = common.Checkpointer("tmp/maddpg", 1, agent0=maddpg_agents.agents[0].actor, global_step=global_step)

    for ep in range(N_EPISODES):
        obs = env.reset()
        done = [False] * N_AGENTS

        for step in range(N_STEPS):
            actions = maddpg_agents.action(obs)
            next_obs, reward, done, info = env.step(actions)

            if step == N_STEPS -1:
                done = [True] * N_AGENTS
            
            memory.add(obs, actions, reward, next_obs, done)

            if len(memory) > BATCH_SIZE * N_STEPS and not display and total_steps % UPDATE_RATE == 0:
                maddpg_agents.train(memory.sample(BATCH_SIZE))

            obs = next_obs

            total_steps += 1
            # global_step.assign_add(1)

        best_agent_idx = np.argmax(reward)
        best_agent = reward[best_agent_idx]
        best_score = max(best_score, best_agent)

        if ep % PRINT_INTERVAL == 0 and ep > 0:
            print(f'episode {ep} best score: {best_score} | current best {best_agent} by agent {best_agent_idx}')
        
        # if ep % SAVE_INTERVAL == 0:
            # checkpointer.save(global_step)
    
    for e in range(4):
        state = env.reset()
        for s in range(32):
            env.render()
            action = maddpg_agents.action(state)
            next_state, reward, done, _ = env.step(action)
            state = next_state

import os
from typing import List, Tuple, Union

import tensorflow as tf
import numpy as np
from gym.spaces import Box, discrete
from gym.spaces.space import Space
from tf_agents.networks import Sequential
from tf_agents.networks.value_network import ValueNetwork
from tf_agents.utils import common

class Agent:
    """Single agent of MADDPG algorithm"""
    def __init__(self, 
                 obs_spaces: List[Box],
                 act_spaces: List[Box], 
                 agent_idx: int, 
                 gamma: float = 0.95, 
                 tau: float = 1e-2, 
                 actor_fc_params = (256,256),
                 critic_fc_params = (256,256),
                 action_activation_fn = 'linear'):
        """Creates a MADDPG Agent"""

        self.use_gumbel = type(act_spaces[0]) is discrete
        self.gamma = gamma
        self.tau = tau
        self.name = f'agent_{agent_idx}'
        self.index = agent_idx
        
        obs_specs = [tf.TensorSpec(space.shape, name=f'obs_{i}') for i, space in enumerate(obs_spaces)]
        act_specs = [tf.TensorSpec(space.shape, name=f'act_{i}') for i, space in enumerate(act_spaces)]
        
        # config actor network
        n_actions = act_specs[agent_idx].shape.num_elements()

        actor_layers = [tf.keras.layers.Dense(units, 'relu') for units in actor_fc_params]
        actor_layers.append(tf.keras.layers.Dense(n_actions, action_activation_fn, name='action'))

        self.actor = Sequential(actor_layers, obs_specs[agent_idx], self.name+'_actor')
        
        # config critic network
        self.critic = ValueNetwork(
            obs_specs + act_specs, 
            preprocessing_combiner=tf.keras.layers.Concatenate(),
            fc_layer_params=critic_fc_params,
            name=self.name+'_critic')

        self.target_actor = self.actor.copy(name=self.name+'_target_actor')
        self.target_critic = self.critic.copy(name=self.name+'_target_critic')

    def initialize(self):
        self.actor.create_variables()
        self.critic.create_variables()
        self.target_actor.create_variables()
        self.target_critic.create_variables()
        
    def update_targets(self):
        common.soft_variables_update(self.actor.variables, self.target_actor.variables, self.tau)
        common.soft_variables_update(self.critic.variables, self.target_critic.variables, self.tau)

    def action(self, obs: tf.Tensor) -> tf.Tensor:
        output = self.actor(obs)[0]
        if self.use_gumbel:
            output = self.gumbel_softmax_sample(output)
        return output

    @classmethod
    def gumbel_softmax_sample(cls, logits):
        noise = tf.random.uniform(tf.shape(logits))
        gumbel = -tf.math.log(-tf.math.log(noise))
        return tf.math.softmax(gumbel + logits)

    # train critic based on all states and actions of environment
    def train_critic(self, states: List[tf.Tensor], actions: List[tf.Tensor], target_q: tf.Tensor,
                     optimizer: tf.keras.optimizers.Optimizer):
        input = states + actions
        with tf.GradientTape() as tape:
            critic_value = self.critic(input, training=True)[0]
            critic_loss = tf.square(target_q - critic_value)
            loss = tf.reduce_mean(critic_loss)
        optimizer.minimize(loss, self.critic.trainable_variables, tape=tape)
        return critic_loss

    # input shape for each arg: (n_agents, batch_size, dims) 
    def train_actor(self, states: List[tf.Tensor], actions: List[tf.Tensor], optimizer: tf.keras.optimizers.Optimizer):
        observations = states[self.index]
        actions = tf.unstack(actions)
        with tf.GradientTape() as tape:
            new_actions = self.actor(observations, training=True)[0]

            # update the new batch of action in the joint actions
            if self.use_gumbel:
                actions[self.index] = self.gumbel_softmax_sample(new_actions)
            else:
                actions[self.index] = new_actions

            q_value = self.critic(states + actions)[0]
            regularization = tf.reduce_mean(tf.square(new_actions))
            loss = -tf.math.reduce_mean(q_value)  + 1e-3 * regularization
        optimizer.minimize(loss, self.actor.trainable_variables, tape=tape)
        return loss

    def save(self, dir: str):
        for network in (self.actor, self.critic, self.target_actor, self.target_critic):
            net_path = os.path.join(dir, network.name)
            np.save(net_path, np.array(network.get_weights(), dtype=list))
    
    def load(self, dir: str):
        for network in (self.actor, self.critic, self.target_actor, self.target_critic):
            if not network.built:
                network.create_variables()
            
            weights_file = os.path.join(dir, network.name + '.npy')
            if os.path.exists(weights_file):
                network.set_weights(np.load(weights_file, allow_pickle=True))
            else:
                print('No weights to load in', weights_file, '. Creating weights.')

class MADDPG:
    """MADDPG Algorithm"""
    def __init__(self,
                 obs_spaces: List[Space],
                 act_spaces: List[Space], 
                 alpha=1e-2,
                 beta=1e-2):
        assert len(obs_spaces) == len(act_spaces)

        self.num_agents = len(obs_spaces)
        self.actor_optimizer = tf.keras.optimizers.Adam(alpha) #, clipnorm=0.5)
        self.critic_optimizer = tf.keras.optimizers.Adam(beta) #, clipnorm=0.5)

        self.agents = [Agent(obs_spaces, act_spaces, i) for i in range(self.num_agents)]

    def __len__(self):
        return len(self.agents)

    def initialize(self):
        for agent in self.agents:
            agent.initialize()

    # state = (n_agents, dimentions)
    def action(self, states: List[Union[np.ndarray, tf.Tensor]]) -> np.ndarray:
        states = [tf.convert_to_tensor(state, tf.float32)[None] for state in states]
        actions = [agent.action(obs).numpy()[0] for agent, obs in  zip(self.agents, states)]
        return actions

    def train(self, experience: Tuple[np.ndarray, ...]):
        
        states, actions, rewards, new_states, dones = experience
        
        dones = [tf.convert_to_tensor(done) for done in dones]
        states = [tf.convert_to_tensor(state) for state in states]
        actions = [tf.convert_to_tensor(action) for action in actions]
        rewards = [tf.convert_to_tensor(reward) for reward in rewards]
        new_states = [tf.convert_to_tensor(new_state) for new_state in new_states]

        # pass to each agent's target actor is own batch of observations
        new_actions = [agent.target_actor(new_states[i])[0] for i, agent in enumerate(self.agents)]
        losses = []
        for i, agent in enumerate(self.agents):
            # Take critic evaluation from new state and taking new actions
            new_critic_value = agent.target_critic(new_states + new_actions)[0]
            target = rewards[i][:, None] + agent.gamma*new_critic_value

            critic_loss = agent.train_critic(states, actions, target, self.critic_optimizer)
            actor_loss = agent.train_actor(states, actions, self.actor_optimizer)

            agent.update_targets()
            losses.append((critic_loss, actor_loss))
        return losses

    def save(self, directory: str):
        for agent in self.agents:
            agent.save(directory)
    
    def load(self, directory: str):
        for agent in self.agents:
            agent.load(directory)
    
    @staticmethod
    def softmax_to_argmax(actions: List[np.ndarray], action_spaces: List[Space]):
        hard_action_n = []
        for action, act_space in zip(actions, action_spaces):
            hard_action_n.append(tf.keras.utils.to_categorical(np.argmax(action), act_space.shape[0]))
        return hard_action_n

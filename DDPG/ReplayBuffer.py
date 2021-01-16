import numpy as np
import tensorflow as tf


class ReplayBuffer:
    def __init__(self, mem_size, obs_shape, action_components):
        # Tamanho maximo da memoria e contador da quantidade
        # de experiencias que ja foram armazenadas
        self.mem_size = mem_size
        self.mem_counter = 0

        # Salvando as observacoes, acoes, recompensas e dones como ndarrays
        # Obs:. Dependendo do ambiente, essa implementacao pode utilizar muita
        # memoria (No caso de ambientes que possuem algum tipo de otimizacao de memoria)
        self.observations = np.zeros((self.mem_size, *obs_shape), dtype=np.float32)
        self.actions = np.zeros((self.mem_size, action_components), dtype=np.float32)
        self.rewards = np.zeros(self.mem_size, dtype=np.float32)
        self.new_observations = np.zeros((self.mem_size, *obs_shape), dtype=np.float32)
        self.terminals = np.zeros(self.mem_size, dtype=np.int32)

    def add_experience(self, obs, act, rew, new_obs, done):
        mem_index = self.mem_counter % self.mem_size

        self.observations[mem_index] = obs
        self.actions[mem_index] = act
        self.rewards[mem_index] = rew
        self.new_observations[mem_index] = new_obs
        self.terminals[mem_index] = (1 - int(done))

        self.mem_counter += 1

    def sample_batch(self, batch_size, convert_to_tensors=False):
        # Garantindo que nao iremos fazer um sample de uma quantidade
        # de experiences superior ao que temos armazenado
        max_mem = min(self.mem_counter, self.mem_size)

        batch = np.random.choice(max_mem, batch_size, replace=False)

        observations = self.observations[batch]
        actions = self.actions[batch]
        rewards = self.rewards[batch]
        new_observations = self.new_observations[batch]
        terminals = self.terminals[batch]

        if convert_to_tensors:
            observations = tf.convert_to_tensor(observations, dtype=tf.float32)
            actions = tf.convert_to_tensor(actions, dtype=tf.float32)
            rewards = tf.convert_to_tensor(rewards, dtype=tf.float32)
            new_observations = tf.convert_to_tensor(new_observations, dtype=tf.float32)
            terminals = tf.convert_to_tensor(terminals, dtype=tf.float32)

        return observations, actions, rewards, new_observations, terminals

    def __len__(self):
        # Caso seja preciso calcular o tamanho do buffer (usado em Agent)
        return self.mem_counter

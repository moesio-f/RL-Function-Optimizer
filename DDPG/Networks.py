import os
import tensorflow as tf
import tensorflow.keras as keras


class CriticNetwork(keras.Model):
    def __init__(self, units1=512, units2=512, model_name='critic', save_dir='tmp/ddpg'):
        super(CriticNetwork, self).__init__()

        self.units1 = units1
        self.units2 = units2

        self.model_name = model_name
        self.checkpoint_dir = save_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, self.model_name + '_ddpg.h5')

        self.dense1 = keras.layers.Dense(units=self.units1, activation='relu')
        self.dense2 = keras.layers.Dense(units=self.units2, activation='relu')
        self.q = keras.layers.Dense(units=1, activation=None)

    def load_model(self):
        self.load_weights(self.checkpoint_file)

    def save_model(self):
        self.save_weights(self.checkpoint_file)

    def call(self, observation, action):
        action_value = self.dense1(tf.concat([observation, action], axis=1))
        action_value = self.dense2(action_value)
        q = self.q(action_value)

        return q


class ActorNetwork(keras.Model):
    def __init__(self, action_components, units1=512, units2=512,
                 model_name='actor', save_dir='tmp/ddpg'):
        super(ActorNetwork, self).__init__()

        self.units1 = units1
        self.units2 = units2
        self.action_components = action_components

        self.model_name = model_name
        self.checkpoint_dir = save_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, self.model_name + '_ddpg.h5')

        self.dense1 = keras.layers.Dense(units=self.units1, activation='relu')
        self.dense2 = keras.layers.Dense(units=self.units2, activation='relu')
        self.mu = keras.layers.Dense(units=self.action_components, activation='tanh')

    def load_model(self):
        self.load_weights(self.checkpoint_file)

    def save_model(self):
        self.save_weights(self.checkpoint_file)

    def call(self, observation):
        prob = self.dense1(observation)
        prob = self.dense2(prob)

        # if action is not bound by +- 1, multiply here
        mu = self.mu(prob)

        return mu

import os
import tensorflow as tf
import tensorflow.keras as keras

# Booleana para salvar diretamente no GDrive se
# estiver no Colab (Adiciona o caminho no nome do modelo)
ON_COLAB = False


class CriticNetwork(keras.Model):
    def __init__(self, units1=512, units2=512, model_name='critic.h5'):
        super(CriticNetwork, self).__init__()

        self.units1 = units1
        self.units2 = units2

        self.model_name = model_name if not ON_COLAB else f"/content/drive/MyDrive/{model_name}"

        self.dense1 = keras.layers.Dense(units=self.units1, activation='relu')
        self.dense2 = keras.layers.Dense(units=self.units2, activation='relu')
        self.q = keras.layers.Dense(units=1, activation='linear')

    def load_model(self):
        self.load_weights(self.model_name)

    def save_model(self):
        self.save_weights(self.model_name)

    def call(self, observation, action):
        action_value = self.dense1(tf.concat([observation, action], axis=1))
        action_value = self.dense2(action_value)
        q = self.q(action_value)

        # Retorna o q-value para esse par (observacao, acao)
        return q


class ActorNetwork(keras.Model):
    def __init__(self, action_components, units1=512, units2=512, model_name='actor.h5'):
        super(ActorNetwork, self).__init__()

        self.units1 = units1
        self.units2 = units2
        self.action_components = action_components

        self.model_name = model_name if not ON_COLAB else f"/content/drive/MyDrive/{model_name}"

        self.dense1 = keras.layers.Dense(units=self.units1, activation='relu')
        self.dense2 = keras.layers.Dense(units=self.units2, activation='relu')
        self.mu = keras.layers.Dense(units=self.action_components, activation='tanh')

    def load_model(self):
        self.load_weights(self.model_name)

    def save_model(self):
        self.save_weights(self.model_name)

    def call(self, observation):
        prob = self.dense1(observation)
        prob = self.dense2(prob)

        # Caso os limites da acao nao sejam (-1, 1), pode multiplicar aqui
        mu = self.mu(prob)

        # Retorna uma acao com a quantidade de componentes
        # Obs:. O valor de cada componente pertence ao intervalo (-1, 1)
        return mu

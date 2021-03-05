from DDPG_ReplayBuffer import *
from DDPG_Networks import *
from datetime import datetime


class DDPGAgent:
    def __init__(self, input_dims, action_components,
                 high=1.0, low=-1.0, alpha=0.0001, beta=0.001, tau=0.001,
                 discount=0.99, units1=400, units2=300, buffer_size=900000,
                 batch_size=64, noise_start=0.1, noise_min=0.06, noise_schedule=10000,
                 min_experience=None):

        # Alguns hiperparametros
        self.discount = discount
        self.tau = tau
        self.alpha = alpha
        self.beta = beta

        # Inicializacao das variaveis relacionadas com o buffer de experience
        self.experience = ReplayBuffer(mem_size=buffer_size, obs_shape=input_dims, action_components=action_components)
        self.batch_size = batch_size
        self.min_experience = self.batch_size
        if min_experience is not None:
            self.min_experience = min_experience

        # Adicionando a quantidade de ruido e a quantidade de componentes da acao
        # Componentes no sentido de componentes/dimensao de um vetor
        # Exemplo:. [a, b, c] possui 3 componentes/dimensao
        self.action_components = action_components
        self.noise = noise_start
        self.noise_min = noise_min
        self.noise_decay = -(noise_start - noise_min)/noise_schedule

        self.high = tf.constant(high, dtype=tf.float32)
        self.low = tf.constant(low, dtype=tf.float32)

        self.actor = ActorNetwork(action_components=self.action_components, units1=units1, units2=units2, const=high)
        self.critic = CriticNetwork(units1=units1, units2=units2)
        self.target_actor = ActorNetwork(action_components=self.action_components, units1=units1, units2=units2,
                                         model_name='target_actor.h5', const=high)
        self.target_critic = CriticNetwork(units1=units1, units2=units2, model_name='target_critic.h5')

        # Otimizadores que serão utilizados pelo actor e peloo critic
        self.actor_optimizer = keras.optimizers.Adam(lr=self.alpha, clipnorm=1.0)
        self.critic_optimizer = keras.optimizers.Adam(lr=self.beta, clipnorm=1.0)

        # Passando um tensor aleatorio pelas redes (buildar as redes)
        random_observation = tf.random.normal(shape=[1, *input_dims])
        random_action = tf.random.normal(shape=[1, action_components])
        self.actor(random_observation)
        self.target_actor(random_observation)
        self.critic(random_observation, random_action)
        self.target_critic(random_observation, random_action)

        # hard update, copiar os pesos das redes online para os targets
        self.update_networks(tau=1.0)

    def set_low_and_high(self, low, high):
        self.high = tf.constant(high, dtype=tf.float32)
        self.low = tf.constant(low, dtype=tf.float32)
        self.actor.set_const(1.0)
        self.target_actor.set_const(1.0)

    def update_target_actor(self, tau):
        weights = []
        targets = self.target_actor.weights
        for i, weight in enumerate(self.actor.weights):
            res = weight * tau + targets[i] * (1 - tau)
            weights.append(res)
        self.target_actor.set_weights(weights)

    def update_target_critic(self, tau):
        weights = []
        targets = self.target_critic.weights
        for i, weight in enumerate(self.critic.weights):
            res = weight * tau + targets[i] * (1 - tau)
            weights.append(res)
        self.target_critic.set_weights(weights)

    def update_networks(self, tau):
        # Estou vendo uma implementação melhor pra essa parte do soft update
        # Acredito que seja mais simples transformar numa ndarray e utilizar
        # as operações padrão no lugar de usar um for (ainda n testei, ai n vou mudar)
        self.update_target_actor(tau)
        self.update_target_critic(tau)

    def memorize(self, obs, act, rew, new_obs, done):
        self.experience.add_experience(obs, act, rew, new_obs, done)

    def save_model(self, make_dir=False):
        directory = None
        if make_dir:
            now = datetime.now()
            directory = now.strftime("models_%d-%m-%Y_%H-%M")

        # Salva todos os pesos das redes
        print('-----saving models-----')
        self.actor.save_model(directory)
        self.critic.save_model(directory)
        self.target_actor.save_model(directory)
        self.target_critic.save_model(directory)
        print('-----models saved-----')

    def load_model(self):
        # Carrega todos os pesos das redes
        print('-----loading models-----')
        self.actor.load_model()
        self.critic.load_model()
        self.target_actor.load_model()
        self.target_critic.load_model()
        print('-----models loaded-----')

    def choose_action(self, observation, current_position, training=True):
        # Converte a observaca para um tensor e faz um pass foward no actor
        batched_obs = tf.convert_to_tensor([observation], dtype=tf.float32)
        action = self.actor(batched_obs)[0]
        if training:
            # Caso esteja em treino, adicionamos "ruido" na acao
            # Assim conseguimos balancear o exploration-exploitation
            action += tf.random.normal(shape=[self.action_components], mean=0.0,
                                       stddev=self.noise)
            self.noise = self.noise - self.noise_decay
            if self.noise < self.noise_min:
                self.noise = self.noise

        # max_action = self.high - current_position
        # min_action = self.low - current_position
        # action = tf.clip_by_value(action, clip_value_min=min_action, clip_value_max=max_action)

        return action

    def __get_critic_loss(self, observations, actions, rewards, new_observations, terminals):
        target_actions = self.target_actor(new_observations)
        target_critic_values = tf.squeeze(self.target_critic(new_observations, target_actions), axis=1)
        critic_value = tf.squeeze(self.critic(observations, actions), axis=1)
        target = tf.stop_gradient(rewards + self.discount * terminals * target_critic_values)
        critic_loss = tf.reduce_sum(keras.losses.MSE(target, critic_value)) / self.batch_size

        return critic_loss

    def __get_actor_loss(self, observations):
        new_policy_actions = self.actor(observations)
        actor_loss = -self.critic(observations, new_policy_actions)
        actor_loss = tf.math.reduce_mean(actor_loss)

        return actor_loss

    def learn(self):
        if len(self.experience) < self.min_experience:
            return

        # Extrai um minibatch do buffer de experience (como tensores)
        observations, actions, rewards, new_observations, terminals = \
            self.experience.sample_batch(self.batch_size, convert_to_tensors=True)

        with tf.GradientTape() as critic_tape:
            critic_tape.watch(self.critic.trainable_variables)
            critic_loss = self.__get_critic_loss(observations, actions, rewards, new_observations, terminals)

        # Aplicando os gradientes
        critic_grads = critic_tape.gradient(critic_loss, self.critic.trainable_variables)
        self.critic_optimizer.apply_gradients(zip(critic_grads, self.critic.trainable_variables))

        with tf.GradientTape() as actor_tape:
            actor_tape.watch(self.actor.trainable_variables)
            actor_loss = self.__get_actor_loss(observations)

        # Aplicando o gradiente
        actor_grads = actor_tape.gradient(actor_loss, self.actor.trainable_variables)
        self.actor_optimizer.apply_gradients(zip(actor_grads, self.actor.trainable_variables))

        # Realizando um soft update com o tau definido na classe
        self.update_networks(tau=self.tau)

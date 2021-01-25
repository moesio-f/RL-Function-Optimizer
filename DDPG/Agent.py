from DDPG.ReplayBuffer import *
from DDPG.Networks import *


class DDPGAgent:
    def __init__(self, input_dims, action_components,
                 max_action, min_action, alpha=0.001, beta=0.002, tau=0.005,
                 discount=0.99, units1=400, units2=300, buffer_size=1000000,
                 batch_size=32, noise=0.08, min_experience=None):

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
        self.noise = noise

        # Limites dos valores das acoes
        self.max_action = max_action
        self.min_action = min_action

        self.actor = ActorNetwork(action_components=self.action_components, units1=units1, units2=units2)
        self.critic = CriticNetwork(units1=units1, units2=units2)
        self.target_actor = ActorNetwork(action_components=self.action_components, units1=units1, units2=units2,
                                         model_name='target_actor.h5')
        self.target_critic = CriticNetwork(units1=units1, units2=units2, model_name='target_critic.h5')

        # Otimizadores que seram utilizados pelo actor e peo critic
        self.actor_optimizer = keras.optimizers.Adam(lr=self.alpha)
        self.critic_optimizer = keras.optimizers.Adam(lr=self.beta)

        # Passando um tensor aleatorio pelas redes (buildar as redes)
        random_observation = tf.random.normal(shape=[1, *input_dims])
        random_action = tf.random.normal(shape=[1, action_components])
        self.actor(random_observation)
        self.target_actor(random_observation)
        self.critic(random_observation, random_action)
        self.target_critic(random_observation, random_action)

        # hard update, copiar os pesos das redes online para os targets
        self.update_networks(tau=1.0)

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

    def save_model(self):
        # Salva todos os pesos das redes
        print('-----saving models-----')
        self.actor.save_model()
        self.critic.save_model()
        self.target_actor.save_model()
        self.target_critic.save_model()
        print('-----models saved-----')

    def load_model(self):
        # Carrega todos os pesos das redes
        print('-----loading models-----')
        self.actor.load_model()
        self.critic.load_model()
        self.target_actor.load_model()
        self.target_critic.load_model()
        print('-----models loaded-----')

    def choose_action(self, observation, training=True):
        # Converte a observaca para um tensor e faz um pass foward no actor
        batched_obs = tf.convert_to_tensor([observation], dtype=np.float32)
        actions = self.actor(batched_obs)
        if training:
            # Caso esteja em treino, adicionamos "ruido" na acao
            # Assim conseguimos balancear o exploration-exploitation
            actions += tf.random.normal(shape=[self.action_components], mean=0.0,
                                        stddev=self.noise)

        # Garantindo que a adicao de ruido não vai fazer as acoes extrapolarem os limites do ambiente
        actions = tf.clip_by_value(actions, clip_value_min=self.min_action, clip_value_max=self.max_action)

        # A rede retorna um tensor na forma (batch_size, *action_components)
        # Retornar um tensor com forma (*action_components)
        return actions[0]

    def learn(self):
        if len(self.experience) < self.min_experience:
            return

        # Extrai um minibatch do buffer de experience (como tensores)
        observations, actions, rewards, new_observations, terminals = \
            self.experience.sample_batch(self.batch_size, convert_to_tensors=True)

        with tf.GradientTape() as critic_tape:
            # Obtendo as acoes que o target actor decidiu tomar nas novas observacoes
            target_actions = self.target_actor(new_observations)
            # Obtendo os valores que o target critic calculou com base nas
            # novas observacoes e nas acoes tomadas pelo target actor
            target_critic_values = tf.squeeze(self.target_critic(new_observations, target_actions), axis=1)
            # Obtendo os valores que o critic previu para as observacoes (estados) atuais
            # com acoes que foram tomadas pelo agente (actor)
            critic_value = tf.squeeze(self.critic(observations, actions), axis=1)
            # Calculando quais deveriam ter sido os valores que o critic deveria ter previsto
            target = rewards + self.discount * terminals * target_critic_values
            # Calcular a perda entre o esperado e o previsto pelo critic
            critic_loss = keras.losses.MSE(target, critic_value)

        # Aplicando os gradientes
        critic_grads = critic_tape.gradient(critic_loss, self.critic.trainable_variables)
        self.critic_optimizer.apply_gradients(zip(critic_grads, self.critic.trainable_variables))

        with tf.GradientTape() as actor_tape:
            # Calculando as acoes que o actor tomaria para as observacoes atuais
            new_policy_actions = self.actor(observations)
            # Utilizando o critic para saber o qual bom foram as acoes e ja determinar
            # a funcao objetiva que precisa ser maximizada (Como usamos gradient descent,
            # precisamos adicionar o - na frente)
            actor_loss = -self.critic(observations, new_policy_actions)
            # Calculando a media desses valores e reduzindo para um escalar
            actor_loss = tf.math.reduce_mean(actor_loss)

        # Aplicando o gradiente
        actor_grads = actor_tape.gradient(actor_loss, self.actor.trainable_variables)
        self.actor_optimizer.apply_gradients(zip(actor_grads, self.actor.trainable_variables))

        # Realizando um soft update com o tau definido na classe
        self.update_networks(tau=self.tau)

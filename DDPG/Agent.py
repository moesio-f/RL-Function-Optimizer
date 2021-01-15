from ReplayBuffer import *
from Networks import *


class DDPGAgent:
    def __init__(self, input_dims, action_components,
                 env=None, alpha=0.001, beta=0.002, tau=0.005,
                 discount=0.99, units1=400, units2=200, buffer_size=1000000,
                 batch_size=64, noise=0.1, min_experience=None):

        self.discount = discount
        self.tau = tau
        self.alpha = alpha
        self.beta = beta

        self.experience = ReplayBuffer(mem_size=buffer_size, obs_shape=input_dims, action_components=action_components)
        self.batch_size = batch_size
        self.min_experience = self.batch_size
        if min_experience is not None:
            self.min_experience = min_experience

        self.action_components = action_components
        self.noise = noise

        self.max_action = env.action_space.high[0]
        self.min_action = env.action_space.low[0]

        self.actor = ActorNetwork(action_components=self.action_components, units1=units1, units2=units2)
        self.critic = CriticNetwork(units1=units1, units2=units2)
        self.target_actor = ActorNetwork(action_components=self.action_components, units1=units1, units2=units2,
                                         model_name='target_actor')
        self.target_critic = CriticNetwork(units1=units1, units2=units2, model_name='target_critic')

        self.actor.compile(optimizer=keras.optimizers.Adam(lr=self.alpha))
        self.critic.compile(optimizer=keras.optimizers.Adam(lr=self.beta))
        self.target_actor.compile(optimizer=keras.optimizers.Adam(lr=self.alpha))
        self.target_critic.compile(optimizer=keras.optimizers.Adam(lr=self.beta))

        # Running random data trough networks (i. e., 'building')
        random_observation = tf.random.normal(shape=(1, *input_dims))
        random_action = tf.random.normal(shape=[1, action_components])
        self.actor(random_observation)
        self.target_actor(random_observation)
        self.critic(random_observation, random_action)
        self.target_critic(random_observation, random_action)

        self.update_networks(tau=1.)

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

    def update_networks(self, tau=None):
        if tau is None:
            tau = self.tau

        self.update_target_actor(tau)
        self.update_target_critic(tau)

    def memorize(self, obs, act, rew, new_obs, done):
        self.experience.add_experience(obs, act, rew, new_obs, done)

    def save_model(self):
        print('-----saving models-----')
        self.actor.save_model()
        self.critic.save_model()
        self.target_actor.save_model()
        self.target_critic.save_model()
        print('-----models saved-----')

    def load_model(self):
        print('-----loading models-----')
        self.actor.load_model()
        self.critic.load_model()
        self.target_actor.load_model()
        self.target_critic.load_model()
        print('-----models loaded-----')

    def choose_action(self, observation, training=True):
        batched_obs = tf.convert_to_tensor([observation], dtype=np.float32)
        actions = self.actor(batched_obs)
        if training:
            actions += tf.random.normal(shape=[self.action_components], mean=0.0,
                                        stddev=self.noise)

        actions = tf.clip_by_value(actions, clip_value_min=self.min_action, clip_value_max=self.max_action)

        return actions[0]

    def learn(self):
        if len(self.experience) < self.min_experience:
            return

        observations, actions, rewards, new_observations, terminals = \
            self.experience.sample_batch(self.batch_size, convert_to_tensors=True)

        with tf.GradientTape() as critic_tape:
            target_actions = self.target_actor(new_observations)
            target_critic_values = tf.squeeze(self.target_critic(new_observations, target_actions), axis=1)
            critic_value = tf.squeeze(self.critic(observations, actions), axis=1)
            target = rewards + self.discount * terminals * target_critic_values
            critic_loss = keras.losses.MSE(target, critic_value)

        critic_grads = critic_tape.gradient(critic_loss, self.critic.trainable_variables)
        self.critic.optimizer.apply_gradients(zip(critic_grads, self.critic.trainable_variables))

        with tf.GradientTape() as actor_tape:
            new_policy_actions = self.actor(observations)
            actor_loss = -self.critic(observations, new_policy_actions)
            actor_loss = tf.math.reduce_mean(actor_loss)

        actor_grads = actor_tape.gradient(actor_loss, self.actor.trainable_variables)
        self.actor.optimizer.apply_gradients(zip(actor_grads, self.actor.trainable_variables))

        self.update_networks()

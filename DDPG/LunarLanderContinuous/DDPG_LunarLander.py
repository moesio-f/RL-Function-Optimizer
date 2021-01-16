from Agent import *
import gym
import matplotlib.pyplot as plt


if __name__ == '__main__':
    env = gym.make('LunarLanderContinuous-v2')
    agent = DDPGAgent(input_dims=env.observation_space.shape,
                      action_components=env.action_space.shape[0],
                      min_experience=256, env=env)

    # Network usada pro LunarLanderContinuous:
    # Critic: Dense(400, relu) + Dense(300, relu) + Dense(1, linear)
    # Actor: Dense(400, relu) + Dense(400, relu) + Dense(2, tanh)
    #        Obs:. Na última densa 2 é a quantidade de componentes da ação
    # Os pesos atuais foram extraídos ao final dos 800 episodios
    # Durante 500 episodios de teste, nao apresentou nenhuma recompensa negativa

    episodes = 500
    rewards = []

    # Variaveis para facilitar depuracao e controle
    # Caso esteja treinando, iremos armazenar as experiencias no buffer, rodar a funcao para aprender,
    # salvar os pesos a cada 25 episodios e salvar os pesos ao final de todos episodios.
    # Caso não esteja treinando, nao precisamos armazenar experiencias, nem salvar os pesos e nem rodar
    # a funcao para aprender
    training = False
    # Controla se vamos carregar os pesos atuais ou nao
    load_models = True
    if load_models:
        agent.load_model()

    for ep in range(episodes):
        done = False
        game_reward = 0
        obs = env.reset()
        while not done:
            # Garantindo que nao iremos adicionar ruido caso estejamos testando.
            # DDPG produz uma policy deterministica
            action = agent.choose_action(obs, training=training)
            new_obs, reward, done, info = env.step(action)

            if training:
                agent.memorize(obs, action, reward, new_obs, done)
                agent.learn()

            game_reward += reward
            obs = new_obs

        rewards.append(game_reward)

        if ep % 25 == 0 and training:
            agent.save_model()

        avg_score = np.mean(rewards[-100:])
        print('episode: ', ep, 'score %.1f' % game_reward,
              'average_score %.1f' % avg_score)

    if training:
        agent.save_model()
    fig, ax = plt.subplots()
    ax.plot(range(len(rewards)), rewards)
    ax.set(xlabel="Number episodes", ylabel="Reward", title="DDPG on LunarLanderContinuous-v2 (Evaluating)")
    ax.grid()
    plt.show()

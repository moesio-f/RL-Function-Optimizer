from Agent import *
import gym
import matplotlib.pyplot as plt


if __name__ == '__main__':
    env = gym.make('LunarLanderContinuous-v2')
    agent = DDPGAgent(input_dims=env.observation_space.shape,
                      action_components=env.action_space.shape[0],
                      min_experience=5000, env=env)

    episodes = 500
    rewards = []

    load_models = False
    if load_models:
        agent.load_model()

    for ep in range(episodes):
        done = False
        game_reward = 0
        obs = env.reset()
        while not done:
            action = agent.choose_action(obs)
            new_obs, reward, done, info = env.step(action)

            agent.memorize(obs, action, reward, new_obs, done)
            agent.learn()

            game_reward += reward
            obs = new_obs

        rewards.append(game_reward)

        if ep % 50 == 0:
            agent.save_model()

        avg_score = np.mean(rewards[-100:])
        print('episode: ', ep, 'score %.1f' % game_reward,
              'average_score %.1f' % avg_score)

    agent.save_model()
    fig, ax = plt.subplots()
    ax.plot(range(len(rewards)), rewards)
    ax.set(xlabel="Number episodes", ylabel="Reward", title="DDQN on LunarLanderContinuous-v2")
    ax.grid()
    plt.show()

from matplotlib import pyplot as plt
import numpy as np
from tf_agents.policies.policy_saver import PolicySaver
from utils.multi_agent.replay_bufer import MultiAgentReplayBuffer
from agents.maddpg import MADDPG


from environments.gym_env import MultiAgentFunctionEnv
from functions.numpy_functions import *

def main():
    BATCH_SIZE = 32
    N_EPISODES = 10_000
    N_STEPS = 25
    N_AGENTS = 20
    DIMS = 20
    display = False

    FUNCTION = Sphere()
    N_ACTIONS = DIMS

    env = MultiAgentFunctionEnv(FUNCTION, DIMS, N_AGENTS, True)

    maddpg_agents = MADDPG(env.observation_space, env.action_space)
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

from environments.gym_env import SimpleMultiAgentEnv, MultiAgentFunctionEnv

def test_simple_env():
    BATCH_SIZE = 32
    STEPS = 25
    UPDATE_RATE = 100
    display = True
    total_steps = 0
    best_score = float('-inf')

    env = SimpleMultiAgentEnv(2)
    
    agents = MADDPG(env.observation_space, env.action_space)
    agents.initialize()
    memory = MultiAgentReplayBuffer(10000, len(agents))

    for ep in range(10_000):
        states = env.reset()
        for step in range(30):
            if display:
                plt.pause(0.05)
                env.render()

            actions = agents.action(states)
            next_states, rewards, dones, infos = env.step(actions)
            memory.add(states, actions, rewards, next_states, dones)
            if (len(memory) > BATCH_SIZE * STEPS and total_steps % UPDATE_RATE == 0):
                agents.train(memory.sample(BATCH_SIZE))
            total_steps += 1
            if step % 10 == 0:
                print('ep:', ep, 'reward:', rewards)
    print(rewards)


if __name__ == '__main__':
    test_simple_env()
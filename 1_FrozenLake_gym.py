# %%
import numpy as np
import gymnasium as gym
import random

#%%
# 這邊可以列出所有可用的環境
#print('\n'.join([str(env) for env in gym.envs.registry]))

#%%
env = gym.make('FrozenLake-v1', is_slippery=False, render_mode='human')
action_size=env.action_space.n
state_size=env.observation_space.n
print(f"Action Size: {action_size}, State Size: {state_size}")
#%%
qtable=np.zeros((state_size,action_size))

#%%
episode_rewards = []
for i in range(10000):
    state, *_ = env.reset()
    total_rewards = 0
    for step in range(50):
        action = env.action_space.sample()
        new_state, reward, done, *_ = env.step(action)
        qtable[state,action]+=0.1*(reward+0.9*np.max(qtable[new_state,:])-qtable[state,action])
        state=new_state
        total_rewards+=reward
    episode_rewards.append(total_rewards)
print(qtable)
#%%
episode_rewards = []
epsilon=1
max_epsilon=1
min_epsilon=0.01
decay_rate=0.005
for episode in range(1000):
    state,*_=env.reset()
    #env.render()
    total_rewards = 0
    for step in range(50):
        exp_exp_tradeoff=random.uniform(0,1)
        ## Exploitation:
        if exp_exp_tradeoff>epsilon:
            action=np.argmax(qtable[state,:])
        else:
            ## Exploration
            action=env.action_space.sample()
        new_state,reward,done,*_=env.step(action)
        qtable[state,action]+=0.9*(reward+0.9*np.max(qtable[new_state,:])-qtable[state,action])
        state=new_state
        total_rewards+=reward
    episode_rewards.append(total_rewards)
    epsilon=min_epsilon+(max_epsilon-min_epsilon)*np.exp(decay_rate*episode)
print(qtable)

#from torch_snippets_ import show
env.reset()
for episode in range(1):
    state, *_ = env.reset()
    step=0
    done=False
    print("-----------------------")
    print("Episode",episode)
    for step in range(50):
        env.render()
        action=np.argmax(qtable[state,:])
        print(action)
        new_state,reward,done,*_=env.step(action)
        if done:
            env.render()
            print("Number of Steps", step+1)
            break
        state=new_state
env.close()
input("Press Enter to exit...")  # Keep the window open until user input
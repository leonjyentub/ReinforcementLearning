import numpy as np
#import gym
import gymnasium as gym
import random
env = gym.make('FrozenLake-v1', is_slippery=False, render_mode='human')

env.reset()
env.render()

env.step(env.action_space.sample())
input("Press Enter to continue...")
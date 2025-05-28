'''
這個是書上的例子
https://github.com/PacktPublishing/Modern-Computer-Vision-with-PyTorch-2E/blob/main/Chapter14/Deep_Q_Learning_Cart_Pole_balancing.ipynb
'''
import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
import numpy as np
from collections import namedtuple, deque
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# DQN
class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_dim, 32)
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, action_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)
    
# DQN Agent
class DQNAgent:
    def __init__(self, state_size, action_size):

        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(0)

        ## hyperparameters
        self.buffer_size = 2000
        self.batch_size = 64
        self.gamma = 0.99
        self.lr = 0.0025
        self.update_every = 4

        # Q-Network
        self.local = DQN(state_size, action_size).to(device)
        self.optimizer = optim.Adam(self.local.parameters(), lr=self.lr)

        # Replay memory
        self.memory = deque(maxlen=self.buffer_size)
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.t_step = 0

    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        self.memory.append(self.experience(state, action, reward, next_state, done))
        # Learn every update_every time steps.
        self.t_step = (self.t_step + 1) % self.update_every
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > self.batch_size:
                experiences = self.sample_experiences()
                self.learn(experiences, self.gamma)
    def act(self, state, eps=0.):
        # Epsilon-greedy action selection
        if random.random() > eps:
            state = torch.from_numpy(state).float().unsqueeze(0).to(device)
            self.local.eval()
            with torch.no_grad():
                action_values = self.local(state)
            self.local.train()
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))
    def learn(self, experiences, gamma):
        states, actions, rewards, next_states, dones = experiences
       # Get expected Q values from local model
        Q_expected = self.local(states).gather(1, actions)

        # Get max predicted Q values (for next states) from local model
        Q_targets_next = self.local(next_states).detach().max(1)[0].unsqueeze(1)
        # Compute Q targets for current states
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

        # Compute loss
        loss = F.mse_loss(Q_expected, Q_targets)

        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
    def sample_experiences(self):
        experiences = random.sample(self.memory, k=self.batch_size)
        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
        return (states, actions, rewards, next_states, dones)

def main():
    env = gym.make("CartPole-v1", render_mode="human")
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = DQNAgent(state_size, action_size)
    scores = [] # list containing scores from each episode
    scores_window = deque(maxlen=100) # last 100 scores
    n_episodes=5000
    max_t=5000
    eps_start=1.0
    eps_end=0.001
    eps_decay=0.9995
    eps = eps_start
    target_update_freq = 10

    
    for i_episode in range(1, n_episodes+1):
        state, *_ = env.reset()
        state = np.reshape(state, [1, state_size]) #有必要reshape成二維嗎？
        score = 0
        for i in range(max_t):
            env.render()
            action = agent.act(state, eps)
            next_state, reward, done, *_ = env.step(action)
            next_state = np.reshape(next_state, [1, state_size]) 
            reward = reward if not done or score == 499 else -10
            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            if done:
                break
            # 中文說明
            action_str = "向左移動" if action == 0 else "向右移動"
            obs_str = f"推車位置: {next_state[0][0]:.2f}, 速度: {next_state[0][1]:.2f}, 棒子角度: {next_state[0][2]:.2f}, 角速度: {next_state[0][3]:.2f}"
            print(f"回合: {i_episode}，動作: {action_str}，觀察值: [{obs_str}]，獎勵: {reward}")
        scores_window.append(score) # save most recent score
        scores.append(score) # save most recent score
        eps = max(eps_end, eps_decay*eps) # decrease epsilon
        print('\rEpisode {}\tReward {} \tAverage Score: {:.2f} \tEpsilon: {:.2f}'.format(i_episode,score,np.mean(scores_window), eps), end="")
        #if i_episode % 100 == 0:
            #print('\rEpisode {}\tAverage Score: {:.2f} \tEpsilon: {:.2f}'.format(i_episode, np.mean(scores_window), eps))

        if i_episode>10 and np.mean(scores[-10:])>450:
            break
    print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
    env.close()


if __name__ == "__main__":
    main()
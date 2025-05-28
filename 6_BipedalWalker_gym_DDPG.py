import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque

'''
env 說明

BipedalWalker-v3 是 OpenAI Gym 提供的經典強化學習環境之一，
目標是讓一個雙足機器人學會在崎嶇地形上行走。
學習的目的是讓 agent 控制四個腿部馬達（連續動作空間，每個動作範圍為 [-1, 1]），
使機器人能夠平衡並前進到終點。
觀察空間（observation space）為 24 維連續數值，包含機器人的速度、角度、腿部接觸狀態等資訊。
每一步的 reward 主要來自於向前移動的距離，跌倒會有額外的負獎勵，成功走到終點會有額外的正獎勵。
訓練方向通常為：讓 agent 學會平衡、避免跌倒、有效率地前進。
常見參數：
- observation_space.shape = (24,)
- action_space.shape = (4,)（四個馬達）
- action_space.low = [-1, -1, -1, -1]
- action_space.high = [1, 1, 1, 1]
- 每個 episode 最多 1600 步
- reward 範圍大致為 [-100, 300]
'''

class QLearningAgent:
    def __init__(self, env, buckets=(3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3), n_actions=5, alpha=0.1, gamma=0.99, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01):
        self.env = env
        self.buckets = buckets  # Discretization for each observation dimension
        self.n_actions = n_actions  # Discretize each action dim into n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.obs_low = env.observation_space.low
        self.obs_high = env.observation_space.high
        self.obs_bins = [np.linspace(self.obs_low[i], self.obs_high[i], self.buckets[i]-1) for i in range(len(self.buckets))]
        self.q_table = dict()

    def discretize(self, obs):
        state = tuple(int(np.digitize(obs[i], self.obs_bins[i])) for i in range(len(self.buckets)))
        return state

    def get_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.uniform(-1, 1, size=self.env.action_space.shape)
        q_vals = self.q_table.get(state)
        if q_vals is None:
            return np.random.uniform(-1, 1, size=self.env.action_space.shape)
        idx = np.argmax(q_vals)
        # Map discrete action index to continuous action
        action = np.linspace(-1, 1, self.n_actions)[idx]
        return np.full(self.env.action_space.shape, action)

    def update(self, state, action, reward, next_state, done):
        idx = int(np.digitize(action[0], np.linspace(-1, 1, self.n_actions)))
        if state not in self.q_table:
            self.q_table[state] = np.zeros(self.n_actions)
        if next_state not in self.q_table:
            self.q_table[next_state] = np.zeros(self.n_actions)
        best_next = np.max(self.q_table[next_state])
        target = reward + (0 if done else self.gamma * best_next)
        self.q_table[state][idx] += self.alpha * (target - self.q_table[state][idx])

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)

class Actor(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, act_dim), nn.Tanh()
        )
    def forward(self, x):
        return self.net(x)

class Critic(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim + act_dim, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, 1)
        )
    def forward(self, x, a):
        return self.net(torch.cat([x, a], dim=-1))

class DDPGAgent:
    def __init__(self, obs_dim, act_dim, act_limit):
        self.actor = Actor(obs_dim, act_dim)
        self.actor_target = Actor(obs_dim, act_dim)
        self.critic = Critic(obs_dim, act_dim)
        self.critic_target = Critic(obs_dim, act_dim)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.actor_optim = optim.Adam(self.actor.parameters(), lr=1e-3)
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=1e-3)
        self.memory = deque(maxlen=100000)
        self.gamma = 0.99
        self.tau = 0.005
        self.act_limit = act_limit

    def get_action(self, obs, noise_scale=0.1):
        obs = torch.FloatTensor(obs).unsqueeze(0)
        action = self.actor(obs).detach().numpy()[0]
        action += noise_scale * np.random.randn(*action.shape)
        return np.clip(action, -self.act_limit, self.act_limit)

    def store(self, obs, act, rew, next_obs, done):
        self.memory.append((obs, act, rew, next_obs, done))

    def sample(self, batch_size=64):
        batch = random.sample(self.memory, batch_size)
        obs, act, rew, next_obs, done = map(np.array, zip(*batch))
        return (torch.FloatTensor(obs),
                torch.FloatTensor(act),
                torch.FloatTensor(rew).unsqueeze(1),
                torch.FloatTensor(next_obs),
                torch.FloatTensor(done).unsqueeze(1))

    def update(self, batch_size=64):
        if len(self.memory) < batch_size:
            return
        obs, act, rew, next_obs, done = self.sample(batch_size)
        with torch.no_grad():
            next_act = self.actor_target(next_obs)
            target_q = self.critic_target(next_obs, next_act)
            target = rew + self.gamma * (1 - done) * target_q
        current_q = self.critic(obs, act)
        critic_loss = nn.MSELoss()(current_q, target)
        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()

        actor_loss = -self.critic(obs, self.actor(obs)).mean()
        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

        # Soft update
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

def train_qlearning(env, episodes=1000):
    agent = QLearningAgent(env)
    for ep in range(episodes):
        obs, info = env.reset()
        state = agent.discretize(obs)
        total_reward = 0
        for t in range(1600):
            env.render()
            action = agent.get_action(state)
            next_obs, reward, terminated, truncated, info = env.step(action)
            next_state = agent.discretize(next_obs)
            agent.update(state, action, reward, next_state, terminated or truncated)
            state = next_state
            total_reward += reward
            if terminated or truncated:
                break
        agent.decay_epsilon()
        print(f"Episode {ep}, Total Reward: {total_reward}, Epsilon: {agent.epsilon:.3f}")

def train_ddpg(env, episodes=5000):
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    act_limit = float(env.action_space.high[0])
    agent = DDPGAgent(obs_dim, act_dim, act_limit)
    for ep in range(episodes):
        obs, info = env.reset()
        total_reward = 0
        for t in range(1600):
            env.render()
            action = agent.get_action(obs)
            next_obs, reward, terminated, truncated, info = env.step(action)
            agent.store(obs, action, reward, next_obs, float(terminated or truncated))
            agent.update()
            obs = next_obs
            total_reward += reward
            if terminated or truncated:
                break
        print(f"Episode {ep}, Total Reward: {total_reward}")

def main():
    env = gym.make("BipedalWalker-v3", render_mode="human")
    print("Select agent: 1) Q-learning  2) DDPG")
    choice = input("Enter 1 or 2: ")
    if choice == "1":
        train_qlearning(env)
    elif choice == "2":
        train_ddpg(env)
    else:
        print("Invalid choice.")
    env.close()

if __name__ == "__main__":
    main()

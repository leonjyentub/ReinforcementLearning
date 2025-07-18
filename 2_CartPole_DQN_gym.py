import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque

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
    def __init__(self, state_dim, action_dim, memory_size=10000, batch_size=64, gamma=0.99, lr=1e-3, epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.memory = deque(maxlen=memory_size)
        self.batch_size = batch_size
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_end
        self.epsilon_decay = epsilon_decay

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_net = DQN(state_dim, action_dim).to(self.device)
        self.target_net = DQN(state_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)

    def select_action(self, state):
        if np.random.rand() < self.epsilon:
            return random.randrange(self.action_dim)
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.policy_net(state)
        return q_values.argmax().item()

    def store(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def update(self):
        if len(self.memory) < self.batch_size:
            return
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        q_values = self.policy_net(states).gather(1, actions)
        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(1, keepdim=True)[0]
            target_q = rewards + (1 - dones) * self.gamma * next_q_values

        loss = nn.MSELoss()(q_values, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Epsilon decay
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def update_target(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

def main_default():
    env = gym.make("CartPole-v1", render_mode="human")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    agent = DQNAgent(state_dim, action_dim)
    num_episodes = 100
    target_update_freq = 10

    for episode in range(num_episodes):
        observation, info = env.reset(seed=None)
        total_reward = 0
        done = False
        step = 0
        while not done:
            env.render()
            action = agent.select_action(observation)
            next_observation, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            agent.store(observation, action, reward, next_observation, done)
            agent.update()
            observation = next_observation
            total_reward += reward
            step += 1

            # 中文說明
            action_str = "向左移動" if action == 0 else "向右移動"
            obs_str = f"推車位置: {observation[0]:.2f}, 速度: {observation[1]:.2f}, 棒子角度: {observation[2]:.2f}, 角速度: {observation[3]:.2f}"
            print(f"回合: {episode+1}，步驟: {step}，動作: {action_str}，觀察值: [{obs_str}]，獎勵: {reward}")

            if done:
                print(f"第 {episode + 1} 回合結束，總獎勵: {total_reward}，已經失去平衡，重新開始！")
                break

        if (episode + 1) % target_update_freq == 0:
            agent.update_target()
            print(f"同步 target network (第 {episode + 1} 回合)")

    env.close()


def main():
    env = gym.make("CartPole-v1", render_mode="human")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    agent = DQNAgent(state_dim, action_dim)
    num_episodes = 300
    target_update_freq = 10

    for episode in range(1, num_episodes + 1):
        state, info = env.reset()
        total_reward = 0
        done = False
        truncated = False
        step = 0
        while not (done or truncated):
            env.render()
            action = agent.select_action(state)
            next_state, reward, done, truncated, info = env.step(action)
            agent.store(state, action, reward, next_state, done or truncated)
            agent.update()
            # 中文說明
            action_str = "向左移動" if action == 0 else "向右移動"
            obs_str = f"推車位置: {next_state[0]:.2f}, 速度: {next_state[1]:.2f}, 棒子角度: {next_state[2]:.2f}, 角速度: {next_state[3]:.2f}"
            print(f"回合: {episode}，動作: {action_str}，觀察值: [{obs_str}]，獎勵: {reward}")
            state = next_state
            total_reward += reward
            step += 1
        if episode % target_update_freq == 0:
            agent.update_target()
        if done or truncated:
            print(f"第 {episode} 回合結束，已經失去平衡，重新開始！")
    env.close()


if __name__ == "__main__":
    main()
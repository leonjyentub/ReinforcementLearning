import numpy as np
import random
import pygame
import pickle  # Add this import for saving/loading Q-table
import torch
import torch.nn as nn
import torch.optim as optim

# Environment
class MazeEnvironment:
    def __init__(self, maze_size=10):
        self.maze_size = maze_size
        self.maze = self.generate_maze()
        self.start = (0, 0)
        self.goal = (maze_size - 1, maze_size - 1)
        self.state = self.start

    def generate_maze(self):
        maze = np.zeros((self.maze_size, self.maze_size))
        # Randomly place obstacles
        for _ in range(int(self.maze_size * 2)):
            x, y = random.randint(0, self.maze_size - 1), random.randint(0, self.maze_size - 1)
            if (x, y) != (0, 0) and (x, y) != (self.maze_size - 1, self.maze_size - 1):
                maze[x, y] = 1  # 1 represents an obstacle
        return maze

    def step(self, action):
        # Actions: 0 = up, 1 = down, 2 = left, 3 = right
        x, y = self.state
        if action == 0 and x > 0:  # Up
            x -= 1
        elif action == 1 and x < self.maze_size - 1:  # Down
            x += 1
        elif action == 2 and y > 0:  # Left
            y -= 1
        elif action == 3 and y < self.maze_size - 1:  # Right
            y += 1

        # Check for obstacles
        if self.maze[x, y] == 1:
            reward = -50  # Penalty for hitting an obstacle
            done = False
        elif (x, y) == self.goal:
            reward = 100  # Reward for reaching the goal
            done = True
        else:
            reward = -1  # Small penalty for each step
            done = False

        self.state = (x, y)
        return self.state, reward, done

    def reset(self):
        self.state = self.start
        return self.state
# Agent
import collections

class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class DQNAgent:
    def __init__(self, state_space, action_space, memory_size=10000, batch_size=64, gamma=0.99, lr=1e-3, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995, target_update=10):
        self.state_dim = state_space[0] * state_space[1]
        self.action_dim = action_space
        self.memory = collections.deque(maxlen=memory_size)
        self.batch_size = batch_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.target_update = target_update
        self.learn_step = 0

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = DQN(self.state_dim, self.action_dim).to(self.device)
        self.target_model = DQN(self.state_dim, self.action_dim).to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()

    def choose_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.action_dim)
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.model(state)
        return q_values.argmax().item()

    def store_transition(self, state, action, reward, next_state, done):
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

        q_values = self.model(states).gather(1, actions)
        with torch.no_grad():
            next_q_values = self.target_model(next_states).max(1, keepdim=True)[0]
            target_q = rewards + self.gamma * next_q_values * (1 - dones)
        loss = self.loss_fn(q_values, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.learn_step += 1

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def decay_exploration(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

# 將 state 轉換為 one-hot 向量

def state_to_onehot(state, maze_size):
    onehot = np.zeros(maze_size * maze_size, dtype=np.float32)
    idx = state[0] * maze_size + state[1]
    onehot[idx] = 1.0
    return onehot

# Main Program
def main():
    pygame.init()
    maze_size = 8
    cell_size = 60
    screen = pygame.display.set_mode((maze_size * cell_size, maze_size * cell_size))
    pygame.display.set_caption("DQN Maze")

    env = MazeEnvironment(maze_size=maze_size)

    agent = DQNAgent(state_space=(maze_size, maze_size), action_space=4)

    clock = pygame.time.Clock()
    episodes = 500
    for episode in range(episodes):
        state = env.reset()
        state_onehot = state_to_onehot(state, maze_size)  # 修正：用 one-hot
        total_reward = 0

        while True:
            # Visualize
            screen.fill((0, 0, 0))
            for x in range(maze_size):
                for y in range(maze_size):
                    color = (255, 255, 255) if env.maze[x, y] == 0 else (0, 0, 255)
                    pygame.draw.rect(screen, color, (y * cell_size, x * cell_size, cell_size, cell_size))
            pygame.draw.rect(screen, (0, 255, 0), (env.goal[1] * cell_size, env.goal[0] * cell_size, cell_size, cell_size))
            pygame.draw.rect(screen, (255, 0, 0), (state[1] * cell_size, state[0] * cell_size, cell_size, cell_size))
            pygame.display.flip()

            # Agent chooses action
            action = agent.choose_action(state_onehot)
            next_state, reward, done = env.step(action)
            next_state_onehot = state_to_onehot(next_state, maze_size)

            agent.store_transition(state_onehot, action, reward, next_state_onehot, done)
            agent.update()
            
            state = next_state
            state_onehot = next_state_onehot
            total_reward += reward

            if done:
                break
            # Control frame rate
            clock.tick(30)

        agent.update_target_model()
        agent.decay_exploration()
        print(f"Episode {episode + 1}: Total Reward = {total_reward}")

    pygame.quit()

def visualize_final_path(env, screen, cell_size, q_table_path, maze_path):
    # Load the Q-table and maze from the files
    maze_size = 8
    screen = pygame.display.set_mode((maze_size * cell_size, maze_size * cell_size))
    pygame.display.set_caption("Q-Learning Maze")
    with open(q_table_path, "rb") as f:
        q_table = pickle.load(f)
    with open(maze_path, "rb") as f:
        env.maze = pickle.load(f)
    print("Loaded Q-Table from", q_table_path)
    print("Loaded Maze from", maze_path)

    state = env.reset()
    path = [state]
    while state != env.goal:
        action = np.argmax(q_table[state])  # Use the loaded Q-table
        next_state, _, _ = env.step(action)
        path.append(next_state)
        state = next_state

    # Visualize the path
    screen.fill((0, 0, 0))
    for x in range(env.maze_size):
        for y in range(env.maze_size):
            color = (255, 255, 255) if env.maze[x, y] == 0 else (0, 0, 255)
            pygame.draw.rect(screen, color, (y * cell_size, x * cell_size, cell_size, cell_size))
    pygame.draw.rect(screen, (0, 255, 0), (env.goal[1] * cell_size, env.goal[0] * cell_size, cell_size, cell_size))
    for state in path:
        pygame.draw.rect(screen, (255, 0, 0), (state[1] * cell_size, state[0] * cell_size, cell_size, cell_size))
        pygame.display.flip()
        pygame.time.delay(100)  # Delay to visualize the path step-by-step
    
    pygame.quit()

if __name__ == "__main__":
    main()
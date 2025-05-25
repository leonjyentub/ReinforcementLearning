import numpy as np
import random
import pygame
import pickle  # Add this import for saving/loading Q-table
import time  # Add this import for timestamping filenames
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
class QLearningAgent:
    def __init__(self, state_space, action_space, learning_rate=0.1, discount_factor=0.9, exploration_rate=1.0):
        self.q_table = np.zeros(state_space + (action_space,))
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.exploration_decay = 0.995
        self.min_exploration_rate = 0.01

    def choose_action(self, state):
        if random.uniform(0, 1) < self.exploration_rate:
            return random.choice([0, 1, 2, 3])  # Explore
        return np.argmax(self.q_table[state])  # Exploit

    def update(self, state, action, reward, next_state):
        max_q = np.max(self.q_table[next_state])
        self.q_table[state][action] += self.learning_rate * (
            reward + self.discount_factor * max_q - self.q_table[state][action]
        )

    def decay_exploration(self):
        self.exploration_rate = max(self.min_exploration_rate, self.exploration_rate * self.exploration_decay)

# DQN Model
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, x):
        return self.fc(x)

# DQN Agent
class DQNAgent:
    def __init__(self, state_space, action_space, learning_rate=0.001, discount_factor=0.99, exploration_rate=1.0):
        self.state_space = state_space
        self.action_space = action_space
        self.model = DQN(state_space[0] * state_space[1], action_space)
        self.target_model = DQN(state_space[0] * state_space[1], action_space)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.exploration_decay = 0.995
        self.min_exploration_rate = 0.01
        self.memory = []
        self.batch_size = 64

    def choose_action(self, state):
        if random.uniform(0, 1) < self.exploration_rate:
            return random.choice(range(self.action_space))  # Explore
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            return torch.argmax(self.model(state_tensor)).item()  # Exploit

    def store_transition(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        if len(self.memory) > 10000:  # Limit memory size
            self.memory.pop(0)

    def update(self):
        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions).unsqueeze(1)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones)

        q_values = self.model(states).gather(1, actions).squeeze()
        with torch.no_grad():
            max_next_q_values = self.target_model(next_states).max(1)[0]
            target_q_values = rewards + self.discount_factor * max_next_q_values * (1 - dones)

        loss = self.criterion(q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def decay_exploration(self):
        self.exploration_rate = max(self.min_exploration_rate, self.exploration_rate * self.exploration_decay)

# Main Program
def main():
    pygame.init()
    maze_size = 8
    cell_size = 60
    screen = pygame.display.set_mode((maze_size * cell_size, maze_size * cell_size))
    pygame.display.set_caption("DQN Maze")

    env = MazeEnvironment(maze_size=maze_size)
    use_dqn = True  # Set to True to use DQN, False for Q-Learning

    if use_dqn:
        agent = DQNAgent(state_space=(maze_size, maze_size), action_space=4)
    else:
        agent = QLearningAgent(state_space=(maze_size, maze_size), action_space=4)

    clock = pygame.time.Clock()
    episodes = 500
    for episode in range(episodes):
        state = env.reset()
        state_flat = np.array(state).flatten()  # Flatten state for DQN
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
            action = agent.choose_action(state_flat)
            next_state, reward, done = env.step(action)
            next_state_flat = np.array(next_state).flatten()

            if use_dqn:
                agent.store_transition(state_flat, action, reward, next_state_flat, done)
                agent.update()
            else:
                agent.update(state, action, reward, next_state)

            state = next_state
            state_flat = next_state_flat
            total_reward += reward

            if done:
                break

            # Control frame rate
            clock.tick(30)

        if use_dqn and episode % 10 == 0:
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
    #main()
    visualize_final_path(MazeEnvironment(maze_size=10), 
                         QLearningAgent(state_space=(10, 10), action_space=4), 
                                        60,
                                        "q_table_20250524-130913.pkl",
                                        "maze_20250524-130913.pkl")


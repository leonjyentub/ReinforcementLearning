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
        return np.argmax(self.q_table[tuple(state)])  # Exploit 
        # 一個三維的q_Table的索引值是一維陣列，應該是切出一個一維陣列的值，但卻跑出三維陣列
        # GPT的說明是self.q_table[state] 會導致 NumPy 的高級索引（advanced indexing）行為，而不是簡單的切片操作。
        # 這是因為 state 是一個一維陣列，而不是單純的兩個索引值。
        # 在高級索引中，self.q_table[state] 的行為如下：
        # state = np.array([5, 2]) 被解釋為兩個索引值 [5, 2]，分別對應到第一維和第二維。
        # NumPy 將這些索引值視為獨立的索引陣列，而不是一個單一的座標。
        # 結果，NumPy 會從第一維和第二維中分別取出索引 5 和索引 2 的所有值，並組合成一個新的陣列。
        # 這種行為會導致返回的結果是一個三維陣列，而不是你期望的一維陣列。
        # 這裡的 tuple(state) 是為了確保 state 被正確地轉換為索引元組。


    def update(self, state, action, reward, next_state):
        #print(f'Updating Q-value for state {state}, action {action}, reward {reward}, next_state {next_state}')
        max_q = np.max(self.q_table[next_state])
        self.q_table[state][action] += self.learning_rate * (
            reward + self.discount_factor * max_q - self.q_table[state][action]
        )

    def decay_exploration(self):
        self.exploration_rate = max(self.min_exploration_rate, self.exploration_rate * self.exploration_decay)

def main():
    pygame.init()
    maze_size = 8
    cell_size = 60
    screen = pygame.display.set_mode((maze_size * cell_size, maze_size * cell_size))
    pygame.display.set_caption("DQN Maze")

    env = MazeEnvironment(maze_size=maze_size)
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
            
            agent.update(state, action, reward, next_state)

            state = next_state
            state_flat = next_state_flat
            total_reward += reward

            if done:
                break

            # Control frame rate
            clock.tick(30)
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
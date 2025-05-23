import numpy as np
import random
import pygame

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
        for _ in range(int(self.maze_size * 1.5)):
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

# Main Program
def main():
    pygame.init()
    maze_size = 5
    cell_size = 60
    screen = pygame.display.set_mode((maze_size * cell_size, maze_size * cell_size))
    pygame.display.set_caption("Q-Learning Maze")

    env = MazeEnvironment(maze_size=maze_size)
    agent = QLearningAgent(state_space=(maze_size, maze_size), action_space=4)

    clock = pygame.time.Clock()
    episodes = 500
    for episode in range(episodes):
        state = env.reset()
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
            action = agent.choose_action(state)
            next_state, reward, done = env.step(action)

            # Update Q-table
            agent.update(state, action, reward, next_state)
            state = next_state
            total_reward += reward

            if done:
                break

            # Control frame rate
            clock.tick(30)

        # Decay exploration rate
        agent.decay_exploration()

        print(f"Episode {episode + 1}: Total Reward = {total_reward}")
        print("Q-Table:")
        print(agent.q_table)

    pygame.quit()

if __name__ == "__main__":
    main()
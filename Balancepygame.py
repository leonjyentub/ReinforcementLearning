import pygame
import numpy as np
import random

# Environment
class BalanceEnvironment:
    def __init__(self, screen_width=400):
        self.screen_width = screen_width
        self.state = screen_width // 2  # Start in the middle
        self.done = False

    def step(self, action):
        # Action: 0 = move left, 1 = move right
        if action == 0:
            self.state -= 10
        elif action == 1:
            self.state += 10

        # Keep the state within bounds
        self.state = max(0, min(self.screen_width, self.state))

        # Reward: +1 for staying near the center, -1 otherwise
        reward = 1 if self.screen_width // 3 < self.state < 2 * self.screen_width // 3 else -1

        # End condition: if the object goes out of bounds
        self.done = self.state == 0 or self.state == self.screen_width

        return self.state, reward, self.done

    def reset(self):
        self.state = self.screen_width // 2
        self.done = False
        return self.state

# Agent
class QLearningAgent:
    def __init__(self, state_space, action_space, learning_rate=0.1, discount_factor=0.9, exploration_rate=1.0):
        self.q_table = np.zeros((state_space, action_space))
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.exploration_decay = 0.995
        self.min_exploration_rate = 0.01

    def choose_action(self, state):
        state = min(state, self.q_table.shape[0] - 1)  # 確保 state 在範圍內
        if random.uniform(0, 1) < self.exploration_rate:
            return random.choice([0, 1])  # Explore
        return np.argmax(self.q_table[state])  # Exploit

    def update(self, state, action, reward, next_state):
        state = min(state, self.q_table.shape[0] - 1)  # 確保 state 在範圍內
        next_state = min(next_state, self.q_table.shape[0] - 1)  # 確保 next_state 在範圍內
        max_q = np.max(self.q_table[next_state])
        self.q_table[state, action] += self.learning_rate * (
            reward + self.discount_factor * max_q - self.q_table[state, action]
        )

    def decay_exploration(self):
        self.exploration_rate = max(self.min_exploration_rate, self.exploration_rate * self.exploration_decay)

# Main Program
def main():
    pygame.init()
    screen_width = 400
    screen_height = 300
    screen = pygame.display.set_mode((screen_width, screen_height))
    pygame.display.set_caption("Q-Learning Balance Game")

    env = BalanceEnvironment(screen_width=screen_width)
    agent = QLearningAgent(state_space=screen_width, action_space=2)

    clock = pygame.time.Clock()
    episodes = 500
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0

        while not env.done:
            # Visualize
            screen.fill((0, 0, 0))
            pygame.draw.rect(screen, (255, 0, 0), (env.state, screen_height // 2, 10, 10))
            pygame.display.flip()

            # Agent chooses action
            action = agent.choose_action(state)
            next_state, reward, done = env.step(action)

            # Update Q-table
            agent.update(state, action, reward, next_state)
            state = next_state
            total_reward += reward

            # Control frame rate
            clock.tick(30)

        # Decay exploration rate
        agent.decay_exploration()

        print(f"Episode {episode + 1}: Total Reward = {total_reward}")

    pygame.quit()

if __name__ == "__main__":
    main()
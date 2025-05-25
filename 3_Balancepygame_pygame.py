"""
程式名稱: Q-Learning 平衡遊戲 (Balance Game with Q-Learning and Pygame)

用途與目的:
本程式利用 Q-Learning 強化學習演算法，讓智能體(agent)學習如何在一維空間中保持物體於畫面中央。遊戲畫面以 Pygame 顯示，紅色方塊代表物體，智能體可選擇向左或向右移動物體。目標是讓物體盡可能維持在畫面中央區域。

演算法說明:
- 環境 (BalanceEnvironment): 定義物體的位置(state)與移動規則，並根據物體是否位於中央區域給予獎勵(reward)。
- 智能體 (QLearningAgent): 使用 Q-Learning 演算法，根據 Q-table 決定每個狀態下的最佳動作。智能體會在探索(exploration)與利用(exploitation)間切換，並逐步降低探索率。
- 主程式: 執行多個回合(episodes)，每回合智能體與環境互動並更新 Q-table，並以 Pygame 視覺化過程。

Q-Learning 主要步驟:
1. 初始化 Q-table。
2. 在每個狀態下，根據探索率選擇動作(隨機或最大 Q-value)。
3. 執行動作，取得新狀態與獎勵。
4. 更新 Q-table。
5. 重複直到回合結束，並逐步降低探索率。
"""

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
            screen.fill((250, 250, 250))
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
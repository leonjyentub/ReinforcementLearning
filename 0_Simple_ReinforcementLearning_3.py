import numpy as np
import random

# 定義環境大小
rows, cols = 2, 3

# 定義折扣因子 gamma 與學習率 alpha
gamma = 0.9
alpha = 0.1

# 定義 reward
rewards = np.zeros((rows, cols))
rewards[1, 2] = 1

# 定義行動：上、下、左、右 (dy, dx)
actions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
num_actions = len(actions)

# 定義 Q-table，大小為 (rows, cols, actions)
Q = np.zeros((rows, cols, num_actions))

# 定義 ε-greedy 參數
epsilon = 0.1

# 定義 episodes 數
num_episodes = 5000

# 定義起點
start_state = (0, 0)

# Q-learning 主迴圈
for episode in range(num_episodes):
    state = start_state

    while state != (1, 2):
        i, j = state

        # ε-greedy 選擇行動
        if random.uniform(0, 1) < epsilon:
            action_index = random.randint(0, num_actions - 1)
        else:
            action_index = np.argmax(Q[i, j])

        # 執行行動
        di, dj = actions[action_index]
        ni, nj = i + di, j + dj

        # 邊界檢查，超出邊界則留在原地
        if 0 <= ni < rows and 0 <= nj < cols:
            next_state = (ni, nj)
        else:
            next_state = state

        # 取得 reward
        reward = rewards[next_state]

        # 更新 Q-value
        Q[i, j, action_index] += alpha * (
            reward + gamma * np.max(Q[next_state]) - Q[i, j, action_index]
        )

        # 移動到下一狀態
        state = next_state

# 計算最終 state-value function
V = np.max(Q, axis=2)

print("Final State Values (from Q-values):")
print(np.round(V, 3))

print("\nFinal Q-Table:")
for a in range(num_actions):
    print(f"Action {a}:")
    print(np.round(Q[:, :, a], 3))
import numpy as np
import random
import matplotlib.pyplot as plt

# 定義環境大小
rows, cols = 6, 6
gamma = 0.9
alpha = 0.1
rewards = np.zeros((rows, cols))

# 終點位置
goal_state = (5, 5)
rewards[goal_state] = 1

# 定義障礙物 (tuple 列表)
obstacles = [
    (2, 2), (3, 2), (4, 4)
]

# 動作設定
actions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
action_labels = ['↑', '↓', '←', '→']
num_actions = len(actions)

# 初始化 Q-table
Q = np.zeros((rows, cols, num_actions))
epsilon = 0.1
num_episodes = 5000
start_state = (0, 0)

# Q-learning 演算法
for episode in range(num_episodes):
    state = start_state
    while state != goal_state:
        i, j = state
        if random.uniform(0, 1) < epsilon:
            action_index = random.randint(0, num_actions - 1)
        else:
            action_index = np.argmax(Q[i, j])

        di, dj = actions[action_index]
        ni, nj = i + di, j + dj

        # 邊界檢查 + 障礙物檢查
        if (0 <= ni < rows and 0 <= nj < cols and (ni, nj) not in obstacles):
            next_state = (ni, nj)
        else:
            next_state = state

        reward = rewards[next_state]
        Q[i, j, action_index] += alpha * (
            reward + gamma * np.max(Q[next_state]) - Q[i, j, action_index]
        )
        state = next_state
    # 每 1000 次 episode 印出進度
    if episode % 1000 == 0:
        print(f"Episode {episode}/{num_episodes} completed.")


# 計算最終 state-value function 和 policy
V = np.max(Q, axis=2)
policy = np.argmax(Q, axis=2)

# 視覺化 V-value heatmap 和 policy arrows
fig, ax = plt.subplots(figsize=(7, 7))
im = ax.imshow(V, cmap='YlGnBu')

# 每格畫上 V-value
for i in range(rows):
    for j in range(cols):
        if (i, j) in obstacles:
            ax.add_patch(plt.Rectangle((j-0.5, i-0.5), 1, 1, color='black'))  # 障礙物
            continue
        if (i, j) == goal_state:
            ax.text(j, i, "G", ha="center", va="center", color="green", fontsize=14, fontweight='bold')
        else:
            ax.text(j, i, f"{V[i, j]:.2f}", ha="center", va="center", color="black", fontsize=10)

# 每格畫上最佳行動箭頭
for i in range(rows):
    for j in range(cols):
        if (i, j) in obstacles or (i, j) == goal_state:
            continue
        action_index = policy[i, j]
        label = action_labels[action_index]
        ax.text(j, i+0.25, label, ha="center", va="center", color="red", fontsize=12)

# 美化圖表
ax.set_xticks(np.arange(cols))
ax.set_yticks(np.arange(rows))
ax.set_xticklabels(np.arange(cols))
ax.set_yticklabels(np.arange(rows))
ax.set_title("6x6 Grid World: State Values & Policy with Obstacles")
plt.colorbar(im, ax=ax)
plt.gca().invert_yaxis()  # (0,0) 在左上
plt.grid(which='both', color='gray', linewidth=0.5)
plt.show()

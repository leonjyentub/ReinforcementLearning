import numpy as np

# 定義環境大小
rows, cols = 2, 3

# 定義折扣因子 gamma
gamma = 0.9

# 定義 reward，只有 (1,2) 終點是 1，其餘是 0
rewards = np.zeros((rows, cols))
rewards[1, 2] = 1

# 定義可能的行動：上、下、左、右 (dy, dx)
actions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

# 初始價值函數 V，全部設為 0
V = np.zeros((rows, cols))

# 終點狀態為固定值
V[1, 2] = 1

# 設定迭代參數
theta = 1e-4  # 收斂判斷門檻
delta = float('inf')

# 開始 value iteration
iteration = 0
while delta > theta:
    delta = 0
    iteration += 1
    V_copy = V.copy()

    for i in range(rows):
        for j in range(cols):
            if (i, j) == (1, 2):
                continue  # 終點狀態不更新

            # 計算四個可能行動後的 V 值
            values = []
            for action in actions:
                ni, nj = i + action[0], j + action[1]

                # 若行動超出邊界，則留在原地
                if 0 <= ni < rows and 0 <= nj < cols:
                    v_next = V_copy[ni, nj]
                else:
                    v_next = V_copy[i, j]

                # 套用 Bellman 更新 (此處 P=1)
                values.append(rewards[ni, nj] + gamma * v_next if 0 <= ni < rows and 0 <= nj < cols else gamma * v_next)

            # 更新 V 值 (取最大行動價值)
            V[i, j] = max(values)

            # 計算最大變化量
            delta = max(delta, abs(V[i, j] - V_copy[i, j]))

    print(f"Iteration {iteration}\n{V}\n")

print("Final State Values:")
print(V)
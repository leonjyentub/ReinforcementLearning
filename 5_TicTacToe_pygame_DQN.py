import pygame
import sys
import numpy as np
import random
from collections import defaultdict, deque
import time
import torch
import torch.nn as nn
import torch.optim as optim

# Pygame 設定
WIDTH, HEIGHT = 600, 600
LINE_WIDTH = 15
BOARD_ROWS = 3
BOARD_COLS = 3
SQUARE_SIZE = WIDTH // BOARD_COLS
CIRCLE_RADIUS = SQUARE_SIZE // 3
CIRCLE_WIDTH = 15
CROSS_WIDTH = 25
SPACE = SQUARE_SIZE // 4
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)

pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption('Tic Tac Toe RL')
# 載入mac的中文字型
#font = pygame.font.Font(None, 36)
font = pygame.font.Font("/System/Library/Fonts/STHeiti Light.ttc", 36)

def draw_board(board):
    screen.fill(WHITE)
    for i in range(1, BOARD_ROWS):
        pygame.draw.line(screen, BLACK, (0, i * SQUARE_SIZE), (WIDTH, i * SQUARE_SIZE), LINE_WIDTH)
        pygame.draw.line(screen, BLACK, (i * SQUARE_SIZE, 0), (i * SQUARE_SIZE, HEIGHT), LINE_WIDTH)
    for row in range(BOARD_ROWS):
        for col in range(BOARD_COLS):
            if board[row][col] == 1:
                pygame.draw.circle(screen, BLACK, (int(col * SQUARE_SIZE + SQUARE_SIZE // 2),
                                                   int(row * SQUARE_SIZE + SQUARE_SIZE // 2)),
                                   CIRCLE_RADIUS, CIRCLE_WIDTH)
            elif board[row][col] == 2:
                start_desc = (col * SQUARE_SIZE + SPACE, row * SQUARE_SIZE + SPACE)
                end_desc = (col * SQUARE_SIZE + SQUARE_SIZE - SPACE, row * SQUARE_SIZE + SQUARE_SIZE - SPACE)
                pygame.draw.line(screen, RED, start_desc, end_desc, CROSS_WIDTH)
                pygame.draw.line(screen, RED,
                                 (col * SQUARE_SIZE + SPACE, row * SQUARE_SIZE + SQUARE_SIZE - SPACE),
                                 (col * SQUARE_SIZE + SQUARE_SIZE - SPACE, row * SQUARE_SIZE + SPACE), CROSS_WIDTH)
    pygame.display.update()

def check_winner(board):
    for i in range(3):
        if board[i, 0] == board[i, 1] == board[i, 2] != 0:
            return board[i, 0]
        if board[0, i] == board[1, i] == board[2, i] != 0:
            return board[0, i]
    if board[0, 0] == board[1, 1] == board[2, 2] != 0:
        return board[0, 0]
    if board[0, 2] == board[1, 1] == board[2, 0] != 0:
        return board[0, 2]
    if np.all(board != 0):
        return 0  # 平手
    return -1  # 未結束

def board_to_tuple(board):
    return tuple(board.reshape(9))

def available_moves(board):
    return [(i, j) for i in range(3) for j in range(3) if board[i, j] == 0]

# Q-learning Agent
class QLearningAgent:
    def __init__(self, player):
        self.Q = defaultdict(float)
        self.player = player

    def choose_action(self, board, available_moves, epsilon=0.1):
        state = board_to_tuple(board)
        if random.uniform(0, 1) < epsilon:
            return random.choice(available_moves)
        qs = [self.Q[(state, move)] for move in available_moves]
        max_q = max(qs)
        best_moves = [move for move, q in zip(available_moves, qs) if q == max_q]
        return random.choice(best_moves)

    def update(self, board, action, reward, next_board, next_available_moves, done):
        ALPHA = 0.7
        GAMMA = 0.9
        state = board_to_tuple(board)
        next_state = board_to_tuple(next_board)
        predict = self.Q[(state, action)]
        if done:
            target = reward
        else:
            next_qs = [self.Q[(next_state, move)] for move in next_available_moves] if next_available_moves else [0]
            target = reward + GAMMA * max(next_qs)
        self.Q[(state, action)] += ALPHA * (target - predict)

# DQN Agent
class DQNNet(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQNNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class DQNAgent:
    def __init__(self, player):
        self.player = player
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = DQNNet(9, 9).to(self.device)
        self.target_model = DQNNet(9, 9).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.memory = deque(maxlen=2000)
        self.gamma = 0.9
        self.batch_size = 64
        self.update_target_steps = 100
        self.steps = 0

    def choose_action(self, board, available_moves, epsilon=0.1):
        if random.random() < epsilon:
            return random.choice(available_moves)
        state = self._board_to_tensor(board)
        with torch.no_grad():
            q_values = self.model(state).cpu().numpy().flatten()
        # 選擇可用動作中 Q 值最大的
        best_move = None
        best_q = -float('inf')
        for move in available_moves:
            idx = move[0] * 3 + move[1]
            if q_values[idx] > best_q:
                best_q = q_values[idx]
                best_move = move
        return best_move

    def store(self, state, action, reward, next_state, done, available_moves):
        self.memory.append((state, action, reward, next_state, done, available_moves))

    def train(self):
        if len(self.memory) < self.batch_size:
            return
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones, available_moves_batch = zip(*batch)
        states = torch.stack([self._board_to_tensor(s) for s in states]).to(self.device)
        next_states = torch.stack([self._board_to_tensor(s) for s in next_states]).to(self.device)
        actions_idx = [a[0]*3+a[1] for a in actions]
        actions_idx = torch.tensor(actions_idx, dtype=torch.long).unsqueeze(1).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(1).to(self.device)
    
        q_values = self.model(states).gather(1, actions_idx)
        
        with torch.no_grad():
            next_q_values = self.target_model(next_states)
            # 只考慮下個狀態可用動作的最大 Q 值
            max_next_q = []
            for i, moves in enumerate(available_moves_batch):
                if not moves:
                    max_next_q.append(0)
                else:
                    idxs = [m[0]*3+m[1] for m in moves]
                    max_next_q.append(next_q_values[i][idxs].max().item())
            max_next_q = torch.tensor(max_next_q, dtype=torch.float32).unsqueeze(1).to(self.device)
            target = rewards + self.gamma * max_next_q * (1 - dones)
        loss = nn.functional.mse_loss(q_values, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.steps += 1
        if self.steps % self.update_target_steps == 0:
            self.target_model.load_state_dict(self.model.state_dict())

    def _board_to_tensor(self, board):
        # Q-learning: 0=空, 1=Q, 2=DQN
        arr = np.where(board == self.player, 1, np.where(board == 0, 0, -1)).astype(np.float32)
        #return torch.tensor(arr.flatten(), dtype=torch.float32, device=self.device).unsqueeze(0)
        return torch.tensor(arr.flatten(), dtype=torch.float32, device=self.device)

def play_game(agent1, agent2, train=True, show=False, delay=0.5):
    board = np.zeros((3, 3), dtype=int)
    agents = [agent1, agent2]
    player = 0
    states_actions = []
    while True:
        moves = available_moves(board)
        action = agents[player].choose_action(board, moves, epsilon=0.1 if train else 0)
        board[action] = agents[player].player
        if show:
            draw_board(board)
            time.sleep(delay)
        winner = check_winner(board)
        states_actions.append((board.copy(), action))
        if winner != -1:
            # 結果處理
            if train:
                for idx, (b, a) in enumerate(states_actions):
                    if winner == 0:
                        reward = 0.5
                    elif winner == agents[idx % 2].player:
                        reward = 1
                    else:
                        reward = -1
                    next_b = states_actions[idx + 1][0] if idx + 1 < len(states_actions) else b
                    next_moves = available_moves(next_b)
                    if isinstance(agents[idx % 2], QLearningAgent):
                        agents[idx % 2].update(b, a, reward, next_b, next_moves, True)
                    else:
                        agents[idx % 2].store(b, a, reward, next_b, True, next_moves)
                        agents[idx % 2].train()
            return winner
        # 更新 Q 值
        if train and len(states_actions) >= 2:
            b, a = states_actions[-2]
            next_b = board.copy()
            next_moves = available_moves(next_b)
            if isinstance(agents[(player + 1) % 2], QLearningAgent):
                agents[(player + 1) % 2].update(b, a, 0, next_b, next_moves, False)
            else:
                agents[(player + 1) % 2].store(b, a, 0, next_b, False, next_moves)
                agents[(player + 1) % 2].train()
        player = 1 - player

def show_winner(winner):
    LIGHT_GRAY = (200, 200, 200)
    if winner == 1:
        text = font.render("Q-learning Agent 獲勝", True, LIGHT_GRAY, BLACK)
    elif winner == 2:
        text = font.render("DQN Agent 獲勝", True, LIGHT_GRAY, BLACK)
    else:
        text = font.render("平手", True, LIGHT_GRAY, BLACK)
    screen.blit(text, (WIDTH // 2 - text.get_width() // 2, HEIGHT // 2 - text.get_height() // 2))
    pygame.display.update()
    time.sleep(2)
    screen.fill(WHITE)
    pygame.display.update()

def main():
    agent1 = QLearningAgent(1)
    agent2 = DQNAgent(2)
    print("訓練中...")
    for i in range(3000):
        play_game(agent1, agent2, train=True, show=False)
    print("訓練完成，AI 對戰展示：")
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
        winner = play_game(agent1, agent2, train=False, show=True, delay=0.5)
        show_winner(winner)

if __name__ == "__main__":
    main()

import pygame
import sys
import numpy as np
import random
from collections import defaultdict
import time

# Q-learning 參數
ALPHA = 0.7
GAMMA = 0.9
EPSILON = 0.1

# Pygame 參數
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
font = pygame.font.SysFont(None, 72)

def draw_board(board):
    screen.fill(WHITE)
    # 畫線
    for i in range(1, BOARD_ROWS):
        pygame.draw.line(screen, BLACK, (0, i * SQUARE_SIZE), (WIDTH, i * SQUARE_SIZE), LINE_WIDTH)
        pygame.draw.line(screen, BLACK, (i * SQUARE_SIZE, 0), (i * SQUARE_SIZE, HEIGHT), LINE_WIDTH)
    # 畫棋子
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
    # 檢查橫、直、斜
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

class QLearningAgent:
    def __init__(self, player):
        self.Q = defaultdict(float)
        self.player = player

    def choose_action(self, board, available_moves, epsilon=EPSILON):
        state = board_to_tuple(board)
        if random.uniform(0, 1) < epsilon:
            return random.choice(available_moves)
        qs = [self.Q[(state, move)] for move in available_moves]
        max_q = max(qs)
        best_moves = [move for move, q in zip(available_moves, qs) if q == max_q]
        return random.choice(best_moves)

    def update(self, board, action, reward, next_board, next_available_moves, done):
        state = board_to_tuple(board)
        next_state = board_to_tuple(next_board)
        predict = self.Q[(state, action)]
        if done:
            target = reward
        else:
            next_qs = [self.Q[(next_state, move)] for move in next_available_moves] if next_available_moves else [0]
            target = reward + GAMMA * max(next_qs)
        self.Q[(state, action)] += ALPHA * (target - predict)

def available_moves(board):
    return [(i, j) for i in range(3) for j in range(3) if board[i, j] == 0]

def play_game(agent1, agent2, train=True, show=False, delay=0.5):
    board = np.zeros((3, 3), dtype=int)
    agents = [agent1, agent2]
    player = 0
    states_actions = []
    while True:
        moves = available_moves(board)
        action = agents[player].choose_action(board, moves) if train else agents[player].choose_action(board, moves, epsilon=0)
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
                        reward = 0.5  # 平手
                    elif winner == agents[idx % 2].player:
                        reward = 1
                    else:
                        reward = -1
                    next_b = states_actions[idx + 1][0] if idx + 1 < len(states_actions) else b
                    next_moves = available_moves(next_b)
                    agents[idx % 2].update(b, a, reward, next_b, next_moves, True)
            return winner
        # 更新 Q 值
        if train and len(states_actions) >= 2:
            b, a = states_actions[-2]
            next_b = board.copy()
            next_moves = available_moves(next_b)
            agents[(player + 1) % 2].update(b, a, 0, next_b, next_moves, False)
        player = 1 - player

def main():
    agent1 = QLearningAgent(1)
    agent2 = QLearningAgent(2)
    # 訓練
    print("訓練中...")
    for i in range(10000):
        play_game(agent1, agent2, train=True, show=False)
    print("訓練完成，AI 對戰展示：")
    # 展示對戰
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
        winner = play_game(agent1, agent2, train=False, show=True, delay=0.5)
        if winner == 0:
            text = font.render("平手!", True, RED)
        else:
            text = font.render(f"玩家 {winner} 勝!", True, RED)
        screen.blit(text, (WIDTH // 4, HEIGHT // 2 - 50))
        pygame.display.update()
        time.sleep(2)
        input("按 Enter 鍵繼續...")
        # 重置畫面
        screen.fill(WHITE)
        pygame.display.update()
        

if __name__ == "__main__":
    main()

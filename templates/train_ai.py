import numpy as np
import json
import os

# Global variables
board = np.zeros((3, 3), dtype=int)
q_table = {}
alpha = 0.1  # Learning rate
gamma = 0.9  # Discount factor
epsilon = 0.2  # Exploration rate

q_table_file = 'q_table.json'

# Load Q-table if it exists
if os.path.exists(q_table_file):
    with open(q_table_file, 'r') as f:
        q_table = json.load(f)
        q_table = {k: np.array(v) for k, v in q_table.items()}
else:
    q_table = {}

def reset_board():
    global board
    board = np.zeros((3, 3), dtype=int)

def check_win(b):
    for i in range(3):
        if np.all(b[i, :] == 1) or np.all(b[:, i] == 1):
            return 1  # Player wins
        if np.all(b[i, :] == -1) or np.all(b[:, i] == -1):
            return -1  # AI wins
    if b[0, 0] == b[1, 1] == b[2, 2] != 0 or b[0, 2] == b[1, 1] == b[2, 0] != 0:
        return b[1, 1]  # Diagonal win
    if not 0 in b:
        return 0  # Draw
    return None  # Game continues

def board_to_state(board):
    return ''.join(map(str, board.flatten()))

def update_q_table(state, action, reward, next_state):
    if state not in q_table:
        q_table[state] = np.zeros(9)
    if next_state not in q_table:
        q_table[next_state] = np.zeros(9)

    q_table[state][action] = q_table[state][action] + alpha * (
        reward + gamma * np.max(q_table[next_state]) - q_table[state][action]
    )

def is_winning_move(board, player):
    for i in range(3):
        if np.sum(board[i, :] == player) == 2 and np.sum(board[i, :] == 0) == 1:
            return (i, np.where(board[i, :] == 0)[0][0])
        if np.sum(board[:, i] == player) == 2 and np.sum(board[:, i] == 0) == 1:
            return (np.where(board[:, i] == 0)[0][0], i)
    if np.sum(np.diag(board) == player) == 2 and np.sum(np.diag(board) == 0) == 1:
        return (np.where(np.diag(board) == 0)[0][0], np.where(np.diag(board) == 0)[0][0])
    if np.sum(np.diag(np.fliplr(board)) == player) == 2 and np.sum(np.diag(np.fliplr(board)) == 0) == 1:
        return (np.where(np.diag(np.fliplr(board)) == 0)[0][0], 2 - np.where(np.diag(np.fliplr(board)) == 0)[0][0])
    return None

def ai_move():
    global board
    state = board_to_state(board)

    # Check if AI can win
    win_move = is_winning_move(board, -1)
    if win_move:
        board[win_move[0], win_move[1]] = -1
        return win_move

    # Check if player can win and block it
    block_move = is_winning_move(board, 1)
    if block_move:
        board[block_move[0], block_move[1]] = -1
        return block_move

    # If no immediate winning or blocking move, proceed with Q-learning
    if state not in q_table:
        q_table[state] = np.zeros(9)

    valid_actions = np.where(board.flatten() == 0)[0]
    action = np.random.choice(valid_actions) if np.random.rand() < epsilon else np.argmax(q_table[state][valid_actions])
    row, col = divmod(action, 3)

    board[row, col] = -1
    return row, col

def play_game():
    reset_board()
    while True:
        state = board_to_state(board)
        move = ai_move()
        new_state = board_to_state(board)
        
        result = check_win(board)
        if result is not None:
            if result == -1:  # AI wins
                update_q_table(state, move[0] * 3 + move[1], 1, new_state)
            elif result == 1:  # Player wins (simulated random moves for player)
                update_q_table(state, move[0] * 3 + move[1], -1, new_state)
            else:  # Draw
                update_q_table(state, move[0] * 3 + move[1], 0.5, new_state)
            return result  # Return the game result

        # Simulate random player move
        valid_actions = np.where(board.flatten() == 0)[0]
        if len(valid_actions) == 0:
            break
        action = np.random.choice(valid_actions)
        row, col = divmod(action, 3)
        board[row, col] = 1

        result = check_win(board)
        if result is not None:
            if result == 1:  # Player wins
                update_q_table(state, move[0] * 3 + move[1], -1, new_state)
            elif result == 0:  # Draw
                update_q_table(state, move[0] * 3 + move[1], 0.5, new_state)
            return result  # Return the game result

    return 0  # Return draw if no winner

def train_ai(num_games=10000):
    ai_wins = 0
    player_wins = 0
    draws = 0

    for _ in range(num_games):
        result = play_game()
        if result == -1:
            ai_wins += 1
        elif result == 1:
            player_wins += 1
        else:
            draws += 1

    # Save Q-table to disk
    with open(q_table_file, 'w') as f:
        json.dump({k: v.tolist() for k, v in q_table.items()}, f)

    # Print the training results
    print(f"Training complete after {num_games} games.")
    print(f"AI Wins: {ai_wins}")
    print(f"Player Wins: {player_wins}")
    print(f"Draws: {draws}")

if __name__ == "__main__":
    train_ai(100000)

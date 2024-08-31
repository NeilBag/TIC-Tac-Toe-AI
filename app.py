import numpy as np
from flask import Flask, render_template, jsonify, request

app = Flask(__name__)

# Global variables
board = np.zeros((3, 3), dtype=int)

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

def evaluate(board):
    winner = check_win(board)
    if winner == -1:
        return 1  # AI wins
    elif winner == 1:
        return -1  # Player wins
    else:
        return 0  # Draw or ongoing game

def minimax(board, depth, is_maximizing):
    score = evaluate(board)
    if score != 0 or depth == 0:
        return score
    if not np.any(board == 0):  # Draw
        return 0

    if is_maximizing:
        best_score = -float('inf')
        for row in range(3):
            for col in range(3):
                if board[row, col] == 0:
                    board[row, col] = -1  # AI move
                    score = minimax(board, depth - 1, False)
                    board[row, col] = 0
                    best_score = max(score, best_score)
        return best_score
    else:
        best_score = float('inf')
        for row in range(3):
            for col in range(3):
                if board[row, col] == 0:
                    board[row, col] = 1  # Player move
                    score = minimax(board, depth - 1, True)
                    board[row, col] = 0
                    best_score = min(score, best_score)
        return best_score

def ai_move():
    global board
    best_score = -float('inf')
    best_move = None

    for row in range(3):
        for col in range(3):
            if board[row, col] == 0:
                board[row, col] = -1  # AI move
                score = minimax(board, 5, False)  # Depth of 5 to limit search
                board[row, col] = 0
                if score > best_score:
                    best_score = score
                    best_move = (row, col)

    if best_move:
        board[best_move[0], best_move[1]] = -1
        return best_move
    return None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/move', methods=['POST'])
def move():
    global board
    data = request.get_json()
    
    row, col = int(data['row']), int(data['col'])

    if board[row, col] == 0:
        board[row, col] = 1  # Player move

        print("Board after player's move:")
        print(board)

        result = check_win(board)
        if result is not None:
            reset_board()
            return jsonify({"result": int(result)})

        best_move = ai_move()

        print(f"AI moved to ({best_move[0]}, {best_move[1]})")

        result = check_win(board)
        if result is not None:
            reset_board()
            return jsonify({"ai_row": int(best_move[0]), "ai_col": int(best_move[1]), "result": int(result)})

        return jsonify({"ai_row": int(best_move[0]), "ai_col": int(best_move[1]), "result": None})

    return jsonify({"result": "Invalid move"})

if __name__ == '__main__':
    app.run(debug=True)

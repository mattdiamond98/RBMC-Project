import numpy as np

piece_map = {
    'P' : 1,
    'R' : 2,
    'N' : 3,
    'B' : 4,
    'Q' : 5,
    'K' : 6,
    'p' : 7,
    'r' : 8,
    'n' : 9,
    'b' : 10,
    'q' : 11,
    'k' : 12,
}

def fen_to_board(board):
    fen = board.fen()
    board = np.zeros((8,8))

    x = 0
    y = 0

    for c in fen:
        if c == ' ':
            break
        
        if c.isdigit():
            x += int(c)
        elif c == '/':
            y += 1
            x = 0
        else:
            board[y][x] = piece_map[c]
            x += 1

    return board

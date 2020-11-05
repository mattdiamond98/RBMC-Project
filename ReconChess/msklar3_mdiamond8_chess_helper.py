import numpy as np
import chess

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

def piece_equal(piece1, piece2):
  if piece1 is None:
    return piece2 is None
  if piece2 is None:
    return False
  return piece1.symbol() == piece2.symbol()

def empty_path_squares(move):
  """
  Returns the known empty squares resulting from a successful move, used in filtering.
  
  :param move: chess.Move -- the move taken, or None
  :return: list(chess.Square) -- the known empty squares based on this move
  """
  if move is None:
    return []
  if chess.square_distance(move.from_square, move.to_square) <= 1:
    return []
  x = chess.square_rank(move.to_square) - chess.square_rank(move.from_square)
  y = chess.square_file(move.to_square) - chess.square_file(move.from_square)
  if x != 0 and y != 0 and abs(x) != abs(y): # knight move check
    return [] 
  empty_squares = chess.between(move.from_square, move.to_square)
  empty_squares.append(move.from_square)
  return empty_squares
  
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
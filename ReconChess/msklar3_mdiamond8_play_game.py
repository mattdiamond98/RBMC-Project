import chess
from game import Game
from datetime import datetime

def play_local_game(white_player, black_player, player_names):
    players = [black_player, white_player]

    game = Game()

    # writing to files
    time = "{}".format(datetime.today()).replace(" ", "_").replace(":", "-").replace(".", "-")
    filename_game = "GameHistory/" + time + "game_boards.txt"
    filename_true = "GameHistory/" + time + "true_boards.txt"
    output = open(filename_game, "w")
    output_true = open(filename_true, "w")
    output.write("Starting Game between {}-WHITE and {}-BLACK\n".format(player_names[0], player_names[1]))
    output_true.write("Starting Game between {}-WHITE and {}-BLACK\n".format(player_names[0], player_names[1]))

    white_player.handle_game_start(chess.WHITE, chess.Board())
    black_player.handle_game_start(chess.BLACK, chess.Board())
    game.start()

    move_number = 1
    while not game.is_over():
        requested_move, taken_move = play_turn(game, players[game.turn], game.turn, move_number, output, output_true)
        move_number += 1

    winner_color, winner_reason = game.get_winner()

    white_game_state = white_player.handle_game_end(winner_color, winner_reason)
    black_game_state = black_player.handle_game_end(winner_color, winner_reason)

    output.write("Game Over!\n")
    if winner_color is not None:
        output.write(winner_reason)
    else:
        output.write('Draw!')
    return winner_color, winner_reason, white_game_state, black_game_state

def play_turn(game, player, turn, move_number, output, output_true):
    possible_moves = game.get_moves()
    possible_sense = list(chess.SQUARES)

    # notify the player of the previous opponent's move
    captured_square = game.opponent_move_result()
    player.handle_opponent_move_result(captured_square is not None, captured_square)

    # play sense action
    sense = player.choose_sense(possible_sense, possible_moves, game.get_seconds_left())
    sense_result = game.handle_sense(sense)
    player.handle_sense_result(sense_result)

    # play move action
    move = player.choose_move(possible_moves, game.get_seconds_left())
    requested_move, taken_move, captured_square, reason = game.handle_move(move)
    player.handle_move_result(requested_move, taken_move, reason, captured_square is not None,
                              captured_square)

    game.end_turn()
    return requested_move, taken_move

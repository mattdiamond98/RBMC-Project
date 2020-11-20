#!/usr/bin/env python3

"""
File Name:      my_agent.py
Authors:        Matthew Sklar and Matan Diamond
Date:           10/31/2020

Description:    Python file for my agent.
Source:         Adapted from recon-chess (https://pypi.org/project/reconchess/)
"""

import copy
import os.path as path
import random
import time
from datetime import datetime

import chess
import chess.engine
import numpy as np
import torch

import msklar3_mdiamond8_chess_helper as helper
import msklar3_mdiamond8_config as config
import msklar3_mdiamond8_mcts as mcts
import msklar3_mdiamond8_memory as memory
import msklar3_mdiamond8_nn as nn
import msklar3_mdiamond8_teacher as teacher
from msklar3_mdiamond8_chess_helper import action_map, fen_to_board, gen_state
from msklar3_mdiamond8_particle_filter import ParticleFilter
from player import Player

MOVE_OPTIONS = 64 * 64
IN_CHANNELS = 8

'''
Theoretical Opening Theory: Theory of the Opening 

Hungarian Opening: Anti-Knight Variation
Summary: Strong approaches to consistently defeat the dreaded knight rush (dun dun dun)
'''
# g2f3 only occurs if knight captured there before
WHITE_OPENING = ['g1f3', 'g2g3', 'f1g2', 'g2f3', 'e1g1']

'''
Theoretical Opening Theory: Theory of the Opening

Bon'Cloud Opening: Anti-Knight Variation
Summary: Strong approaches to consistently defeat the dreaded knight rush (dun dun dun)
'''
BLACK_OPENING = ['e7e6', 'e8e7', 'g8f6']

class MagnusDLuffy(Player):

    def __init__(self):
        try:
            self.network = torch.load('network.torch')
        except:
            print('failed to find network.torch')
            self.network = nn.Net(IN_CHANNELS, MOVE_OPTIONS)
        self.mcts = None
        self.game_history = memory.GameMemory()

        # self.stockfish = chess.engine.SimpleEngine.popen_uci(path.abspath('msklar3_mdiamond8_stockfish'))
        
    def handle_game_start(self, color, board):
        """
        This function is called at the start of the game.

        :param color: chess.BLACK or chess.WHITE -- your color assignment for the game
        :param board: chess.Board -- initial board state
        :return:
        """
        self.color = color
        self.state = ParticleFilter(board, color)
        self.legal_move_made = False    # train network to make legal moves
        self.opening_turn = 0
        self.invalid_pos = False
     
    def handle_opponent_move_result(self, captured_piece, captured_square):
        """
        This function is called at the start of your turn and gives you the chance to update your board.

        :param captured_piece: bool - true if your opponents captured your piece with their last move
        :param captured_square: chess.Square - position where your piece was captured
        """
        if self.opening_turn > 0:
            self.state.update_opponent_move_result(captured_piece, captured_square, self.network)

    def choose_sense(self, possible_sense, possible_moves, seconds_left):
        """
        This function is called to choose a square to perform a sense on.

        :param possible_sense: List(chess.SQUARES) -- list of squares to sense around
        :param possible_moves: List(chess.Moves) -- list of acceptable moves based on current board
        :param seconds_left: float -- seconds left in the game

        :return: chess.SQUARE -- the center of 3x3 section of the board you want to sense
        :example: choice = chess.A1
        """
        if self.color == chess.WHITE:
            if self.opening_turn == 3:  # check if knight will capture king in white opening after castling
                return chess.Square(chess.E4)


        sample = self.state.sample_from_particles(50)
        
        if not sample:
          return random.choice(possible_sense)
        
        most_uncertain = None
        most_uncertain_score = None
        for square in possible_sense:
          rank, file = chess.square_rank(square), chess.square_file(square)
          if rank - 1 < 0 or rank + 1 > 7 or file - 1 < 0 or file + 1 > 7:
            continue
          distribution = [dict() for _ in range(9)] # a dict for each square, containing sum weights
          for (board, weight) in sample:
            for delta_rank in [-1, 0, 1]:
              for delta_file in [-1, 0, 1]:
                 sense_square = chess.square(file + delta_file, rank + delta_rank)
                 piece = board.piece_at(sense_square)

                 i = 3 * (delta_rank + 1) + (delta_file + 1)
                 if piece in distribution[i]:
                   d = distribution[i]
                   d[piece] += weight
                   distribution[i] = d
                 else:
                   d = distribution[i]
                   d[piece] = weight
                   distribution[i] = d

          total_score = 0
          for weights in distribution:
            total_score += max(weights.values()) / sum(weights.values())
          
          if most_uncertain is None or total_score < most_uncertain_score:
            most_uncertain = square
            most_uncertain_score = total_score
          
        return most_uncertain
        
    def handle_sense_result(self, sense_result):
        """
        This is a function called after your picked your 3x3 square to sense and gives you the chance to update your
        board.

        :param sense_result: A list of tuples, where each tuple contains a :class:`Square` in the sense, and if there
                             was a piece on the square, then the corresponding :class:`chess.Piece`, otherwise `None`.
        :example:
        [
            (A8, Piece(ROOK, BLACK)), (B8, Piece(KNIGHT, BLACK)), (C8, Piece(BISHOP, BLACK)),
            (A7, Piece(PAWN, BLACK)), (B7, Piece(PAWN, BLACK)), (C7, Piece(PAWN, BLACK)),
            (A6, None), (B6, None), (C8, None)
        ]
        """
        if self.color == chess.WHITE:   # on turn 3 of white turn handle knight rush capture
            for square in sense_result:
                if self.opening_turn == 3:
                    if square[0] == chess.F3 and square[1] != chess.Piece(chess.KNIGHT, chess.BLACK):
                        self.opening_turn += 1

        self.state.update_sense_result(sense_result)

    def choose_move(self, possible_moves, seconds_left):
        """
        Choose a move to enact from a list of possible moves.

        :param possible_moves: List(chess.Moves) -- list of acceptable moves based only on pieces
        :param seconds_left: float -- seconds left to make a move
        
        :return: chess.Move -- object that includes the square you're moving from to the square you're moving to
        :example: choice = chess.Move(chess.F2, chess.F4)
        
        :condition: If you intend to move a pawn for promotion other than Queen, please specify the promotion parameter
        :example: choice = chess.Move(chess.G7, chess.G8, promotion=chess.KNIGHT) *default is Queen
        """
        # # NOTE: for training, we randomly sample but for tournament we should select the most likely always
        self.stockfish = chess.engine.SimpleEngine.popen_uci(path.abspath('msklar3_mdiamond8_stockfish'))

        opening_move = self.opening()

        if opening_move != None:
            self.opening_turn += 1
            return opening_move

        particle_sample = self.state.sample_from_particles()
        if not particle_sample:
          return random.choice(possible_moves)
        sample, weight = particle_sample[0] # sample a single state from the particles

        state, moves = gen_state(sample, self.state.color)

        action = self.pick_action(state, sample, moves)

        self.game_history.add_turn(memory.TurnMemory(
            (state, moves), # state and pseudo-legal moves
            torch.tensor(action[3])
        ))

        choice = action_map(action[0])
        if choice in possible_moves:
            self.legal_move_made = True

        self.stockfish.quit()

        return choice
        
    def handle_move_result(self, requested_move, taken_move, reason, captured_piece, captured_square):
        """
        This is a function called at the end of your turn/after your move was made and gives you the chance to update
        your board.

        :param requested_move: chess.Move -- the move you intended to make
        :param taken_move: chess.Move -- the move that was actually made
        :param reason: String -- description of the result from trying to make requested_move
        :param captured_piece: bool - true if you captured your opponents piece
        :param captured_square: chess.Square - position where you captured the piece
        """
        self.state.update_handle_move_result(taken_move, captured_piece, captured_square)
        
    def handle_game_end(self, winner_color, win_reason):  # possible GameHistory object...
        """
        This function is called at the end of the game to declare a winner.

        :param winner_color: Chess.BLACK/chess.WHITE -- the winning color
        :param win_reason: String -- the reason for the game ending
        """
        v = [1] if self.color == winner_color else [-1]

        # self.stockfish.quit()

        self.game_history.v = torch.tensor(v, dtype=torch.float32)
        print('loss game', v)
        torch.save(self.network, 'network.torch')

        print("I'm gonna be king of the chess players!")
        print('network grad', self.network.policy_linear.weight.grad)
        
        return self.game_history

    '''
    Pick an action and get data for memory.

    Returns a tuple containing:
        0 -> the id of the selected action
        1 -> the value of the state from the neural network
        2 -> the probability distribution from the neural network
        3 -> the probability distribution from the MCTS
    '''
    def pick_action(self, state, sample, possible_moves):
        # Get value of the state from the neural network and probability distsribution from the neural network
        nn_policy, nn_value = self.network.forward(state, possible_moves)

        # Create MCT
        print('mct sample\n', sample, sample.is_valid())
        root = mcts.Node((state, possible_moves), sample, self.state.color)
        self.mcts = mcts.MCTS(root, self.state.color)

        # Train the MCT
        for _ in range(config.MCTS_SIMULATIONS):
            self.simulate(state, possible_moves)

        # Choose the optimal action given the MCT
        action, pi = self.select_move(config.TAU)
        # self.mcts.to_string()
        return action, nn_value, nn_policy, pi

    def simulate(self, state, possible_moves):
        # Selection
        leaf, path = self.mcts.select()
        sample = leaf.sample

        # Evaluation
        pi, v = self.evaluate_leaf(leaf)

        if sample.is_valid():            
            analysis = self.stockfish.analyse(sample, chess.engine.Limit(time=.1))
            score = analysis['score'].relative
            turn = analysis['score'].turn

            # Leaf node of game
            if score.is_mate() and score.mate() == 0:
                # Backup
                stockfish_v = 1 if turn == leaf.color else 0
                v = config.NN_DECISION_WEIGHT_ALPHA * v + (1 - config.NN_DECISION_WEIGHT_ALPHA) * (stockfish_v)

                self.mcts.backfill(v, path)
                return

            if 'pv' in analysis and len(analysis['pv']) != 0:
                if not (score.is_mate() and score.mate() == 0):
                    best_move = analysis['pv'][0]
                    # print('best move is ', best_move, 'with score', score)
                    stockfish_actions = helper.best_move_to_action_map(best_move)

                pi = config.NN_DECISION_WEIGHT_ALPHA * pi.detach().numpy() + (1 - config.NN_DECISION_WEIGHT_ALPHA) * stockfish_actions
                pi = torch.tensor(pi)

                if score.is_mate():
                    stockfish_v = score.mate() / abs(score.mate())
                else:
                    stockfish_v = helper.cp_to_win_probability(score.score())

                v = config.NN_DECISION_WEIGHT_ALPHA * v + (1 - config.NN_DECISION_WEIGHT_ALPHA) * stockfish_v

        # Expansion
        best_policies = torch.topk(pi, config.SIMULATION_EXPANSION)[1]

        for action_id in best_policies:
            action_move = helper.action_map(action_id)
            
            if action_move not in sample.pseudo_legal_moves:
                continue

            new_sample = copy.deepcopy(sample)
            new_sample.push(action_move)
            new_state, _ = gen_state(new_sample, not self.mcts.leaf.color)

            self.mcts.leaf.edges.append(mcts.Edge(
                self.mcts.leaf,
                mcts.Node((new_state, possible_moves), new_sample, not self.mcts.leaf.color),
                action_id,
                pi[action_id]))

        # Backup
        self.mcts.backfill(v, path)

    def evaluate_leaf(self, leaf):
        pi, v = self.network.forward(leaf.state[0], leaf.state[1])

        return pi, v

    '''
    Select a move to use and return the action to make the move and probability
    distribution of the policies.
    '''
    def select_move(self, tau):
        pi, values = self.policy(tau)

        if tau == 0:    # Deterministic
            action = random.choice(np.anywhere(pi == max(pi)))
        else:           # Stochastically
            # self.mcts.to_string()
            print('pi is', pi)
            action_id = np.random.multinomial(1, pi)
            action = np.where(action_id == 1)[0][0]

        value = values[action]

        print('selected move from action:', action, 'with value:', value)

        return action, pi

    # Generate pi and get values to pass through
    def policy(self, tau):
        print(self.mcts.root.sample)
        edges = self.mcts.root.edges
        pi = np.zeros(MOVE_OPTIONS, dtype=np.double)
        values = np.zeros(MOVE_OPTIONS, dtype=np.float32)
        if len(edges) == 0:
            print("there are no edges here")
        for edge in edges:
            print('edge is ', edge)
            edge.to_string()
            values[edge.action] = edge.data['Q']

            if tau == 0:
                pi[edge.action] = edge.data['N']
            else:
                pi[edge.action] = pow(edge.data['N'], 1 / tau)

        pi = pi / np.sum(pi)

        return pi, values

    def opening(self):
        if self.opening_turn > 0:
            return None
        if self.color == chess.WHITE:
            if self.opening_turn > 4:
                return None

            return chess.Move.from_uci(WHITE_OPENING[self.opening_turn])

        if self.color == chess.BLACK:
            if self.opening_turn > 3:
                return None

            return chess.Move.from_uci(BLACK_OPENING[self.opening_turn])

        return None

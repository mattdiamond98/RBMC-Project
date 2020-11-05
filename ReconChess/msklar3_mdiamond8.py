#!/usr/bin/env python3

"""
File Name:      my_agent.py
Authors:        Matthew Sklar and Matan Diamond
Date:           10/31/2020

Description:    Python file for my agent.
Source:         Adapted from recon-chess (https://pypi.org/project/reconchess/)
"""

import random

import chess
import numpy as np
import torch

import msklar3_mdiamond8_chess_helper as chess_helper
import msklar3_mdiamond8_config as config
import msklar3_mdiamond8_mcts as mcts
import msklar3_mdiamond8_nn as nn
from msklar3_mdiamond8_particle_filter import ParticleFilter
from player import Player

MOVE_OPTIONS = 64 * 64
IN_CHANNELS = 2

class MagnusDLuffy(Player):

    def __init__(self):
        self.network = nn.Net(IN_CHANNELS, MOVE_OPTIONS)
        self.mcts = None
        
    def handle_game_start(self, color, board):
        """
        This function is called at the start of the game.

        :param color: chess.BLACK or chess.WHITE -- your color assignment for the game
        :param board: chess.Board -- initial board state
        :return:
        """
        self.state = ParticleFilter(color, board)
     
    def handle_opponent_move_result(self, captured_piece, captured_square):
        """
        This function is called at the start of your turn and gives you the chance to update your board.

        :param captured_piece: bool - true if your opponents captured your piece with their last move
        :param captured_square: chess.Square - position where your piece was captured
        """
        self.state.update_opponent_move_result(captured_piece, captured_square)

    def choose_sense(self, possible_sense, possible_moves, seconds_left):
        """
        This function is called to choose a square to perform a sense on.

        :param possible_sense: List(chess.SQUARES) -- list of squares to sense around
        :param possible_moves: List(chess.Moves) -- list of acceptable moves based on current board
        :param seconds_left: float -- seconds left in the game

        :return: chess.SQUARE -- the center of 3x3 section of the board you want to sense
        :example: choice = chess.A1
        """
        # TODO: update this method
        return random.choice(possible_sense)
        
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
        # TODO: update this method
        action = self.pick_action(self.gen_state(self.board))
        print(action)
        # self.mcts.to_string()
        choice = random.choice(possible_moves)
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
        self.particle_filter.update_handle_move_result(taken_move, captured_piece, captured_square)
        
    def handle_game_end(self, winner_color, win_reason):  # possible GameHistory object...
        """
        This function is called at the end of the game to declare a winner.

        :param winner_color: Chess.BLACK/chess.WHITE -- the winning color
        :param win_reason: String -- the reason for the game ending
        """
        print("I'm gonna be king of the chess players!")

    def pick_action(self, state):
        # Create MCT
        root = mcts.Node(state)
        self.mcts = mcts.MCTS(root, self.color)

        # Train the MCT
        for _ in range(config.MCTS_SIMULATIONS):
            self.simulate()

        # Choose the optimal action given the MCT
        action = self.select_move(config.TAU)

        # Take action and get next state

        return action


    def simulate(self):
        # Selection
        leaf, path = self.mcts.select()

        # Evaluation and Expansion TODO: Expansion
        pi, v = self.evaluate_leaf(leaf)

        # TODO: Filter out impossible moves?
        # possible_moves = self.board.legal_moves

        # TODO: Add policies to MCTS
        # this is wrong just placeholder, actions here don't mean anything
        for action, prob in enumerate(pi):
            self.mcts.leaf.edges.append(mcts.Edge(
                self.mcts.leaf,
                mcts.Node(self.state),
                action,
                prob))

        # Backup
        self.mcts.backfill(v, path)

    def evaluate_leaf(self, leaf):
        pi, v = self.network.forward(leaf.state)

        return pi, v

    def select_move(self, tau):
        pi, values = self.policy(tau)

        if tau == 0:    # Deterministic
            action = random.choice(np.anywhere(pi == max(pi)))
        else:           # Stochastically
            action_id = np.random.multinomial(1, pi)
            action = np.where(action_id == 1)[0][0]

        value = values[action]

        print('selected move from action:', action, 'with value:', value)

        return action, value

    # Generate pi and get values to pass through
    def policy(self, tau):
        edges = self.mcts.root.edges
        pi = np.zeros(MOVE_OPTIONS, dtype=np.float32)
        values = np.zeros(MOVE_OPTIONS, dtype=np.float32)

        for edge in edges:
            values[edge.action] = edge.data['Q']

            if tau == 0:
                pi[edge.action] = edge.data['N']
            else:
                pi[edge.action] = pow(edge.data['N'], 1 / tau)

        pi = pi / np.sum(pi)

        return pi, values

    # Generate the representation of the state for a neural network
    def gen_state(self, node, board, color):
        board_array = chess_helper.fen_to_board(board)
        player_layer = np.full((8,8), color)
        
        nn_state = torch.tensor([[board_array, player_layer]])
        
        return nn_state


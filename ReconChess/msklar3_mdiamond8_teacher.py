import chess
import torch

import msklar3_mdiamond8 as agent
import msklar3_mdiamond8_config as config
import random_agent
from msklar3_mdiamond8_memory import GameMemory
from msklar3_mdiamond8_play_game import play_local_game

'''
Play games and record data. Then train using the recorded data
'''
def epoch(games_per_epoch):
    game_memory = []

    for _ in range(games_per_epoch):
        pass

class Teacher():
    def __init__(self, agent, opponent, games_per_epoch):
        self.game_history = []  # list of game history objects
        self.agent = agent
        self.opponent = opponent
        self.games_per_epoch = games_per_epoch

        self.network = torch.load('network.torch')
        print(self.network)

    def epoch(self):
        for _ in range(self.games_per_epoch):
            self.play_game()

    '''
    Play a game and obtain game history data
    '''
    def play_game(self):
        results = play_local_game(self.agent, self.opponent, ['white','black'])
        
        if self.agent.color == chess.WHITE:
            self.add_game(results[2])
        elif self.agent.color == chess.BLACK:
            self.add_game(results[3])

    def add_game(self, game):
        if isinstance(game, GameMemory):
            self.game_history.append(game)

if __name__ == "__main__":
    teacher = Teacher(agent.MagnusDLuffy(), random_agent.Random(), config.GAMES_PER_EPOCH)

    teacher.epoch()

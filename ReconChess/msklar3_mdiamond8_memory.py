class GameMemory:
    '''
    Stored game information

    turns := array of TurnMemory objects for each turn in the game
    v := actual outcome of the game
    game_in_play := whether this game is currently in play
    '''
    def __init__(self):
        self.turns = []
        self.v = None
        self.game_in_play = False

    def add_turn(self, turn):
        self.turns.append(turn)

class TurnMemory:
    '''
    Stored turn information

    state := state of the game
    v_t := prediction outcome of the game at timestep t
    pi := probability distribution for a state s at time step t
    p := probability distribution from MCTS of taking actions at time step t
    '''
    def __init__(self, state, v_t, pi, p):
        self.state = state
        self.v_t = v_t
        self.pi = pi
        self.p = p
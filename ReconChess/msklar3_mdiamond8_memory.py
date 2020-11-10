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

    def to_string(self):
        string = ''
        
        string += '{value}\n'.format(value=self.v)
        
        for turn in self.turns:
            string += turn.to_string()

        return string

class TurnMemory:
    '''
    Stored turn information

    state := state of the game
    v_t := prediction outcome of the game at timestep t
    p := probability distribution from MCTS of taking actions at time step t
    '''
    def __init__(self, state, v_t, pi, p):
        self.state = state
        self.v_t = v_t
        self.pi = pi
        self.p = p

    def to_string(self):
        return '{state}, {v_t}, {pi}, {p}\n'.format(
            state=str(self.state.detach()),
            v_t = str(self.v_t),
            pi=str(self.pi),
            p=str(self.p)
        )

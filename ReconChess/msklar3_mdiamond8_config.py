MCTS_SIMULATIONS = 200
TAU = 0.3   # Move selection exploration
LOG_EPSILON = 1e-15 # Add to probabilities so you never take log 0
LAMBDA = 0.1   # For l2-normalization overfitting via regulation prevention (TM)

NN_DECISION_WEIGHT_ALPHA = .3    # how much the neural network output decides the action

SIMULATION_EXPANSION = 15

EPOCHS = 10
GAMES_PER_EPOCH = 5 # Games played before training network
TRAINING_TURNS_PER_EPOCH = 250

STOCKFISH_DEPTH = 20
RUN_STOCKFISH = True

moves_till_loss = 200
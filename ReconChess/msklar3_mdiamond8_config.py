EPSILON = 0.5
MCTS_SIMULATIONS = 100
TAU = 0.3   # Move selection exploration
LOG_EPSILON = 1e-15 # Add to probailities so you never take log 0
LAMBDA = 0.1   # For l2-normalization overfitting via regulation prevention (TM)

NN_DECISION_WEIGHT_ALPHA = 0    # how much the neural network output decides the action

SIMULATION_EXPANSION = 10

EPOCHS = 5
GAMES_PER_EPOCH = 10 # Games played before training network
TRAINING_TURNS_PER_EPOCH = 100

STOCKFISH_DEPTH = 2

moves_till_loss = 200
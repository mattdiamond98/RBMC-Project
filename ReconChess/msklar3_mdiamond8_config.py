EPSILON = 0.5
MCTS_SIMULATIONS = 100
TAU = 0.8   # Move selection exploration
LOG_EPSILON = 1e-15 # Add to probailities so you never take log 0
LAMBDA = 0.1   # For l2-normalization overfitting via regulation prevention (TM)

SIMULATION_EXPANSION = 10

EPOCHS = 10
GAMES_PER_EPOCH = 5 # Games played before training network
TRAINING_TURNS_PER_EPOCH = 20
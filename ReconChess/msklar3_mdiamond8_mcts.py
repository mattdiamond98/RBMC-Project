import msklar3_mdiamond8_config as config

class Node():
    def __init__(self, state, id):
        self.state = state
        self.id = id

        self.edges = []

    def is_leaf(self):
        return self.edges == 0

    def is_agent(self):
        return self.state[1][0][0]

class Edge():
    def __init__(self, in_node, out_node, action, prior):
        self.in_node = in_node
        self.out_node = out_node
        self.action = action

        self.data = {
            'N' : 0,        # number of times action a has been taken from state s
            'W' : 0,        # total value of next state
            'Q' : 0,        # mean value of the next state
            'P' : prior,    # prior probability of selecting action a
        }

class MCTS():
    def __init__(self, root):
        self.root = root
        self.tree = {}

        self.addNode(root)

    # Navigate to and select a leaf for expansion
    def select(self):
        node = self.root
        path = []

        while not node.is_leaf():
            action = self.max_action(node.edges)

            path.append(node)
            node = action.out_node

        return node, path

    # Update the Monte-Carle Tree based on simulation results
    def backfill(self, leaf_node, v, path):
        for action in path:
            action.data['N'] = action.data['N'] + 1
            action.data['W'] = action.data['W'] + v * action.in_node.is_agent()
            action.data['Q'] = action.data['W'] / action.data['N']
            

    # Select action that maximizes Q + U. Return the edge representing the best action
    def max_action(self, actions):
        max_a = None
        max_QU = None

        for a in actions:
            Q = a.data['Q']
            U = self.calculate_U(a.data['P'], a.data['N'])

            if max_QU == None or Q + U > max_QU:
                max_a = a
                max_QU = Q + U

        return max_a

    # Calculate U value
    # TODO: Might want to update this to include epsilon but must be proportional to this
    def calculate_U(self, P, N):
        return P / (1 + N)

    def addNode(self, node):
        self.tree[node.id] = node

import numpy as np
from typing import List, Dict, Optional, Tuple
import math
from copy import deepcopy
from game.core import BoardState, BattleshipEnv, _HIT_IDX, _MISS_IDX

class Node:
    def __init__(self, state: BoardState, parent: Optional['Node'] = None, action: Optional[int] = None):
        self.state = deepcopy(state)
        self.parent = parent
        self.action = action
        self.children: Dict[int, Node] = {}
        self.visits = 0
        self.value = 0.0
        self.untried_actions = self._get_valid_actions()

    def _get_valid_actions(self) -> List[int]:
        """Returns list of valid actions (unattacked positions) for current player."""
        opponent = 1 - self.state.current_player
        valid = []
        hit_board = self.state.hit_boards[opponent]
        for i in range(100):  # 10x10 board
            row, col = divmod(i, 10)
            if hit_board[row, col] not in [_HIT_IDX, _MISS_IDX]:
                valid.append(i)
        return valid

    def is_terminal(self) -> bool:
        return self.state.done

    def is_fully_expanded(self) -> bool:
        return len(self.untried_actions) == 0

class MCTS:
    def __init__(self, env: BattleshipEnv, exploration_constant: float = 1.414):
        self.env = env
        self.exploration_constant = exploration_constant

    def search(self, root_state: BoardState, num_simulations: int) -> int:
        """Conduct MCTS search from root state and return best action."""
        root = Node(root_state)
        
        for _ in range(num_simulations):
            node = root
            state = deepcopy(root_state)

            # Selection
            while not node.is_terminal() and node.is_fully_expanded():
                node = self._select_child(node)
                state = deepcopy(node.state)

            # Expansion
            if not node.is_terminal() and not node.is_fully_expanded():
                action = np.random.choice(node.untried_actions)
                node.untried_actions.remove(action)
                new_state, reward, done = self.env.step(deepcopy(state), action)
                node = self._add_child(node, new_state, action)
                state = new_state

            # Simulation
            reward = self._simulate(state)

            # Backpropagation
            while node is not None:
                node.visits += 1
                # Negamax value update: flip sign when moving up the tree
                node.value += reward if node.state.current_player == root_state.current_player else -reward
                node = node.parent
                reward = -reward

        # Select best action based on average value
        return self._select_best_action(root)

    def _select_child(self, node: Node) -> Node:
        """Select child node using UCT formula."""
        return max(node.children.values(),
                  key=lambda n: (n.value / n.visits if n.visits > 0 else float('inf')) + 
                               self.exploration_constant * math.sqrt(math.log(node.visits) / n.visits if n.visits > 0 else float('inf')))

    def _add_child(self, node: Node, state: BoardState, action: int) -> Node:
        """Add new child node to the tree."""
        child = Node(state, parent=node, action=action)
        node.children[action] = child
        return child

    def _simulate(self, state: BoardState) -> float:
        """Simulate random playout and return reward."""
        state = deepcopy(state)
        last_reward = 0.0  # Initialize reward
        
        while not state.done:
            # Random policy: choose random valid action
            opponent = 1 - state.current_player
            valid_actions = []
            for i in range(100):
                row, col = divmod(i, 10)
                if state.hit_boards[opponent][row, col] not in [_HIT_IDX, _MISS_IDX]:
                    valid_actions.append(i)
            
            if not valid_actions:
                break
                
            action = np.random.choice(valid_actions)
            state, reward, done = self.env.step(state, action)
            last_reward = reward  # Keep track of the last reward
            
        # Return final reward from perspective of player who started simulation
        # If game is done, use last_reward, otherwise use a draw value (0)
        if state.done:
            return last_reward if state.current_player == 1 else -last_reward
        else:
            return 0.0  # Draw/stalemate case

    def _select_best_action(self, root: Node) -> int:
        """Select best action based on average value."""
        return max(root.children.items(),
                  key=lambda x: x[1].visits)[0]  # Using most visited child as best action

class MCTSPlayer:
    def __init__(self, env: BattleshipEnv, num_simulations: int = 1000):
        self.mcts = MCTS(env)
        self.num_simulations = num_simulations

    def get_action(self, state: BoardState) -> int:
        """Get best action for current state using MCTS."""
        return self.mcts.search(state, self.num_simulations)

# Example usage:
def play_game():
    env = BattleshipEnv()
    state = env.reset()
    
    # Create two MCTS players
    player1 = MCTSPlayer(env, num_simulations=10)
    player2 = MCTSPlayer(env, num_simulations=10)
    
    while not state.done:
        if state.current_player == 0:
            action = player1.get_action(state)
        else:
            action = player2.get_action(state)
            
        state, reward, done = env.step(state, action)
        
    return state.current_player, reward
import heapq
import numpy as np
from copy import deepcopy
import time

class Node:
    def __init__(self, state, depth, log_probability):
        self.parent = None
        self.state = state
        self.depth = depth
        self.log_probability = log_probability
        self.cost = np.log(depth) - log_probability

    def __lt__(self, other):
        return self.cost < other.cost

class LevinTree:
    def __init__(self, model, initial_state):
        self.model = model
        self.action_list = model.decoder.id2word.keys()
        #self.action_list = LevinTreeSearch._filter_action_list(model)
        self.open_list = [Node(initial_state, 1, 0.0)] # root node

    def fit(self):
        self.num_expansion = 1
        self.num_generated = 1
        start = time.time()
        while len(self.open_list) > 0:
            result_state, node = self._expand()
            if result_state is not None:
                end = time.time()
                print(f"Time {end - start}")
                print(f"Number of expansions: {self.num_expansion}")
                print(f"Number of generation: {self.num_generated}")
                print(f"Path length: {node.depth}")
                return result_state
        

    def _expand(self):
        node = heapq.heappop(self.open_list)
        self.num_expansion += 1
        if self.model.is_solution(node.state):
            return node.state, node
        for action_index in self.action_list:
            new_state = self.model.apply_action(deepcopy(node.state), idx=action_index) # force to take this action
            probability = self.model.get_policy(new_state)[0][action_index] # and then retrieve probability from it
            child_node = Node(new_state, node.depth + 1, np.log(probability) + node.log_probability)
            # Terminate at the first solution
            #if self.model.is_solution(new_state):
            #    return new_state, child_node
            # Skip constants
            #if self._is_constant(action_index):
            #    continue
            heapq.heappush(self.open_list, child_node)
            self.num_generated += 1
        return None, None # not finish training
    
    def _is_constant(self, action_index):
        return self.model.decoder.id2word[action_index] in ["+", "-"]

    @staticmethod
    def _filter_action_list(model):
        # Filter out constants, leave symbols, this reduce branch factors
        filtered = {k: v for k, v in model.decoder.id2word.items() if v[0] != 'E' and v[0] != 'N'}
        return filtered.keys() # we only want ids, that is action indices

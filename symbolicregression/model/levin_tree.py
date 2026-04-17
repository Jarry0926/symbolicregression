import heapq
import numpy as np
import torch
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
        #self.action_list = model.decoder.id2word.keys()
        #self.action_list = LevinTreeSearch._filter_action_list(model)
        self.open_list = [Node(initial_state, 1, 0.0)] # root node

    def search(self):
        budget = int(1e6)
        self.num_expansion = 0
        self.num_generated = 1
        start = time.time()
        while len(self.open_list) > 0 and budget > 0:
            result_state, node = self._expand()
            budget -= 1
            if result_state is not None:
                end = time.time()
                print(f"Time {end - start}")
                print(f"Number of expansions: {self.num_expansion}")
                print(f"Number of generation: {self.num_generated}")
                print(f"Path length: {node.depth}")
                return result_state
        print("Out of budget\n")
        return None

    def _expand(self):
        node = heapq.heappop(self.open_list)
        self.num_expansion += 1
        #print(f"depth={node.depth}, log probability={node.log_probability}, cost={node.cost}")
        if self.model.is_solution(node.state):
            return node.state, node
        policy_list = self.model.get_policy(node.state)[0]
        k = 2
        policy_list_top_k, action_list_top_k = torch.topk(policy_list, k)
        policy_list_top_k = policy_list_top_k.detach().cpu().tolist()
        action_list_top_k = action_list_top_k.detach().cpu().tolist()
        policy_list_top_k[1] *= 1e-2
        for i in range(k):
            new_state = self.model.apply_action(deepcopy(node.state), idx=action_list_top_k[i])
            probability = policy_list_top_k[i]
            # Skip zero probability
            if probability == 0.0:
                continue
            child_node = Node(new_state, node.depth + 1, np.log(probability) + node.log_probability)
            # Skip constants
            #if self._is_constant(action_index):
            #    continue
            heapq.heappush(self.open_list, child_node)
            self.num_generated += 1
        return None, None
    
    def _is_constant(self, action_index):
        return self.model.decoder.id2word[action_index] in ["+", "-"]

    @staticmethod
    def _filter_action_list(model):
        # Filter out constants, leave symbols, this reduce branch factors
        filtered = {k: v for k, v in model.decoder.id2word.items() if v[0] != 'E' and v[0] != 'N'}
        return filtered.keys() # we only want ids, that is action indices

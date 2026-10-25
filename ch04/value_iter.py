if "__file__" in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from collections import defaultdict
from common.gridworld import GridWorld

def value_iter_onestep(V, env, gamma):
    for state in env.states():
        if state == env.goal_state:
            V[state] = 0
            continue
    
        action_values = []
        for action in env.actions():
            next_state = env.next_state(state, action)
            r = env.reward(state, action , next_state)
            value = r + gamma * V[next_state]
            action_values.append(value)
        
        V[state] = max(action_values)
    
    return V

def value_iter(V, env, gamma, threshold=0.001, is_render=True):
    while True:
        if is_render:
            env.render_V(V)
        
        old_V = V.copy()
        V = value_iter_onestep(V, env, gamma)

        delta = 0
        
        for state in V.keys():
            t = abs(V[state] - old_V[state])
            if delta < t:
                delta = t
                
        if delta < threshold:
            break
        
    return V

def argmax(d):
    """d (dict)"""
    max_value = max(d.values())
    max_key = -1
    for key, value in d.items():
        if value == max_value:
            max_key = key
    return max_key

def greedy_policy(V, env, gamma):
    pi = {}

    for state in env.states():
        action_values = {}

        for action in env.actions():
            next_state = env.next_state(state, action)
            r = env.reward(state, action, next_state)
            value = r + gamma * V[next_state]
            action_values[action] = value
        
        max_action = argmax(action_values)
        action_probs = {0: 0, 1: 0, 2: 0, 3: 0}
        action_probs[max_action] = 1.0
        pi[state] = action_probs
    
    return pi

if __name__ == "__main__":
    V = defaultdict(lambda : 0)
    env = GridWorld()
    gamma = 0.9
    
    V = value_iter(V, env, gamma)
    pi = greedy_policy(V, env, gamma)
    env.render_V(V, pi)
            
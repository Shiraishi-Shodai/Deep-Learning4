if "__file__" in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from common.gridworld import GridWorld
from collections import defaultdict

def eval_onestop(pi, V, env, gamma=0.9):
    for state in env.states():
        if state == env.goal_state:
            V[state] = 0
            continue
        
        action_probs = pi[state]
        new_V = 0
        # 状態stateにおける各行動actionをとったときの収益の期待値を計算
        for action in env.actions():
            next_state = env.next_state(state, action)
            r = env.reward(state, action, next_state)
            new_V += action_probs[action] * (r + gamma * V[next_state])

        V[state] = new_V
    
    return V

def policy_eval(pi, V, env, gamma, threshold=0.001):
    while True:
        old_V = V.copy() # 更新前の状態価値関数
        V = eval_onestop(pi, V, env, gamma)

        delta = 0.0
        # 更新された量の最大値を求める
        for state in V.keys():
            t = abs(V[state] - old_V[state])

            if delta < t:
                delta = t

        # 閾値との比較
        if delta < threshold:
            break
    
    return V

def argmax(d):
    max_value = max(d.values())
    max_key = None
    for key in d.keys():
        if max_value == d[key]:
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
        
        action_probs = {0: 0, 1: 0, 2: 0, 3: 0}
        max_action = argmax(action_values)
        action_probs[max_action] = 1.0
        pi[state] = action_probs
    return pi

def policy_iter(env, gamma, threshold=0.001, is_render=True):
    pi = defaultdict(lambda: {0: 0.25, 1: 0.25, 2: 0.25, 3: 0.25})
    V = defaultdict(lambda: 0)
    cnt = 0
    
    while True:
        V = policy_eval(pi, V, env, gamma, threshold)
        new_pi = greedy_policy(V, env, gamma)
        cnt += 1
        
        if is_render:
            env.render_V(V, new_pi)
            
        if new_pi == pi:
            break
        
        pi = new_pi.copy()
    # print(f"{cnt}回目で最適方策発見")
    return pi

if __name__ == "__main__":
    env = GridWorld()
    V = defaultdict(lambda: 0)
    pi = defaultdict(lambda : {0 : 1, 1: 0, 2 : 0, 3: 0})
    gamma = 0.9
    
    # optimal_V = policy_eval(pi, V, env, gamma)
    pi = policy_iter(env, gamma, threshold=0.001)
    for i, k in pi.items():
        print(i, k)
    
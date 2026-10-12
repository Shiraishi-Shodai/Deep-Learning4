import numpy as np
from matplotlib import pyplot as plt

class Bandit:
    def __init__(self, arms):
        self.arms = arms
        self.rates = np.random.rand(arms)
    
    def play(self, arm):
        rate = self.rates[arm]
        self.rates += 0.1 * np.random.randn(self.arms)
        
        if rate > np.random.rand():
            return 1
        else:
            return 0

class Agent:
    def __init__(self, actions, epsilon, alpha):
        self.actions = actions
        self.epsilon = epsilon
        self.alpha = alpha
        self.ns = np.zeros((actions,))
        self.Qs = np.zeros((actions,))
    
    def update(self, action, reward):
        self.ns[action] += 1
        self.Qs[action] += self.alpha * (reward - self.Qs[action])
    
    def get_action(self):
        utilization_probability = np.random.randn()
        if utilization_probability < self.epsilon:
            return np.random.randint(self.actions)
        
        return np.argmax(self.Qs)
    

if __name__ == "__main__":
    runs = 10
    steps = 1000
    epsilon = 0.1
    alpha = 0.9
    actions = 10
    
    bandit = Bandit(actions)
    agent = Agent(actions=actions, epsilon=epsilon, alpha=alpha)
    
    total_rewards_history = np.zeros((runs, steps))
    rates_history = np.zeros((runs, steps))

    for run in range(runs):
        total_reward = 0
        total_rewards = np.zeros((steps))
        rates = np.zeros((steps))
        for step in range(steps):
            arm = agent.get_action()
            reward = bandit.play(arm)
            agent.update(action=arm, reward=reward)
            
            total_reward += reward
            total_rewards[step] = total_reward
            rates[step] = total_reward / (step + 1)
        
        total_rewards_history[run] = total_rewards
        rates_history[run] = rates
    
    print(f"最終的な期待値: {rates_history.mean(axis=0)[steps - 1]}")
    
    plt.xlabel("Steps")
    plt.ylabel("Total Rewards")
    plt.plot(total_rewards_history.T)
    plt.show()

    plt.xlabel("Steps")
    plt.ylabel("Rewards")
    plt.plot(rates_history.T)
    plt.show() 

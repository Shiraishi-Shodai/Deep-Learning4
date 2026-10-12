import numpy as np
from matplotlib import pyplot as plt

class Bandit:
    def __init__(self, arms=10):
        self.rates = np.random.rand(arms)
    
    def play(self, arm):
        if self.rates[arm] > np.random.rand():
            return 1
        else:
            return 0

class Agent:
    def __init__(self, epsilon=0.1, actions=10):
        self.epsilon = epsilon
        self.ns = np.zeros((actions,))
        self.Qs = np.zeros((actions,))
    
    def update(self, action, reward):
        self.ns[action] += 1
        self.Qs[action] += (reward - self.Qs[action]) / self.ns[action]
    
    def get_action(self):
        utilization_probability = np.random.rand()
        if utilization_probability < self.epsilon:
            return np.random.randint(len(self.Qs))
        return np.argmax(self.Qs)


if __name__ == "__main__":
    runs = 200
    steps = 1000
    actions = 10
    epsilon = 0.1
    
    bandit = Bandit(arms=actions)
    agent = Agent(actions=actions)
        
    total_rewards_history = np.zeros((runs, steps))
    rates_history = np.zeros((runs, steps))
    
    # print(f"bandit : arms{bandit.rates}")
    for run in range(runs):
        total_reward = 0
        total_rewards = np.zeros((steps, ))
        rates = np.zeros((steps, ))
        for step in range(steps):
            
            action = agent.get_action()
            reward = bandit.play(action)
            agent.update(action=action, reward=reward)

            total_reward += reward
            total_rewards[step] = total_reward
            rates[step] = total_reward / (step + 1)
            
            # print(f"get action: {action}")
        total_rewards_history[run] = total_rewards
        rates_history[run] = rates
        print(total_reward)
        
    print(total_rewards_history.shape)
    print(rates_history.shape)
    plt.ylabel('Total reward')
    plt.xlabel('Steps')
    # plt.plot(total_rewards_history.T)
    plt.plot(total_rewards_history.mean(axis=0))
    plt.show()

    plt.ylabel('Rates')
    plt.xlabel('Steps')
    # plt.plot(rates_history.T)
    plt.plot(rates_history.mean(axis=0))
    plt.show()
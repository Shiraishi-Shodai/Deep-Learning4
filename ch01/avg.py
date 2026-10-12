import numpy as np


np.random.seed(0)
rewards = []

print(f"pattern1")
for n in range(1, 11):
    reward = np.random.rand()
    rewards.append(reward)
    Q = sum(rewards) / n
    print(Q)


np.random.seed(0)
print(f"pattern2")

Q = 0

for n in range(1, 11):
    reward = np.random.rand()
    Q += (reward - Q) /n
    print(Q)
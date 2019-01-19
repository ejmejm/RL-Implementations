import numpy as np

def discount_rewards(rewards, gamma=0.99):
    new_rewards = [rewards[-1]]
    for i in reversed(range(len(rewards)-1)):
        new_rewards.append(rewards[i] + gamma * new_rewards[-1])
    return new_rewards[::-1]
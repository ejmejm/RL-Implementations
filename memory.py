import numpy as np
from rewards import discount_rewards

class MemoryBuffer():
    def __init__(self, max_size):
        self.max_size = max_size
        self.rollouts = []
        self.rollout_idx = -1
    
    def start_rollout(self):
        self.rollout_idx = (self.rollout_idx + 1) % self.max_size
        if self.rollout_idx >= len(self.rollouts):
            self.rollouts.append([])
        else:
            self.rollouts[self.rollout_idx] = []
            
    def end_rollout(self):
        self.start_rollout()
    
    def record(self, obs, act, rew):
        self.rollouts[self.rollout_idx].append([obs, act, rew])
        
    def to_data(self, reset=True):
        all_data = []
        
        try:
            for rollout in self.rollouts:
                rollout = np.array(rollout)
                # Discount the rewards for every rollout
                rollout[:,2] = discount_rewards(rollout[:,2])
                all_data.extend(list(rollout))

            if reset:
                self.reset()
        except IndexError:
            return np.array([])
            
        return np.array(all_data)
                
    def reset(self):
        self.rollouts = []
        self.rollout_idx = -1
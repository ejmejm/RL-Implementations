import numpy as np
from rewards import discount_rewards
import threading
import multiprocessing

class MemoryBuffer():
    def __init__(self, incl_next_state=False, max_size=1e6):
        self.max_size = int(max_size)
        self.rollouts = []
        self.rollout_idx = -1
        if incl_next_state:
            self.n_vars = 4
            self.record = self._record_4_vars
        else:
            self.n_vars = 3
            self.record = self._record_3_vars
    
    def start_rollout(self):
        self.rollout_idx = (self.rollout_idx + 1) % self.max_size
        if self.rollout_idx >= len(self.rollouts):
            self.rollouts.append([])
        else:
            self.rollouts[self.rollout_idx] = []
            
    def end_rollout(self):
        self.start_rollout()
    
    def _record_3_vars(self, obs, act, rew):
        self.rollouts[self.rollout_idx].append([obs, act, rew])
    
    def _record_4_vars(self, obs, act, rew, obs_next):
        self.rollouts[self.rollout_idx].append([obs, act, rew, obs_next])
        
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
                
    def get_avg_reward(self):
        total_reward = 0
        for rollout in self.rollouts:
            rollout = np.array(rollout)
            rewards = rollout[:,2]
            total_reward += np.sum(rewards)
            
        return total_reward / len(self.rollouts)
        
    def reset(self):
        self.rollouts = []
        self.rollout_idx = -1
        
class MTMemoryBuffer():
    """
    Multi-threading Memory Buffer
    """
    def __init__(self, max_size=1e6):
        self.max_size = max_size
        self.rollouts = []
        self.rollout_idx = -1
        self.agent_map = {}
        self.lock = threading.Lock()
    
    def start_rollout(self, agent_id):
        with self.lock:
            self.rollout_idx = int((self.rollout_idx + 1) % self.max_size)
            if self.rollout_idx >= len(self.rollouts):
                self.rollouts.append([])
            else:
                self.rollouts[self.rollout_idx] = []
            self.agent_map[agent_id] = self.rollout_idx
            
    def end_rollout(self, agent_id):
        self.start_rollout(agent_id)
    
    def record(self, agent_id, obs, act, rew):
        self.rollouts[self.agent_map[agent_id]].append([obs, act, rew])
        
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
                
    def get_avg_reward(self):
        total_reward = 0
        for rollout in self.rollouts:
            rollout = np.array(rollout)
            rewards = rollout[:,2]
            total_reward += np.sum(rewards)
            
        return total_reward / len(self.rollouts)
                
    def reset(self):
        self.rollouts = []
        self.rollout_idx = -1
        
class MPMemoryBuffer():
    """
    Multi-processing Memory Buffer
    """
    def __init__(self, max_size=1e6):
        self.max_size = max_size
        self.rollouts = []
        self.rollout_idx = -1
        self.agent_map = {}
        self.lock = multiprocessing.Lock()
    
    def start_rollout(self, agent_id):
#         with self.lock:
        self.rollout_idx = int((self.rollout_idx + 1) % self.max_size)
        if self.rollout_idx >= len(self.rollouts):
            self.rollouts.append([])
        else:
            self.rollouts[self.rollout_idx] = []
        self.agent_map[agent_id] = self.rollout_idx
            
    def end_rollout(self, agent_id):
        self.start_rollout(agent_id)
    
    def record(self, agent_id, obs, act, rew):
        self.rollouts[self.agent_map[agent_id]].append([obs, act, rew])
        
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
                
    def get_avg_reward(self):
        total_reward = 0
        for rollout in self.rollouts:
            rollout = np.array(rollout)
            rewards = rollout[:,2]
            total_reward += np.sum(rewards)
            
        return total_reward / len(self.rollouts)
                
    def reset(self):
        self.rollouts = []
        self.rollout_idx = -1
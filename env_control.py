import numpy as np
from memory import MTMemoryBuffer
import threading
import time

class EnvController():
    def __init__(self, make_env, n_threads=1, memory_buffer=MTMemoryBuffer(), 
                 obs_transform=None, act_transform=None):
        self.make_env = make_env
        self.mb = memory_buffer
        self.n_threads = n_threads
        if obs_transform is not None:
            self.obs_transform = obs_transform
        if act_transform is not None:
            self.act_transform = act_transform
        self.act_lock = threading.Lock()
        self.init_lock = threading.Lock()
        
    def obs_transform(self, obs):
        return obs.squeeze()
    
    def act_transform(self, act):
        return act
    
    def set_obs_transform(self, transform_func):
        self.obs_transform = transform_func
    
    def set_act_transform(transform_func):
        self.act_transform = transform_func
        
    def sim_thread(self, agent_id, network, n_episodes=1, max_steps=200, render=False):
        with self.init_lock:
            env = self.make_env()
        
        for episode in range(n_episodes):
            self.mb.start_rollout(agent_id)
            obs = env.reset()
            obs = self.obs_transform(obs)
            for step in range(max_steps):
                with self.act_lock:
                    act = network.gen_act(obs)
                act = self.act_transform(act)

                obs_next, rew, d, _ = env.step(act)
                obs_next = self.obs_transform(obs_next)

                if render:
                    env.render()
                    time.sleep(0.02)

                self.mb.record(agent_id, obs, act, rew, obs_next)
                obs = obs_next

                if d:
                    break
                    
    def sim_episodes(self, network, n_episodes=1, max_steps=200, render=False, return_data=False):
        threads = []
        ept = [int(n_episodes // self.n_threads) for i in range(self.n_threads)] # Episodes per thread
        ept[:(n_episodes % self.n_threads)] += np.ones((n_episodes % self.n_threads,))
        for i in range(self.n_threads):
            new_thread = threading.Thread(target=self.sim_thread, args=(i, network, int(ept[i]), max_steps, render))
            threads.append(new_thread)
            new_thread.start()
            
        for thread in threads:
            thread.join()
        
        if return_data:
            return self.mb.to_data()
        
    def get_avg_reward(self):
        return self.mb.get_avg_reward()
    
    def get_data(self):
        return self.mb.to_data()
    
    def render_episodes(self, network, n_episodes=1, max_steps=200):
        with self.init_lock:
            env = self.make_env()
        
        for episode in range(n_episodes):
            obs = env.reset()
            obs = self.obs_transform(obs)
            for step in range(max_steps):
                with self.act_lock:
                    act = network.gen_act(obs)
                act = self.act_transform(act)

                obs_next, rew, d, _ = env.step(act)
                obs_next = self.obs_transform(obs_next)
                obs = obs_next

                env.render()
                time.sleep(0.02)

                if d:
                    break
        
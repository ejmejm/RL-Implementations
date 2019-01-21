import tensorflow as tf
from utils import reshape_train_var
import numpy as np

class VPGTrainer():
    def __init__(self, in_op, out_op, act_type='discrete', sess=None):
        """
        Create a wrapper for RL networks for easy training.
        Args:
            in_op (tf.Placeholder): Observation input to architecture
            out_op (tf.Variable): Action output of architecture
            act_type (string): 'discrete' for a discrete actions space or 'continuous'
                               for a continuous actions space
            sess (tf.Session): A session if you would like to use a custom session,
                               if left none it will be automatically created
        """

        if not sess:
            self.renew_sess()
        
        self.in_op = in_op
        self.out_op = out_op
        
        if act_type in ('discrete', 'd'):
            self.train = self._create_discrete_trainer()
            self.act_type = 'discrete'
        elif act_type in ('continuous', 'c'):
            self.train = self._create_continuous_trainer()
            self.act_type = 'continuous'
        else:
            raise TypeError('act_type must be \'discrete\' or \'continuous\'')
        
    def renew_sess(self):
        """
        Starts a new internal Tensorflow session
        """
        self.sess = tf.Session()
        
    def end_sess(self):
        """
        Ends the internal Tensorflow session if it exists
        """
        if self.sess:
            self.sess.close()
        
    def _create_discrete_trainer(self, optimizer=tf.train.AdamOptimizer()):
        """
        Creates a function for vanilla policy training with a discrete action space
        """
        self.act_holders = tf.placeholder(tf.int64, shape=[None])
        self.reward_holders = tf.placeholder(tf.float64, shape=[None])
        
        self.act_masks = tf.one_hot(self.act_holders, self.out_op.shape[1].value, dtype=tf.float64)
        self.log_probs = tf.log(self.out_op)
        
        self.resp_acts = tf.reduce_sum(self.act_masks *  self.log_probs, axis=1)
        self.loss = -tf.reduce_mean(self.resp_acts * self.reward_holders)
        
        self.optimizer = optimizer
        self.update = self.optimizer.minimize(self.loss)
        
        update_func = lambda train_data: self.sess.run(self.update, 
                                                       feed_dict={self.in_op: reshape_train_var(train_data[:, 0]),
                                                            self.act_holders: reshape_train_var(train_data[:, 1]),
                                                            self.reward_holders: train_data[:, 2]})
        
        self.sess.run(tf.global_variables_initializer())
        
        return update_func
        
    def _create_continuous_trainer(self, optimizer=tf.train.AdamOptimizer()):
        """
        Creates a function for vanilla policy training with a continuous action space
        """
        self.act_holders = tf.placeholder(tf.float64, shape=[None, self.out_op.shape[1].value])
        self.reward_holders = tf.placeholder(tf.float64, shape=[None])
        
        self.log_probs = tf.log(self.out_op)
        
        self.act_means = tf.reduce_mean(self.log_probs, axis=1)
        self.loss = -tf.reduce_mean(self.act_means * self.reward_holders)
        
        self.optimizer = optimizer
        self.update = self.optimizer.minimize(self.loss)
        
        update_func = lambda train_data: self.sess.run(self.update, 
                                                       feed_dict={self.in_op: reshape_train_var(train_data[:, 0]),
                                                            self.act_holders: reshape_train_var(train_data[:, 1]), # vstack
                                                            self.reward_holders: train_data[:, 2]})
        
        self.sess.run(tf.global_variables_initializer())
        
        return update_func
        
    def _gen_discrete_act(self, obs):
        act_probs = self.sess.run(self.out_op, feed_dict={self.in_op: obs})
        act = np.random.choice(list(range(len(act_probs)+1)), p=act_probs[0])
        
        return act
    
    def _gen_continuous_act(self, obs):
        act_vect = self.sess.run(self.out_op, feed_dict={self.in_op: obs})[0]
        
        # TODO: Add gaussian noise to action vector
        act_vect = [a + np.random.normal(0., 0.1) for a in act_vect]
        
        return np.array(act_vect)
        
    def gen_act(self, obs):
        if self.act_type == 'discrete':
            return self._gen_discrete_act(obs)
        else:
            return self._gen_continuous_act(obs)
        
    def train(self, obs, rewards, acts):
        raise RuntimeError('The train method was not properly created')
        
class PPOTrainer():
    def __init__(self, in_op, out_op, value_out_op, act_type='discrete', sess=None):
        """
        Create a wrapper for RL networks for easy training.
        Args:
            in_op (tf.Placeholder): Observation input to architecture
            out_op (tf.Variable): Action output of architecture
            act_type (string): 'discrete' for a discrete actions space or 'continuous'
                               for a continuous actions space
            sess (tf.Session): A session if you would like to use a custom session,
                               if left none it will be automatically created
        """

        if not sess:
            self.renew_sess()
        
        self.in_op = in_op
        self.out_op = out_op
        self.value_out_op = value_out_op
        self._prev_weights = None
        
        if act_type in ('discrete', 'd'):
            self.train = self._create_discrete_trainer()
            self.act_type = 'discrete'
        elif act_type in ('continuous', 'c'):
            self.train = self._create_continuous_trainer()
            self.act_type = 'continuous'
        else:
            raise TypeError('act_type must be \'discrete\' or \'continuous\'')
        
    def renew_sess(self):
        """
        Starts a new internal Tensorflow session
        """
        self.sess = tf.Session()
        
    def end_sess(self):
        """
        Ends the internal Tensorflow session if it exists
        """
        if self.sess:
            self.sess.close()
        
    def _create_discrete_trainer(self, optimizer=tf.train.AdamOptimizer()):
        """
        Creates a function for vanilla policy training with a discrete action space
        """
        self.act_holders = tf.placeholder(tf.int64, shape=[None])
        self.reward_holders = tf.placeholder(tf.float64, shape=[None])
        
        self.act_masks = tf.one_hot(self.act_holders, self.out_op.shape[1].value, dtype=tf.float64)
        self.log_probs = tf.log(self.out_op)
        
        self.resp_acts = tf.reduce_sum(self.act_masks *  self.log_probs, axis=1)
        self.actor_loss = -tf.reduce_mean(self.resp_acts * self.reward_holders)
        
        self.value_loss = tf.reduce_mean(tf.math.square(self.rewards_holders - self.value_out_op))
        
        self.optimizer = optimizer
        self.actor_update = self.optimizer.minimize(self.actor_loss)
        self.value_update = self.optimizer.minimize(self.value_loss)
        
        update_func = lambda train_data: self.sess.run([self.actor_update, self.value_update], 
                                                       feed_dict={self.in_op: reshape_train_var(train_data[:, 0]),
                                                            self.act_holders: reshape_train_var(train_data[:, 1]),
                                                            self.reward_holders: train_data[:, 2]})
        
        self.sess.run(tf.global_variables_initializer())
        self._prev_weights = tf.trainabale_variables()
        
        return update_func
        
    def _create_continuous_trainer(self, optimizer=tf.train.AdamOptimizer()):
        """
        Creates a function for vanilla policy training with a continuous action space
        """
        self.act_holders = tf.placeholder(tf.float64, shape=[None, self.out_op.shape[1].value])
        self.reward_holders = tf.placeholder(tf.float64, shape=[None])
        
        self.log_probs = tf.log(self.out_op)
        
        self.act_means = tf.reduce_mean(self.log_probs, axis=1)
        self.actor_loss = -tf.reduce_mean(self.act_means * self.reward_holders)
        
        self.value_loss = tf.reduce_mean(tf.math.square(self.rewards_holders - self.value_out_op))
        
        self.optimizer = optimizer
        self.actor_update = self.optimizer.minimize(self.actor_loss)
        self.value_update = self.optimizer.minimize(self.value_loss)
        
        update_func = lambda train_data: self.sess.run([self.actor_update, self.value_update], 
                                                       feed_dict={self.in_op: reshape_train_var(train_data[:, 0]),
                                                            self.act_holders: reshape_train_var(train_data[:, 1]), # vstack
                                                            self.reward_holders: train_data[:, 2]})
        
        self.sess.run(tf.global_variables_initializer())
        
        return update_func
        
    def _gen_discrete_act(self, obs):
        act_probs = self.sess.run(self.out_op, feed_dict={self.in_op: obs})
        act = np.random.choice(list(range(len(act_probs)+1)), p=act_probs[0])
        
        return act
    
    def _gen_continuous_act(self, obs):
        act_vect = self.sess.run(self.out_op, feed_dict={self.in_op: obs})[0]
        
        # TODO: Add gaussian noise to action vector
        act_vect = [a + np.random.normal(0., 0.1) for a in act_vect]
        
        return np.array(act_vect)
        
    def gen_act(self, obs):
        if self.act_type == 'discrete':
            return self._gen_discrete_act(obs)
        else:
            return self._gen_continuous_act(obs)
        
    def train(self, obs, rewards, acts):
        raise RuntimeError('The train method was not properly created')
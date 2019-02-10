import gym
from cart_pole_cont import CartPoleEnv

def make_cart_pole():
    return gym.make('CartPole-v1')

def make_cart_pole_c():
    return CartPoleEnv()

def make_car_race():
    return gym.make('CarRacing-v0')

def make_lunar_lander_d():
    return gym.make('LunarLander-v2')

def make_lunar_lander_c():
    return gym.make('LunarLanderContinuous-v2')

import gymnasium as gym
import numpy as np
import random as rand
from gymnasium.spaces import Discrete, MultiDiscrete, Box
from gymnasium import Env

class empty_env(Env):

    def __init__(self):
        #define the observation and action spaces and the starting state of the enviroment
        pass

    def step(self, action):
        #Perform the action on the enviroment and calculate rewards
        pass

    def render(self):
        #render your enviroment if needed
        pass

    def reset(self):
        #reset the enviroment to the initilazation
        pass
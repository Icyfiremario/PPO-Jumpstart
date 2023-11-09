import torch
import gymnasium as gym

from ppo import PPO
from enviroment import empty_env

#Check for Nvidia GPU
if torch.cuda.is_available():
    DEVICE = torch.device("cuda:0")
else:
    DEVICE = torch.device("cpu")

#create the enviroment
#env = empty_env()  #custom env
env = gym.make("MountainCarContinuous-v0")

#init the model
model = PPO(env, DEVICE)

#start training loop for 2M timesteps
model.learn(2_000_000)
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
env = gym.make("Pendulum-v1")

#init the model
model = PPO(env, DEVICE)

#start training loop for 200M timesteps
model.learn(200_000_000)
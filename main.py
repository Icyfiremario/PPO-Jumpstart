from ppo import PPO
from enviroment import empty_env

#create the enviroment
env = empty_env()

#init the model
model = PPO(env)

#start training loop for 200M timesteps
model.learn(200_000_000)
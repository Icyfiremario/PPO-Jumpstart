import torch
import numpy as np
import gymnasium as gym

from ppo import PPO
from network import FeedForwardNN

if torch.cuda.is_available():
    DEVICE = torch.device("cuda:0")
else:
    DEVICE = torch.device("cpu")

env = gym.make("Pendulum-v1", render_mode='human')

obs_dim = env.observation_space.shape[0]
act_dim = env.action_space.shape[0]

policy = FeedForwardNN(obs_dim, act_dim)

policy.load_state_dict(torch.load("./ppo_actor.pth"))

policy.to(DEVICE)

while True:

    obs = env.reset()
    obs = obs[0]
    done = False

    t = 0

    ep_len = 0
    ep_ret = 0


    while not done:

        t += 1

        action = policy(obs).detach().numpy()

        obs, rew, done, _, _ = env.step(action)


    ep_len = t
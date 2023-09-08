import torch
import torch.nn as nn
import numpy as np
from torch.distributions import MultivariateNormal
from torch.optim import Adam

from network import FeedForwardNN

class PPO:

    def __init__(self, env, device):

        self._init_hyperparameters()

        #initalize class variables for the enviroment, Neural networks, optimizers, and covalence matrix
        self.env = env
        self.device = device

        #Get the input and output spaces
        self.obs_dim = env.observation_space.shape[0]
        self.act_dim = env.action_space.shape[0]

        self.actor = FeedForwardNN(self.obs_dim, self.act_dim)
        self.critic = FeedForwardNN(self.obs_dim, 1)

        self.actor.to(device)
        self.critic.to(device)

        #Define model optimizers
        self.actor_optim = Adam(self.actor.parameters(), lr=self.lr)
        self.critic_optim = Adam(self.critic.parameters(), lr=self.lr)

        #Create the covalence matrix for evaluation and action generation
        self.cov_var = torch.full(size=(self.act_dim,), fill_value=0.5).to(device)
        self.cov_mat = torch.diag(self.cov_var).to(device)

    def _init_hyperparameters(self):

        #Setup values used in training
        self.timesteps_per_batch = 4800
        self.max_timesteps_per_episode = 1600
        self.n_updates_per_iteration = 5
        self.gamma = 0.95
        self.clip = 0.2
        self.lr = 0.005

        self.save_freq = 10

    def rollout(self):

        #collect 1 batch of data

        batch_obs = []
        batch_acts = []
        batch_log_probs = []
        batch_rews = []
        batch_rtgs = []
        batch_lens = []

        ep_rews = []

        t = 0

        #start new batch
        while t < self.timesteps_per_batch:

            ep_rews = []

            #Start new episode
            obs = self.env.reset()
            obs = obs[0] #The enviroment thats currently in use returns a tuple when the reset function is called. Make sure to remove this if your enviroment doesn't return a tuple.
            done = False

            for ep_t in range(self.max_timesteps_per_episode):

                t += 1

                batch_obs.append(obs)

                #Get action to take from actor network and perform the action
                action, log_prob = self.get_action(obs)
                obs, rew, done, _, _ = self.env.step(action)

                ep_rews.append(rew)
                batch_acts.append(action)
                batch_log_probs.append(log_prob)

                if done:
                    break

            batch_lens.append(ep_t + 1)
            batch_rews.append(ep_rews)

        #convert batch data into tensors
        batch_obs = np.array(batch_obs)
        batch_acts = np.array(batch_acts)
        batch_log_probs = np.array(batch_log_probs)

        batch_obs = torch.tensor(batch_obs, dtype=torch.float).to(self.device)
        batch_acts = torch.tensor(batch_acts, dtype=torch.float).to(self.device)
        batch_log_probs = torch.tensor(batch_log_probs, dtype=torch.float).to(self.device)

        batch_rtgs = self.compute_rtgs(batch_rews) #Rewards to go. Takes the rewards and discounts them accordingly

        return batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens
    
    def compute_rtgs(self, batch_rews):

        batch_rtgs = []

        for ep_rews in reversed(batch_rews):

            discounted_reward = 0

            for rew in  reversed(ep_rews):

                discounted_reward = rew + discounted_reward * self.gamma
                batch_rtgs.insert(0, discounted_reward)
        
        batch_rtgs = torch.tensor(batch_rtgs, dtype=torch.float).to(self.device)

        return batch_rtgs
    
    def get_action(self, obs):

        mean = self.actor(obs)

        dist = MultivariateNormal(mean, self.cov_mat)

        action = dist.sample()

        log_prob = dist.log_prob(action)

        return action.detach().numpy(), log_prob.detach()
    
    def evaluate(self, batch_obs, batch_acts):

        V = self.critic(batch_obs).squeeze()

        mean = self.actor(batch_obs)
        dist = MultivariateNormal(mean, self.cov_mat)
        log_probs = dist.log_prob(batch_acts)

        return V, log_probs
    
    def learn(self, total_timesteps):

        print("Starting training")
        print(f"Training for {total_timesteps} timesteps")

        t_so_far = 0
        i_so_far = 0

        #start training loop
        while t_so_far < total_timesteps:

            print(f"Timesteps ran: {t_so_far}")

            batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens = self.rollout() #Get one batch of data

            t_so_far += np.sum(batch_lens)
            i_so_far += 1

            #Calculate ratios
            V, _ = self.evaluate(batch_obs, batch_acts)

            A_k = batch_rtgs - V.detach()

            A_k = (A_k - A_k.mean()) / (A_k.std() + 1e-10)

            for _ in range(self.n_updates_per_iteration):


                #Evaluate performance
                V, curr_log_probs = self.evaluate(batch_obs, batch_acts)

                ratios = torch.exp(curr_log_probs - batch_log_probs)

                surr1 = ratios * A_k
                surr2 = torch.clamp(ratios, 1 - self.clip, 1 + self.clip) * A_k


                #Calculate loss and adjust policy according to Evaluation return and ratios
                actor_loss = (-torch.min(surr1, surr2)).mean().to(self.device)
                critic_loss = nn.MSELoss()(V, batch_rtgs)

                self.actor_optim.zero_grad()
                actor_loss.backward(retain_graph=True)
                self.actor_optim.step()

                self.critic_optim.zero_grad()
                critic_loss.backward()
                self.critic_optim.step()

            if i_so_far % self.save_freq == 0:
                torch.save(self.actor.state_dict(), "./ppo_actor.pth")
                torch.save(self.critic.state_dict(), "./ppo_critic.pth")
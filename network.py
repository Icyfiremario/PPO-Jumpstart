import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gymnasium as gym

class FeedForwardNN(nn.Module):
    
    def __init__(self, in_dim, out_dim):

        super(FeedForwardNN, self).__init__()

        #Define the networks layers. Look at PyTorch's documentation for more info on the specific layers
        self.layer1 = nn.Linear(in_dim, 64)
        self.layer2 =nn.Linear(64, 64)
        self.layer3 = nn.Linear(64, out_dim)

    def forward(self, obs):

        #Make sure that input is a tensor
        if isinstance(obs, np.ndarray):
            obs = torch.tensor(obs, dtype=torch.float)

        #Perform linear transform activation
        activation1 = F.relu(self.layer1(obs))
        activation2 = F.relu(self.layer2(activation1))
        output = self.layer3(activation2)

        return output
import torch 
import torch.nn as nn 
import torch.optim as optim 
import torch.nn.functional as F 
import numpy as np
import random 
from collections import deque

L = 2
J = 3
state_dim = 4 ##this includes BER, the number of codewords we've built, the power of the signal required, the number of decoder steps
multiplier = 2
hidden_dim = state_dim*multiplier 

class Actor(nn.Module):  ##this will be our main policy network driving the codebook generation. 


    def __init__(self, state_dim,L, J): ##we shall define all of these dimensions above, allowing for a clean network class

        super().__init__()
        
        self.backbone = nn.Sequential(nn.Linear(state_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim), nn.ReLU()) ##all the heads will use this as their basis
        index_size = 2**J
        self.heads = nn.ModuelList([nn.Linear(hidden_dim, index_size) for _ in range(L)])
    

    def forward(self, state, tao):
      
        B = state.size()
        h = self.backbone(state)

        soft_samples = []
        hard_indices = []

        for head in self.heads:
            logits = head(h)  # Shape: (B, M)

            # Sample from Gumbel(0, 1) — enables differentiable categorical
            gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits) + 1e-8) + 1e-8)

            # Add Gumbel noise to logits and apply softmax with temperature
            y_soft = F.softmax((logits + gumbel_noise) / tao, dim=-1)  # Shape: (B, M)

            # Use argmax to get the hard index (non-differentiable)
            # This will be used to pick the actual message index
            index = y_soft.argmax(dim=-1)  # Shape: (B,)

            # Store soft and hard values
            soft_samples.append(y_soft)      # differentiable
            hard_indices.append(index)       # used in environment

        # Stack over sections: (B, L, M) and (B, L)
        soft_samples = torch.stack(soft_samples, dim=1)
        hard_indices = torch.stack(hard_indices, dim=1)

        return soft_samples, hard_indices
    


class Critic(nn.Module):


    def __init__(self, state, hidden_dim, J):

        
        self.fc1 = nn.Linear(state+J, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
        self.relu = nn.ReLU()


    def foward(self, state, action):


        input = torch.cat([state, action])

        x = self.relu(self.fc1(input))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)

        return x
    



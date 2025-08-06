##lets use this script to create an algorithm for producing the main RL algorithm for this codebook generation 

import torch 
import torch.nn as nn
import torch.nn.functional as F 
import numpy as np
from collections import deque 
import random 
import torch.optim as optim 



state_dim = 10 
action_dim = 10
hidden_dim = action_dim*2

capacity = 1000
batch_size = 32
temp = 0.32
gamma = 0.99
tau = 0.005  # soft update coefficient

num_episodes = 1000
lr = 0.0001

class CodewordPolicy(nn.Module):


    def __init__(self, state_dim, action_dim, hidden_dim):
        super().__init__()

        self.fc1 = nn.Linear(state_dim, hidden_dim) 
        self.fc_std = nn.Linear(hidden_dim , action_dim) #the log of the std. 
        self.relu = nn.ReLU()

    def forward(self, x):

        x = self.relu(self.fc1(x))

        log_std = self.fc_std(x) #this will output the log standard deviation for continuous action vectors

        std = torch.exp(log_std)

        # Create normal distribution with mean 0 and learned std
        dist = torch.distributions.Normal(torch.zeros_like(std), std)

        action = dist.sample()

        log_prob = dist.log_prob(action).sum(dim=-1)  # Sum over action dimensions

        return action, log_prob
    


class QNetwork(nn.Module):


    def __init__(self, state_dim, action_dim, hidden_dim):
        super().__init__()

        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)
        self.relu = nn.ReLU()

    def forward(self, state, action):

        input = torch.concat([state, action], dim=-1)

        x = self.relu(self.fc1(input))

        x = self.fc2(x)

        return x




class ReplayBuffer():


    def __init__(self,capacity, batch_size):


        self.buffer = deque(maxlen=capacity)
        self.batch_size = batch_size

    def push(self,state,action, next_state, reward, done):


        self.buffer.append((state, action, next_state, reward, done))

    def sample(self):

        batch = random.sample(self.buffer, self.batch_size)
        states, actions, next_states, rewards, dones = zip(*batch)

        return (torch.stack(states), 
                torch.stack(actions),
                torch.stack(next_states),
                torch.tensor(rewards, dtype=torch.float32), 
                torch.tensor(dones, dtype= torch.float32))
    


buffer = ReplayBuffer(capacity, batch_size)

matrix_policy = CodewordPolicy(state_dim, action_dim, hidden_dim)
q_1 = QNetwork(state_dim, action_dim, hidden_dim)
q_2 = QNetwork(state_dim, action_dim, hidden_dim)

tq_1 = QNetwork(state_dim, action_dim, hidden_dim)
tq_1.load_state_dict(q_1.state_dict())

tq_2 = QNetwork(state_dim, action_dim, hidden_dim)
tq_2.load_state_dict(q_2.state_dict())

criterion = nn.MSELoss()
    
optimizer1 = optim.Adam(params = q_1.parameters(), lr=lr)
optimizer2 = optim.Adam(params = q_2.parameters(), lr=lr)

optimizer3 = optim.Adam(params = matrix_policy.parameters(), lr=lr)

for episode in range(num_episodes):


    done = False
    log_probs = []
    states = []
    actions = []
    next_states = []
    rewards = []

    # state, info = env.reset()
    # state = torch.tensor(state, dtype=torch.float32)

    while not done: 

        action, log_prob = matrix_policy(state)

        # next_state, reward, terminated, truncated, info = env.step(action) #takes in the next column vector and determines our next states and rewards
        
        # Store experience in buffer
        # buffer.push(state, action, torch.tensor(next_state, dtype=torch.float32), reward, terminated or truncated)
        
        # state = torch.tensor(next_state, dtype=torch.float32)
        # done = terminated or truncated
    
    # Only train if buffer has enough samples
    if len(buffer.buffer) < batch_size:
        continue

    num_epochs = 10
    for epoch in range(num_epochs): 

        states, actions, next_states, rewards, dones = buffer.sample()
        
        # Compute target Q-values with no gradients
        with torch.no_grad():
            next_actions, next_log_prob = matrix_policy(next_states)
            
            q1_next = tq_1(next_states, next_actions)
            q2_next = tq_2(next_states, next_actions)
            
            # SAC uses entropy regularization in the target
            min_q_next = torch.min(q1_next, q2_next)
            target_value = rewards + gamma * (1-dones) * (min_q_next - temp * next_log_prob)

        # Update Q-networks
        loss1 = criterion(q_1(states, actions), target_value)

        optimizer1.zero_grad()
        loss1.backward()
        optimizer1.step()

        optimizer2.zero_grad()
        
        loss2 = criterion(q_2(states, actions), target_value)
        loss2.backward()
        optimizer2.step()

        # Update policy (actor) - detach Q-values to prevent gradients flowing through critics
        current_actions, current_log_prob = matrix_policy(states)
        
        q1_current = q_1(states, current_actions)
        q2_current = q_2(states, current_actions)
        min_q_current = torch.min(q1_current, q2_current)
        
        # SAC actor loss: maximize Q-value while regularizing with entropy
        actor_loss = torch.mean(temp * current_log_prob - min_q_current)
        
        optimizer3.zero_grad()
        actor_loss.backward()
        optimizer3.step()

        # Soft update target networks
        for target_param, param in zip(tq_1.parameters(), q_1.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
            
        for target_param, param in zip(tq_2.parameters(), q_2.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

        
        




        









        

          


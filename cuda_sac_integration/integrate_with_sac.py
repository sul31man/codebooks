#!/usr/bin/env python3
"""
This script shows exactly how to modify your SAC.py file to use the CUDA environment.
Copy the relevant parts from this file into your SAC.py.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F 
import numpy as np
from collections import deque 
import random 
import torch.optim as optim
import sys
import os

# Add the environments directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from environment_wrapper import CUDAEnvironment
##reminder to set devices to cuda so that we minimise data travel
# Your existing SAC classes (keep these as they are)
state_dim = 10 
action_dim = 1024  # Changed to match hardcoded CUDA dimensions: L*N=16*64=1024
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
        self.fc_std = nn.Linear(hidden_dim, action_dim) #the log of the std. 
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        
        log_std = self.fc_std(x)  # this will output the log standard deviation for continuous action vectors
        
        # Clamp log_std to prevent numerical instability
        log_std = torch.clamp(log_std, -20, 2)
        
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
    def __init__(self, capacity, batch_size, device):
        self.buffer = deque(maxlen=capacity)
        self.batch_size = batch_size
        self.device = device

    def push(self, state, action, next_state, reward, done):
        # Move tensors to CPU for storage to save GPU memory
        state_cpu = state.cpu() if isinstance(state, torch.Tensor) else state
        action_cpu = action.cpu() if isinstance(action, torch.Tensor) else action
        next_state_cpu = next_state.cpu() if isinstance(next_state, torch.Tensor) else next_state
        self.buffer.append((state_cpu, action_cpu, next_state_cpu, reward, done))

    def sample(self):
        batch = random.sample(self.buffer, self.batch_size)
        states, actions, next_states, rewards, dones = zip(*batch)
        return (torch.stack(states).to(self.device), 
                torch.stack(actions).to(self.device),
                torch.stack(next_states).to(self.device),
                torch.tensor(rewards, dtype=torch.float32, device=self.device), 
                torch.tensor(dones, dtype=torch.float32, device=self.device))

def main():
    """Main training loop with CUDA environment integration."""
    
    # Check for CUDA availability but use CPU for PyTorch to avoid conflicts with custom CUDA library
    device = torch.device("cpu")  # Force CPU to avoid CUDA context conflicts
    print(f"Using device: {device}")
    print("Note: Using CPU for PyTorch to avoid conflicts with custom CUDA environment")
    if torch.cuda.is_available():
        print(f"GPU available: {torch.cuda.get_device_name()}")
        print("Custom CUDA environment will still use GPU acceleration")
    
    # Initialize the CUDA environment with hardcoded dimensions
    print("Initializing CUDA environment...")
    # Dimensions are hardcoded in CUDA: n=512 rows, L*N=16*64=1024 cols
    env = CUDAEnvironment(Ka=1, num_sims=1000)
    
    # Initialize SAC components and move to GPU
    buffer = ReplayBuffer(capacity, batch_size, device)
    matrix_policy = CodewordPolicy(state_dim, action_dim, hidden_dim).to(device)
    q_1 = QNetwork(state_dim, action_dim, hidden_dim).to(device)
    q_2 = QNetwork(state_dim, action_dim, hidden_dim).to(device)
    
    tq_1 = QNetwork(state_dim, action_dim, hidden_dim).to(device)
    tq_1.load_state_dict(q_1.state_dict())
    
    tq_2 = QNetwork(state_dim, action_dim, hidden_dim).to(device)
    tq_2.load_state_dict(q_2.state_dict())
    
    criterion = nn.MSELoss()
    optimizer1 = optim.Adam(params=q_1.parameters(), lr=lr)
    optimizer2 = optim.Adam(params=q_2.parameters(), lr=lr)
    optimizer3 = optim.Adam(params=matrix_policy.parameters(), lr=lr)
    
    print("Starting SAC training with CUDA environment...")
    
    for episode in range(num_episodes):
        print(f"\n=== Episode {episode} ===")
        done = False
        log_probs = []
        states = []
        actions = []
        next_states = []
        rewards = []
        
        # Reset environment and get initial state
        print("  Resetting environment...")
        state = env.reset()
        state = torch.tensor(state, dtype=torch.float32, device=device)
        
        episode_reward = 0.0
        step_count = 0
        
        while not done:
            step_count += 1
            if step_count % 10 == 0:
                print(f"    Step {step_count}...")
            
            # Generate action using SAC policy (on GPU)
            action, log_prob = matrix_policy(state)
            
            # Take step in CUDA environment (convert action to CPU for environment)
            action_cpu = action.cpu() if action.is_cuda else action
            next_state, reward, done, info = env.step(action_cpu)
            next_state = torch.tensor(next_state, dtype=torch.float32, device=device)
            
            # Store experience in buffer (action stays on GPU for buffer)
            buffer.push(state, action, next_state, reward, done)
            
            episode_reward += reward
            state = next_state
            # done is already set from the environment step
        
        print(f"  Episode completed: {step_count} steps, reward={episode_reward:.4f}")
        
        # Only train if buffer has enough samples
        if len(buffer.buffer) < batch_size:
            print(f"  Skipping training - buffer size: {len(buffer.buffer)}/{batch_size}")
            continue
        
        print(f"  Starting training with buffer size: {len(buffer.buffer)}")
        
        # Training loop
        num_epochs = 10
        for epoch in range(num_epochs):
            if epoch % 5 == 0:
                print(f"    Training epoch {epoch}/{num_epochs}")
            
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
            
            # Update policy (actor)
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
        
        print(f"  Training completed for episode {episode}")
        
        # Print progress
        if episode % 10 == 0:  # Changed from 100 to 10 for more frequent updates
            print(f"*** Episode {episode}: Reward={episode_reward:.4f}, Best Hit Rate={env.best_hit_rate:.4f} ***")
    
    print("Training completed!")
    print(f"Final best hit rate: {env.best_hit_rate:.4f}")
    
    # Get final codebook
    final_codebook = env.get_codebook()
    print(f"Final codebook shape: {final_codebook.shape}")
    
    return env, matrix_policy

if __name__ == "__main__":
    try:
        env, policy = main()
        print("\nIntegration successful! Your SAC agent is now using the CUDA environment.")
    except Exception as e:
        print(f"Error during training: {e}")
        print("Please ensure CUDA is installed and the library can be compiled.") 
#!/usr/bin/env python3
"""
GPU-accelerated SAC with safe tensor pointer management.
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
import time

# Add the environments directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from environment_wrapper import CUDAEnvironment

# Configuration
state_dim = 10 
action_dim = 1024
hidden_dim = 256  # Smaller for speed
capacity = 1000
batch_size = 32
temp = 0.32
gamma = 0.99
tau = 0.005
num_episodes = 50  # Reduced for testing
lr = 0.001

class CodewordPolicy(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim) 
        self.fc_std = nn.Linear(hidden_dim, action_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        log_std = self.fc_std(x)
        
        # Clamp log_std to prevent numerical instability
        log_std = torch.clamp(log_std, -20, 2)
        std = torch.exp(log_std)
        
        # Create normal distribution with mean 0 and learned std
        dist = torch.distributions.Normal(torch.zeros_like(std), std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        
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
        # Store on CPU to save GPU memory and avoid pointer issues
        state_cpu = state.cpu() if isinstance(state, torch.Tensor) else state
        action_cpu = action.cpu() if isinstance(action, torch.Tensor) else action
        next_state_cpu = next_state.cpu() if isinstance(next_state, torch.Tensor) else next_state
        self.buffer.append((state_cpu, action_cpu, next_state_cpu, reward, done))

    def sample(self):
        batch = random.sample(self.buffer, self.batch_size)
        states, actions, next_states, rewards, dones = zip(*batch)
        
        # Move to device with proper error handling
        try:
            states_batch = torch.stack(states).to(self.device, dtype=torch.float32, non_blocking=True)
            actions_batch = torch.stack(actions).to(self.device, dtype=torch.float32, non_blocking=True)
            next_states_batch = torch.stack(next_states).to(self.device, dtype=torch.float32, non_blocking=True)
            rewards_batch = torch.tensor(rewards, dtype=torch.float32, device=self.device)
            dones_batch = torch.tensor(dones, dtype=torch.float32, device=self.device)
            
            return (states_batch, actions_batch, next_states_batch, rewards_batch, dones_batch)
        except Exception as e:
            print(f"Error in buffer sampling: {e}")
            raise

def safe_tensor_to_env(tensor):
    """Safely convert tensor for environment interaction."""
    if isinstance(tensor, torch.Tensor):
        # Ensure tensor is on CPU with proper dtype and contiguous
        if tensor.is_cuda:
            tensor = tensor.cpu()
        
        tensor = tensor.to(dtype=torch.float32)
        
        if not tensor.is_contiguous():
            tensor = tensor.contiguous()
        
        # Detach and copy to ensure memory safety
        tensor = tensor.detach().clone()
    
    return tensor

def main():
    """Main training loop with safe GPU-accelerated SAC."""
    
    # Device setup with fallback
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name()}")
        print("GPU Memory available:", torch.cuda.get_device_properties(0).total_memory // 1024**3, "GB")
    else:
        device = torch.device("cpu")
        print("CUDA not available, using CPU")
    
    # Initialize environment first (before CUDA context)
    print("Initializing CUDA environment...")
    env = CUDAEnvironment(Ka=1, num_sims=100)  # Reduced sims for speed
    print("Environment initialized successfully")
    
    # Initialize SAC components
    buffer = ReplayBuffer(capacity, batch_size, device)
    
    # Create networks on GPU
    matrix_policy = CodewordPolicy(state_dim, action_dim, hidden_dim).to(device)
    q_1 = QNetwork(state_dim, action_dim, hidden_dim).to(device)
    q_2 = QNetwork(state_dim, action_dim, hidden_dim).to(device)
    
    # Target networks
    tq_1 = QNetwork(state_dim, action_dim, hidden_dim).to(device)
    tq_1.load_state_dict(q_1.state_dict())
    
    tq_2 = QNetwork(state_dim, action_dim, hidden_dim).to(device)
    tq_2.load_state_dict(q_2.state_dict())
    
    # Optimizers
    criterion = nn.MSELoss()
    optimizer1 = optim.Adam(params=q_1.parameters(), lr=lr)
    optimizer2 = optim.Adam(params=q_2.parameters(), lr=lr)
    optimizer3 = optim.Adam(params=matrix_policy.parameters(), lr=lr)
    
    print(f"Networks created on {device}")
    print("Starting safe GPU-accelerated SAC training...")
    
    start_time = time.time()
    
    for episode in range(num_episodes):
        episode_start = time.time()
        
        if episode % 5 == 0:
            print(f"\n=== Episode {episode} ===")
        
        done = False
        
        # Reset environment and get initial state
        state = env.reset()
        state = torch.tensor(state, dtype=torch.float32, device=device)
        
        episode_reward = 0.0
        step_count = 0
        
        while not done:
            step_count += 1
            
            # Generate action using SAC policy (on GPU)
            with torch.no_grad():
                action, log_prob = matrix_policy(state)
            
            # Safe conversion for environment
            action_safe = safe_tensor_to_env(action)
            
            # Take step in environment
            try:
                next_state, reward, done, info = env.step(action_safe)
            except Exception as e:
                print(f"Environment step failed: {e}")
                print(f"Action tensor: shape={action.shape}, dtype={action.dtype}, device={action.device}")
                print(f"Action safe: shape={action_safe.shape}, dtype={action_safe.dtype}")
                raise
            
            # Convert next state to tensor on GPU
            next_state = torch.tensor(next_state, dtype=torch.float32, device=device)
            
            # Store experience in buffer
            buffer.push(state, action, next_state, reward, done)
            
            episode_reward += reward
            state = next_state
        
        if episode % 5 == 0:
            episode_time = time.time() - episode_start
            print(f"  Episode {episode}: {step_count} steps, reward={episode_reward:.4f}, time={episode_time:.2f}s")
        
        # Training
        if len(buffer.buffer) >= batch_size:
            num_epochs = 3  # Reduced for speed
            
            for epoch in range(num_epochs):
                try:
                    states, actions, next_states, rewards, dones = buffer.sample()
                    
                    # Compute target Q-values
                    with torch.no_grad():
                        next_actions, next_log_prob = matrix_policy(next_states)
                        q1_next = tq_1(next_states, next_actions)
                        q2_next = tq_2(next_states, next_actions)
                        
                        min_q_next = torch.min(q1_next, q2_next)
                        target_value = rewards.unsqueeze(1) + gamma * (1-dones.unsqueeze(1)) * (min_q_next - temp * next_log_prob.unsqueeze(1))
                    
                    # Update Q-networks
                    q1_pred = q_1(states, actions)
                    q2_pred = q_2(states, actions)
                    
                    loss1 = criterion(q1_pred, target_value.detach())
                    loss2 = criterion(q2_pred, target_value.detach())
                    
                    optimizer1.zero_grad()
                    loss1.backward()
                    optimizer1.step()
                    
                    optimizer2.zero_grad()
                    loss2.backward()
                    optimizer2.step()
                    
                    # Update policy
                    current_actions, current_log_prob = matrix_policy(states)
                    q1_current = q_1(states, current_actions)
                    q2_current = q_2(states, current_actions)
                    min_q_current = torch.min(q1_current, q2_current)
                    
                    actor_loss = torch.mean(temp * current_log_prob.unsqueeze(1) - min_q_current)
                    optimizer3.zero_grad()
                    actor_loss.backward()
                    optimizer3.step()
                    
                    # Soft update target networks
                    for target_param, param in zip(tq_1.parameters(), q_1.parameters()):
                        target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
                        
                    for target_param, param in zip(tq_2.parameters(), q_2.parameters()):
                        target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
                
                except Exception as e:
                    print(f"Training error at episode {episode}, epoch {epoch}: {e}")
                    raise
        
        # Progress report
        if episode % 5 == 0:
            elapsed_time = time.time() - start_time
            print(f"*** Episode {episode}: Reward={episode_reward:.4f}, Best Hit Rate={env.best_hit_rate:.4f}, Total Time={elapsed_time:.1f}s ***")
    
    total_time = time.time() - start_time
    print(f"\nTraining completed in {total_time:.1f}s!")
    print(f"Final best hit rate: {env.best_hit_rate:.4f}")
    
    # Get final codebook
    final_codebook = env.get_codebook()
    print(f"Final codebook shape: {final_codebook.shape}")
    
    return env, matrix_policy

if __name__ == "__main__":
    try:
        env, policy = main()
        print("\nSafe GPU-accelerated integration successful!")
    except Exception as e:
        print(f"Error during training: {e}")
        import traceback
        traceback.print_exc() 
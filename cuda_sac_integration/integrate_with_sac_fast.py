#!/usr/bin/env python3
"""
Fast hybrid SAC: PyTorch on CPU + Custom CUDA Environment
Optimized for speed while avoiding CUDA context conflicts.
"""

import torch
import torch.nn as nn
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

# Optimized Configuration
state_dim = 10 
action_dim = 1024
hidden_dim = 128  # Much smaller for speed
capacity = 500    # Smaller buffer
batch_size = 16   # Smaller batches for faster training
temp = 0.32
gamma = 0.99
tau = 0.01        # Faster soft updates
num_episodes = 30 # Fewer episodes for testing
lr = 0.003        # Higher learning rate

class CodewordPolicy(nn.Module):
    """Lightweight policy network optimized for CPU speed."""
    def __init__(self, state_dim, action_dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim) 
        self.fc_std = nn.Linear(hidden_dim, action_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        log_std = self.fc_std(x)
        
        # Clamp for stability
        log_std = torch.clamp(log_std, -10, 2)  # Less aggressive clamping
        std = torch.exp(log_std)
        
        # Normal distribution
        dist = torch.distributions.Normal(torch.zeros_like(std), std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        
        return action, log_prob

class QNetwork(nn.Module):
    """Lightweight Q-network optimized for CPU speed."""
    def __init__(self, state_dim, action_dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)
        self.relu = nn.ReLU()

    def forward(self, state, action):
        input = torch.cat([state, action], dim=-1)  # Use cat instead of concat for speed
        x = self.relu(self.fc1(input))
        return self.fc2(x)

class ReplayBuffer():
    """Optimized replay buffer for CPU operations."""
    def __init__(self, capacity, batch_size):
        self.buffer = deque(maxlen=capacity)
        self.batch_size = batch_size

    def push(self, state, action, next_state, reward, done):
        # Keep everything as numpy for speed
        if isinstance(state, torch.Tensor):
            state = state.detach().numpy()
        if isinstance(action, torch.Tensor):
            action = action.detach().numpy()
        if isinstance(next_state, torch.Tensor):
            next_state = next_state.detach().numpy()
            
        self.buffer.append((state, action, next_state, reward, done))

    def sample(self):
        batch = random.sample(self.buffer, self.batch_size)
        states, actions, next_states, rewards, dones = zip(*batch)
        
        # Convert to tensors efficiently
        return (torch.tensor(np.array(states), dtype=torch.float32),
                torch.tensor(np.array(actions), dtype=torch.float32),
                torch.tensor(np.array(next_states), dtype=torch.float32),
                torch.tensor(rewards, dtype=torch.float32),
                torch.tensor(dones, dtype=torch.float32))

def safe_tensor_to_env(tensor):
    """Ultra-safe tensor conversion for environment interaction."""
    if isinstance(tensor, torch.Tensor):
        # Ensure CPU, float32, contiguous
        tensor = tensor.detach().cpu().to(dtype=torch.float32)
        if not tensor.is_contiguous():
            tensor = tensor.contiguous()
        # Convert to numpy with copy for safety
        return tensor.numpy().copy()
    return tensor

def main():
    """Fast hybrid training: CPU PyTorch + GPU Environment."""
    
    # Force CPU for PyTorch (avoid CUDA context conflicts)
    device = torch.device("cpu")
    print("Using CPU for PyTorch (avoiding CUDA context conflicts)")
    print("Environment will still use GPU for CUDA kernels")
    
    # Initialize environment
    print("Initializing CUDA environment...")
    env = CUDAEnvironment(Ka=1, num_sims=50)  # Reduced for speed
    print("Environment initialized successfully")
    
    # Initialize fast SAC components
    buffer = ReplayBuffer(capacity, batch_size)
    
    # Create compact networks on CPU
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
    optimizer1 = optim.Adam(q_1.parameters(), lr=lr)
    optimizer2 = optim.Adam(q_2.parameters(), lr=lr)
    optimizer3 = optim.Adam(matrix_policy.parameters(), lr=lr)
    
    print(f"Networks created on {device}")
    print(f"Network sizes: Policy={sum(p.numel() for p in matrix_policy.parameters())}, Q={sum(p.numel() for p in q_1.parameters())}")
    print("Starting fast hybrid SAC training...")
    
    start_time = time.time()
    
    for episode in range(num_episodes):
        episode_start = time.time()
        
        if episode % 3 == 0:
            print(f"\n=== Episode {episode} ===")
        
        done = False
        
        # Reset environment
        state = env.reset()
        state = torch.tensor(state, dtype=torch.float32, device=device)
        
        episode_reward = 0.0
        step_count = 0
        
        while not done:
            step_count += 1
            
            # Generate action (fast CPU inference)
            with torch.no_grad():
                action, log_prob = matrix_policy(state)
            
            # Safe conversion for environment
            action_safe = safe_tensor_to_env(action)
            
            # Environment step (GPU CUDA kernels)
            try:
                next_state, reward, done, info = env.step(action_safe)
            except Exception as e:
                print(f"Environment step failed: {e}")
                print(f"Action safe shape: {action_safe.shape}, dtype: {action_safe.dtype}")
                raise
            
            # Convert next state (CPU tensor)
            next_state = torch.tensor(next_state, dtype=torch.float32, device=device)
            
            # Store experience
            buffer.push(state, action, next_state, reward, done)
            
            episode_reward += reward
            state = next_state
        
        if episode % 3 == 0:
            episode_time = time.time() - episode_start
            print(f"  Episode {episode}: {step_count} steps, reward={episode_reward:.4f}, time={episode_time:.2f}s")
        
        # Fast training
        if len(buffer.buffer) >= batch_size:
            # Fewer epochs for speed
            for epoch in range(2):
                try:
                    states, actions, next_states, rewards, dones = buffer.sample()
                    
                    # Efficient target computation
                    with torch.no_grad():
                        next_actions, next_log_prob = matrix_policy(next_states)
                        q1_next = tq_1(next_states, next_actions)
                        q2_next = tq_2(next_states, next_actions)
                        min_q_next = torch.min(q1_next, q2_next)
                        target_value = rewards.unsqueeze(1) + gamma * (1-dones.unsqueeze(1)) * (min_q_next - temp * next_log_prob.unsqueeze(1))
                    
                    # Fast Q-network updates
                    q1_pred = q_1(states, actions)
                    q2_pred = q_2(states, actions)
                    
                    loss1 = criterion(q1_pred, target_value)
                    loss2 = criterion(q2_pred, target_value)
                    
                    # Update Q1
                    optimizer1.zero_grad()
                    loss1.backward()
                    optimizer1.step()
                    
                    # Update Q2
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
                    
                    # Fast soft updates
                    for target_param, param in zip(tq_1.parameters(), q_1.parameters()):
                        target_param.data.mul_(1 - tau).add_(param.data, alpha=tau)
                        
                    for target_param, param in zip(tq_2.parameters(), q_2.parameters()):
                        target_param.data.mul_(1 - tau).add_(param.data, alpha=tau)
                
                except Exception as e:
                    print(f"Training error: {e}")
                    raise
        
        # Frequent progress reports
        if episode % 3 == 0:
            elapsed_time = time.time() - start_time
            avg_time = elapsed_time / (episode + 1)
            print(f"*** Episode {episode}: Reward={episode_reward:.4f}, Best Hit Rate={env.best_hit_rate:.4f} ***")
            print(f"*** Time: {episode_time:.2f}s/episode, Avg: {avg_time:.2f}s, Total: {elapsed_time:.1f}s ***")
    
    total_time = time.time() - start_time
    print(f"\nFast hybrid training completed in {total_time:.1f}s!")
    print(f"Average time per episode: {total_time/num_episodes:.2f}s")
    print(f"Final best hit rate: {env.best_hit_rate:.4f}")
    
    # Get final results
    final_codebook = env.get_codebook()
    print(f"Final codebook shape: {final_codebook.shape}")
    
    return env, matrix_policy

if __name__ == "__main__":
    try:
        env, policy = main()
        print("\nFast hybrid integration successful!")
        print("✅ CUDA environment ran on GPU")
        print("✅ PyTorch networks ran on CPU") 
        print("✅ No CUDA context conflicts")
    except Exception as e:
        print(f"Error during training: {e}")
        import traceback
        traceback.print_exc() 
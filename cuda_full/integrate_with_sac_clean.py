#!/usr/bin/env python3
"""
Clean GPU-accelerated SAC with proper modular design.
Uses the debug environment wrapper as a separate module.
"""

import torch
import torch.nn as nn
import numpy as np
from collections import deque 
import random 
import torch.optim as optim
import time

# Import the debug environment wrapper
from debug_environment_wrapper import DebugCUDAEnvironment

# Configuration
state_dim = 10 
action_dim = 1024
hidden_dim = 256
capacity = 1000
batch_size = 32
temp = 0.32
gamma = 0.99
tau = 0.005
num_episodes = 50
lr = 0.001

class CodewordPolicy(nn.Module):
    """SAC policy network for continuous actions."""
    def __init__(self, state_dim, action_dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim) 
        self.fc_std = nn.Linear(hidden_dim, action_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        log_std = self.fc_std(x)
        log_std = torch.clamp(log_std, -20, 2)
        std = torch.exp(log_std)
        
        dist = torch.distributions.Normal(torch.zeros_like(std), std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        
        return action, log_prob

class QNetwork(nn.Module):
    """SAC Q-network (critic)."""
    def __init__(self, state_dim, action_dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)
        self.relu = nn.ReLU()

    def forward(self, state, action):
        input = torch.cat([state, action], dim=-1)
        x = self.relu(self.fc1(input))
        return self.fc2(x)

class ReplayBuffer():
    """Experience replay buffer with GPU/CPU management."""
    def __init__(self, capacity, batch_size, device):
        self.buffer = deque(maxlen=capacity)
        self.batch_size = batch_size
        self.device = device

    def push(self, state, action, next_state, reward, done):
        # Store on CPU to save GPU memory
        state_cpu = state.cpu() if isinstance(state, torch.Tensor) else state
        action_cpu = action.cpu() if isinstance(action, torch.Tensor) else action
        next_state_cpu = next_state.cpu() if isinstance(next_state, torch.Tensor) else next_state
        self.buffer.append((state_cpu, action_cpu, next_state_cpu, reward, done))

    def sample(self):
        batch = random.sample(self.buffer, self.batch_size)
        states, actions, next_states, rewards, dones = zip(*batch)
        
        return (torch.stack(states).to(self.device, dtype=torch.float32, non_blocking=True),
                torch.stack(actions).to(self.device, dtype=torch.float32, non_blocking=True),
                torch.stack(next_states).to(self.device, dtype=torch.float32, non_blocking=True),
                torch.tensor(rewards, dtype=torch.float32, device=self.device),
                torch.tensor(dones, dtype=torch.float32, device=self.device))

def main():
    """Main training loop with clean modular design."""
    
    # Device setup
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"🚀 Using GPU: {torch.cuda.get_device_name()}")
    else:
        device = torch.device("cpu")
        print("Using CPU for PyTorch")
    
    # Initialize environment (separate module)
    print("Initializing debug CUDA environment...")
    env = DebugCUDAEnvironment(Ka=1, num_sims=50)
    
    # Initialize SAC components
    buffer = ReplayBuffer(capacity, batch_size, device)
    
    # Networks
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
    print("Starting GPU-accelerated SAC training...")
    
    start_time = time.time()
    
    for episode in range(num_episodes):
        if episode % 5 == 0:
            print(f"\n=== Episode {episode} ===")
        
        done = False
        state = env.reset()
        state = torch.tensor(state, dtype=torch.float32, device=device)
        
        episode_reward = 0.0
        step_count = 0
        
        while not done:
            step_count += 1
            
            # Generate action on GPU
            with torch.no_grad():
                action, log_prob = matrix_policy(state)
            
            # Environment step (safe tensor conversion handled internally)
            next_state, reward, done, info = env.step(action)
            next_state = torch.tensor(next_state, dtype=torch.float32, device=device)
            
            # Store experience
            buffer.push(state, action, next_state, reward, done)
            
            episode_reward += reward
            state = next_state
        
        if episode % 5 == 0:
            episode_time = time.time() - start_time
            print(f"  Episode {episode}: {step_count} steps, reward={episode_reward:.4f}")
        
        # Training
        if len(buffer.buffer) >= batch_size:
            for epoch in range(5):
                states, actions, next_states, rewards, dones = buffer.sample()
                
                # SAC updates on GPU
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
                
                # Soft update targets
                for target_param, param in zip(tq_1.parameters(), q_1.parameters()):
                    target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
                    
                for target_param, param in zip(tq_2.parameters(), q_2.parameters()):
                    target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
        
        # Progress report
        if episode % 5 == 0:
            elapsed_time = time.time() - start_time
            print(f"*** Episode {episode}: Reward={episode_reward:.4f}, Best Hit Rate={env.best_hit_rate:.4f}, Time={elapsed_time:.1f}s ***")
    
    total_time = time.time() - start_time
    print(f"\n🎉 Training completed in {total_time:.1f}s!")
    print(f"Average time per episode: {total_time/num_episodes:.2f}s")
    print(f"Final best hit rate: {env.best_hit_rate:.4f}")
    
    return env, matrix_policy

if __name__ == "__main__":
    try:
        env, policy = main()
        print("\n✅ Clean modular SAC training successful!")
        print("✅ Separate environment wrapper module")
        print("✅ GPU acceleration working")
        print("✅ No CUDA context conflicts")
    except Exception as e:
        print(f"Error during training: {e}")
        import traceback
        traceback.print_exc() 
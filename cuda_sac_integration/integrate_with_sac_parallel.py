#!/usr/bin/env python3
"""
GPU-accelerated SAC with parallel Ka value processing.
This version processes multiple Ka values simultaneously for better performance.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np
import time
from parallel_environment_wrapper import ParallelCUDAEnvironment

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class CodewordPolicy(nn.Module):
    """Actor network for SAC."""
    def __init__(self, state_dim, action_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)
        
        # Move to device
        self.to(device)
        
    def forward(self, state):
        x = self.net(state)
        mean = self.mean(x)
        log_std = self.log_std(x)
        log_std = torch.clamp(log_std, -20, 2)
        return mean, log_std
    
    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        x = normal.rsample()
        action = torch.tanh(x)
        log_prob = normal.log_prob(x)
        log_prob -= torch.log(1 - action.pow(2) + 1e-6)
        return action, log_prob.sum(1, keepdim=True)

class QNetwork(nn.Module):
    """Critic network for SAC."""
    def __init__(self, state_dim, action_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Move to device
        self.to(device)
        
    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        return self.net(x)

class ReplayBuffer:
    """Experience replay buffer."""
    def __init__(self, capacity, state_dim, action_dim, device):
        self.capacity = capacity
        self.device = device
        
        # Store on CPU, only transfer to GPU when sampling
        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.dones = np.zeros((capacity, 1), dtype=np.float32)
        
        self.size = 0
        self.position = 0
        
    def push(self, state, action, reward, next_state, done):
        self.states[self.position] = state
        self.actions[self.position] = action
        self.rewards[self.position] = reward
        self.next_states[self.position] = next_state
        self.dones[self.position] = done
        
        self.size = min(self.size + 1, self.capacity)
        self.position = (self.position + 1) % self.capacity
        
    def sample(self, batch_size):
        indices = np.random.randint(0, self.size, size=batch_size)
        
        # Convert to tensors and move to device
        return (
            torch.FloatTensor(self.states[indices]).to(self.device),
            torch.FloatTensor(self.actions[indices]).to(self.device),
            torch.FloatTensor(self.rewards[indices]).to(self.device),
            torch.FloatTensor(self.next_states[indices]).to(self.device),
            torch.FloatTensor(self.dones[indices]).to(self.device)
        )

def main():
    """Main training loop."""
    print("🚀 Starting parallel SAC training...")
    
    # Hyperparameters
    state_dim = 10
    action_dim = 1024
    hidden_dim = 256
    buffer_capacity = 100000
    batch_size = 32
    num_episodes = 50
    num_epochs = 5
    alpha = 0.2  # Temperature parameter
    lr = 3e-4
    tau = 0.005  # Soft update parameter
    
    # Initialize environment with multiple Ka values
    env = ParallelCUDAEnvironment(
        Ka_values=[5, 12, 20, 27, 35],  # Process 5 Ka values in parallel
        num_sims=100
    )
    
    # Initialize networks
    policy = CodewordPolicy(state_dim, action_dim, hidden_dim)
    q1 = QNetwork(state_dim, action_dim, hidden_dim)
    q2 = QNetwork(state_dim, action_dim, hidden_dim)
    q1_target = QNetwork(state_dim, action_dim, hidden_dim)
    q2_target = QNetwork(state_dim, action_dim, hidden_dim)
    
    # Copy parameters
    q1_target.load_state_dict(q1.state_dict())
    q2_target.load_state_dict(q2.state_dict())
    
    # Initialize optimizers
    policy_optimizer = optim.Adam(policy.parameters(), lr=lr)
    q1_optimizer = optim.Adam(q1.parameters(), lr=lr)
    q2_optimizer = optim.Adam(q2.parameters(), lr=lr)
    
    # Initialize replay buffer
    replay_buffer = ReplayBuffer(buffer_capacity, state_dim, action_dim, device)
    
    # Training loop
    total_steps = 0
    best_reward = float('-inf')
    training_start = time.time()
    
    print("\n📊 Training Progress:")
    print("Episode | Steps | Avg Reward | Best Reward | Time/Step (ms)")
    print("-" * 58)
    
    for episode in range(num_episodes):
        episode_start = time.time()
        state = env.reset()
        episode_reward = 0
        steps = 0
        
        while True:
            # Convert state to tensor
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            
            # Get action from policy
            with torch.no_grad():
                action, _ = policy.sample(state_tensor)
                action = action.squeeze(0)
            
            # Take step in environment
            next_state, reward, done, info = env.step(action)
            
            # Store in replay buffer
            replay_buffer.push(state, action.cpu().numpy(), reward, next_state, done)
            
            # Update statistics
            total_steps += 1
            steps += 1
            episode_reward += reward
            
            # Training (if enough samples)
            if replay_buffer.size > batch_size:
                for _ in range(num_epochs):
                    # Sample from replay buffer
                    states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)
                    
                    # Update Q-functions
                    with torch.no_grad():
                        next_actions, next_log_probs = policy.sample(next_states)
                        q1_next = q1_target(next_states, next_actions)
                        q2_next = q2_target(next_states, next_actions)
                        q_next = torch.min(q1_next, q2_next) - alpha * next_log_probs
                        q_target = rewards + (1 - dones) * 0.99 * q_next
                    
                    # Q1 update
                    q1_loss = F.mse_loss(q1(states, actions), q_target)
                    q1_optimizer.zero_grad()
                    q1_loss.backward()
                    q1_optimizer.step()
                    
                    # Q2 update
                    q2_loss = F.mse_loss(q2(states, actions), q_target)
                    q2_optimizer.zero_grad()
                    q2_loss.backward()
                    q2_optimizer.step()
                    
                    # Policy update
                    new_actions, log_probs = policy.sample(states)
                    q1_new = q1(states, new_actions)
                    q2_new = q2(states, new_actions)
                    q_new = torch.min(q1_new, q2_new)
                    
                    policy_loss = (alpha * log_probs - q_new).mean()
                    policy_optimizer.zero_grad()
                    policy_loss.backward()
                    policy_optimizer.step()
                    
                    # Soft update of target networks
                    for target_param, param in zip(q1_target.parameters(), q1.parameters()):
                        target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
                    for target_param, param in zip(q2_target.parameters(), q2.parameters()):
                        target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
            
            # Update state
            state = next_state
            
            # Check if episode is done
            if done:
                break
        
        # Episode statistics
        episode_time = time.time() - episode_start
        steps_per_second = steps / episode_time
        ms_per_step = 1000 / steps_per_second
        
        # Update best reward
        best_reward = max(best_reward, episode_reward)
        
        # Progress update
        print(f"{episode:7d} | {steps:5d} | {episode_reward:10.4f} | {best_reward:11.4f} | {ms_per_step:13.2f}")
        
        # Detailed info every 10 episodes
        if (episode + 1) % 10 == 0:
            print(f"\n📈 Detailed Stats (Episode {episode + 1}):")
            print(f"Hit rates for Ka values {env.Ka_values}:")
            print(info['hit_rates'])
            print(f"Average hit rate: {info['hit_rate']:.4f}")
            print(f"Best hit rate: {info['best_hit_rate']:.4f}")
            print("-" * 58)
    
    # Training complete
    total_time = time.time() - training_start
    print(f"\n🎉 Parallel training completed in {total_time:.1f}s!")
    print(f"Average time per episode: {total_time/num_episodes:.2f}s")
    print(f"Final best reward: {best_reward:.4f}")
    
    # Save final codebook
    final_codebook = env.get_codebook()
    np.save("final_codebook_parallel.npy", final_codebook)
    print("\n💾 Final codebook saved to 'final_codebook_parallel.npy'")

if __name__ == "__main__":
    main() 
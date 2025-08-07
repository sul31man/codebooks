#!/usr/bin/env python3
"""
Soft Actor-Critic (SAC) Algorithm Components
Optimized for CUDA codebook generation environment.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np

class CodewordPolicy(nn.Module):
    """Actor network for SAC - generates codebook actions."""
    
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
        
    def forward(self, state):
        x = self.net(state)
        mean = self.mean(x)
        log_std = self.log_std(x)
        log_std = torch.clamp(log_std, -20, 2)  # Stability
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
    """Critic network for SAC - evaluates state-action pairs."""
    
    def __init__(self, state_dim, action_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        return self.net(x)

class ReplayBuffer:
    """Experience replay buffer for SAC."""
    
    def __init__(self, capacity, state_dim, action_dim, device):
        self.capacity = capacity
        self.device = device
        
        # Store on CPU, transfer to GPU when sampling (memory efficient)
        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.dones = np.zeros((capacity, 1), dtype=np.float32)
        
        self.size = 0
        self.position = 0
        
    def push(self, state, action, reward, next_state, done):
        """Add experience to buffer."""
        self.states[self.position] = state
        self.actions[self.position] = action
        self.rewards[self.position] = reward
        self.next_states[self.position] = next_state
        self.dones[self.position] = done
        
        self.size = min(self.size + 1, self.capacity)
        self.position = (self.position + 1) % self.capacity
        
    def sample(self, batch_size):
        """Sample batch of experiences."""
        if self.size < batch_size:
            return None
            
        indices = np.random.randint(0, self.size, size=batch_size)
        
        # Convert to tensors and move to device
        return (
            torch.FloatTensor(self.states[indices]).to(self.device),
            torch.FloatTensor(self.actions[indices]).to(self.device),
            torch.FloatTensor(self.rewards[indices]).to(self.device),
            torch.FloatTensor(self.next_states[indices]).to(self.device),
            torch.FloatTensor(self.dones[indices]).to(self.device)
        )

class SACAgent:
    """Complete SAC agent for codebook optimization."""
    
    def __init__(self, state_dim, action_dim, hidden_dim=256, lr=3e-4, 
                 alpha=0.2, tau=0.005, gamma=0.99, device=None):
        """
        Initialize SAC agent.
        
        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space (codebook size)
            hidden_dim: Hidden layer size
            lr: Learning rate
            alpha: Temperature parameter for entropy
            tau: Soft update parameter
            gamma: Discount factor
            device: Device to run on (CPU/CUDA)
        """
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.alpha = alpha
        self.tau = tau
        self.gamma = gamma
        
        # Initialize networks
        self.policy = CodewordPolicy(state_dim, action_dim, hidden_dim).to(self.device)
        self.q1 = QNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.q2 = QNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.q1_target = QNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.q2_target = QNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        
        # Copy parameters to target networks
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())
        
        # Initialize optimizers
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.q1_optimizer = optim.Adam(self.q1.parameters(), lr=lr)
        self.q2_optimizer = optim.Adam(self.q2.parameters(), lr=lr)
        
        print(f"🤖 SAC Agent initialized on {self.device}")
        print(f"📊 State dim: {state_dim}, Action dim: {action_dim}")
        print(f"🧠 Hidden dim: {hidden_dim}, Learning rate: {lr}")
    
    def select_action(self, state, evaluate=False):
        """Select action using policy."""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            if evaluate:
                # Deterministic action for evaluation
                mean, _ = self.policy(state_tensor)
                action = torch.tanh(mean)
            else:
                # Stochastic action for exploration
                action, _ = self.policy.sample(state_tensor)
        
        return action.squeeze(0)
    
    def update(self, replay_buffer, batch_size=64, updates=1):
        """Update SAC agent."""
        if replay_buffer.size < batch_size:
            return {}
        
        total_policy_loss = 0
        total_q1_loss = 0
        total_q2_loss = 0
        
        for _ in range(updates):
            # Sample batch
            batch = replay_buffer.sample(batch_size)
            if batch is None:
                continue
                
            states, actions, rewards, next_states, dones = batch
            
            # Update Q-functions
            with torch.no_grad():
                next_actions, next_log_probs = self.policy.sample(next_states)
                q1_next = self.q1_target(next_states, next_actions)
                q2_next = self.q2_target(next_states, next_actions)
                q_next = torch.min(q1_next, q2_next) - self.alpha * next_log_probs
                q_target = rewards + (1 - dones) * self.gamma * q_next
            
            # Q1 update
            q1_loss = F.mse_loss(self.q1(states, actions), q_target)
            self.q1_optimizer.zero_grad()
            q1_loss.backward()
            self.q1_optimizer.step()
            
            # Q2 update
            q2_loss = F.mse_loss(self.q2(states, actions), q_target)
            self.q2_optimizer.zero_grad()
            q2_loss.backward()
            self.q2_optimizer.step()
            
            # Policy update
            new_actions, log_probs = self.policy.sample(states)
            q1_new = self.q1(states, new_actions)
            q2_new = self.q2(states, new_actions)
            q_new = torch.min(q1_new, q2_new)
            
            policy_loss = (self.alpha * log_probs - q_new).mean()
            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            self.policy_optimizer.step()
            
            # Soft update of target networks
            self._soft_update(self.q1_target, self.q1)
            self._soft_update(self.q2_target, self.q2)
            
            # Accumulate losses
            total_policy_loss += policy_loss.item()
            total_q1_loss += q1_loss.item()
            total_q2_loss += q2_loss.item()
        
        return {
            'policy_loss': total_policy_loss / updates,
            'q1_loss': total_q1_loss / updates,
            'q2_loss': total_q2_loss / updates
        }
    
    def _soft_update(self, target, source):
        """Soft update target network."""
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - self.tau) + param.data * self.tau
            )
    
    def save(self, filepath):
        """Save agent state."""
        torch.save({
            'policy': self.policy.state_dict(),
            'q1': self.q1.state_dict(),
            'q2': self.q2.state_dict(),
            'q1_target': self.q1_target.state_dict(),
            'q2_target': self.q2_target.state_dict(),
            'policy_optimizer': self.policy_optimizer.state_dict(),
            'q1_optimizer': self.q1_optimizer.state_dict(),
            'q2_optimizer': self.q2_optimizer.state_dict(),
        }, filepath)
    
    def load(self, filepath):
        """Load agent state."""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.policy.load_state_dict(checkpoint['policy'])
        self.q1.load_state_dict(checkpoint['q1'])
        self.q2.load_state_dict(checkpoint['q2'])
        self.q1_target.load_state_dict(checkpoint['q1_target'])
        self.q2_target.load_state_dict(checkpoint['q2_target'])
        self.policy_optimizer.load_state_dict(checkpoint['policy_optimizer'])
        self.q1_optimizer.load_state_dict(checkpoint['q1_optimizer'])
        self.q2_optimizer.load_state_dict(checkpoint['q2_optimizer']) 
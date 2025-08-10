import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import env2

class PolicyNetwork(nn.Module):
    """Simple policy network for REINFORCE"""
    def __init__(self, obs_dim, action_dim, hidden_dim=64):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(obs_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        logits = self.fc3(x)
        return F.softmax(logits, dim=-1)

class REINFORCE:
    """REINFORCE algorithm with batched CUDA environments"""
    def __init__(self, obs_dim, action_dim, batch_size=32, lr=1e-3, gamma=0.99):
        self.batch_size = batch_size
        self.action_dim = action_dim
        self.obs_dim = obs_dim
        self.gamma = gamma
        
        # Policy network
        self.policy = PolicyNetwork(obs_dim, action_dim).cuda()
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        
        # Episode storage
        self.reset_episode_data()
    
    def reset_episode_data(self):
        """Reset episode storage"""
        self.episode_states = []
        self.episode_actions = []
        self.episode_rewards = []
        self.episode_log_probs = []
    
    def select_action(self, states):
        """Select actions using policy network"""
        with torch.no_grad():
            action_probs = self.policy(states)
            
        # Sample actions from probability distribution
        dist = torch.distributions.Categorical(action_probs)
        actions = dist.sample()
        log_probs = dist.log_prob(actions)
        
        return actions, log_probs
    
    def compute_returns(self, rewards):
        """Compute discounted returns"""
        returns = []
        R = 0
        for r in reversed(rewards):
            R = r + self.gamma * R
            returns.insert(0, R)
        return torch.tensor(returns, device='cuda', dtype=torch.float32)
    
    def update_policy(self):
        """Update policy using REINFORCE"""
        if len(self.episode_rewards) == 0:
            return 0.0
            
        # Compute returns for each timestep
        returns = self.compute_returns(self.episode_rewards)
        
        # Normalize returns
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        # Compute policy loss
        policy_loss = []
        for log_prob, R in zip(self.episode_log_probs, returns):
            policy_loss.append(-log_prob.mean() * R)  # REINFORCE loss
        
        # Update policy
        self.optimizer.zero_grad()
        loss = torch.stack(policy_loss).sum()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def train_episode(self, max_steps=100):
        """Train one episode using batched environments"""
        self.reset_episode_data()
        
        # Initialize random states (normally you'd reset environment)
        states = torch.randn(self.batch_size, self.obs_dim).cuda()
        total_reward = 0
        
        for step in range(max_steps):
            # Select actions
            actions, log_probs = self.select_action(states)
            
            # Convert to integer actions for environment
            env_actions = actions.unsqueeze(1).repeat(1, self.action_dim)
            
            # Environment step
            next_states, rewards, dones = env2.batched_step(env_actions)
            
            # Store episode data
            self.episode_states.append(states)
            self.episode_actions.append(actions)
            self.episode_rewards.append(rewards.mean())  # Average reward across batch
            self.episode_log_probs.append(log_probs)
            
            total_reward += rewards.mean().item()
            
            # Update states
            states = next_states
            
            # Check if any environments are done (simple termination)
            if dones.any() or step == max_steps - 1:
                break
        
        # Update policy
        loss = self.update_policy()
        return total_reward, loss, step + 1

def main():
    """Main training loop"""
    # Environment parameters
    batch_size = 16
    obs_dim = 3  # Same as action_dim in your environment
    action_dim = 5  # Number of discrete actions
    
    # Training parameters
    num_episodes = 100
    max_steps = 50
    
    # Initialize REINFORCE agent
    agent = REINFORCE(obs_dim, action_dim, batch_size)
    
    print("=== REINFORCE Training with Batched CUDA Environments ===")
    print(f"Batch size: {batch_size}")
    print(f"Obs dim: {obs_dim}, Action dim: {action_dim}")
    print(f"Training for {num_episodes} episodes...")
    print()
    
    # Training loop
    for episode in range(num_episodes):
        total_reward, loss, steps = agent.train_episode(max_steps)
        
        if episode % 10 == 0:
            print(f"Episode {episode:3d} | "
                  f"Reward: {total_reward:7.2f} | "
                  f"Loss: {loss:8.4f} | "
                  f"Steps: {steps:2d}")
    
    print("\nTraining completed!")
    
    # Test the trained policy
    print("\n=== Testing Trained Policy ===")
    with torch.no_grad():
        test_states = torch.randn(batch_size, obs_dim).cuda()
        actions, _ = agent.select_action(test_states)
        env_actions = actions.unsqueeze(1).repeat(1, obs_dim)
        
        obs, rewards, dones = env2.batched_step(env_actions)
        
        print(f"Test states shape: {test_states.shape}")
        print(f"Selected actions: {actions}")
        print(f"Rewards: {rewards}")
        print(f"Average reward: {rewards.mean():.3f}")

if __name__ == "__main__":
    main()

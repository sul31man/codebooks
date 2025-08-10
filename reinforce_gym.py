import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import gymnasium as gym

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

class VectorizedEnv:
    """Wrapper to vectorize gym environments for batched training"""
    def __init__(self, env_name, num_envs):
        self.num_envs = num_envs
        self.envs = [gym.make(env_name) for _ in range(num_envs)]
        
        # Get environment specs from first env
        sample_env = self.envs[0]
        self.obs_dim = sample_env.observation_space.shape[0]
        self.action_dim = sample_env.action_space.n
        
    def reset(self):
        """Reset all environments"""
        observations = []
        for env in self.envs:
            obs, _ = env.reset()
            observations.append(obs)
        return torch.tensor(np.array(observations), dtype=torch.float32)
    
    def step(self, actions):
        """Step all environments with given actions"""
        observations = []
        rewards = []
        dones = []
        
        for i, env in enumerate(self.envs):
            obs, reward, terminated, truncated, _ = env.step(actions[i].item())
            observations.append(obs)
            rewards.append(reward)
            dones.append(terminated or truncated)
        
        return (
            torch.tensor(np.array(observations), dtype=torch.float32),
            torch.tensor(rewards, dtype=torch.float32),
            torch.tensor(dones, dtype=torch.bool)
        )
    
    def close(self):
        """Close all environments"""
        for env in self.envs:
            env.close()

class REINFORCE:
    """REINFORCE algorithm with vectorized Gym environments"""
    def __init__(self, obs_dim, action_dim, batch_size=32, lr=1e-3, gamma=0.99, device='cpu'):
        self.batch_size = batch_size
        self.action_dim = action_dim
        self.obs_dim = obs_dim
        self.gamma = gamma
        self.device = device
        
        # Policy network
        self.policy = PolicyNetwork(obs_dim, action_dim).to(device)
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
        states = states.to(self.device)
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
        return torch.tensor(returns, device=self.device, dtype=torch.float32)
    
    def update_policy(self):
        """Update policy using REINFORCE"""
        if len(self.episode_rewards) == 0:
            return 0.0
            
        # Compute returns for each timestep
        returns = self.compute_returns(self.episode_rewards)
        
        # Normalize returns
        if len(returns) > 1:
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
    
    def train_episode(self, env, max_steps=500):
        """Train one episode using vectorized environments"""
        self.reset_episode_data()
        
        # Reset environments
        states = env.reset()
        total_reward = 0
        
        for step in range(max_steps):
            # Select actions
            actions, log_probs = self.select_action(states)
            
            # Environment step
            next_states, rewards, dones = env.step(actions)
            
            # Store episode data
            self.episode_states.append(states)
            self.episode_actions.append(actions)
            self.episode_rewards.append(rewards.mean())  # Average reward across batch
            self.episode_log_probs.append(log_probs)
            
            total_reward += rewards.mean().item()
            
            # Update states
            states = next_states
            
            # Check if any environments are done
            if dones.any() or step == max_steps - 1:
                break
        
        # Update policy
        loss = self.update_policy()
        return total_reward, loss, step + 1

def main():
    """Main training loop"""
    # Environment parameters
    env_name = "CartPole-v1"  # Classic control environment
    batch_size = 8
    
    # Create vectorized environment
    env = VectorizedEnv(env_name, batch_size)
    obs_dim = env.obs_dim
    action_dim = env.action_dim
    
    # Training parameters
    num_episodes = 200
    max_steps = 500
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Initialize REINFORCE agent
    agent = REINFORCE(obs_dim, action_dim, batch_size, device=device)
    
    print("=== REINFORCE Training with Gym Environments ===")
    print(f"Environment: {env_name}")
    print(f"Device: {device}")
    print(f"Batch size: {batch_size}")
    print(f"Obs dim: {obs_dim}, Action dim: {action_dim}")
    print(f"Training for {num_episodes} episodes...")
    print()
    
    # Training loop
    best_reward = float('-inf')
    
    for episode in range(num_episodes):
        total_reward, loss, steps = agent.train_episode(env, max_steps)
        
        if total_reward > best_reward:
            best_reward = total_reward
        
        if episode % 20 == 0:
            print(f"Episode {episode:3d} | "
                  f"Reward: {total_reward:7.2f} | "
                  f"Best: {best_reward:7.2f} | "
                  f"Loss: {loss:8.4f} | "
                  f"Steps: {steps:3d}")
    
    print(f"\nTraining completed! Best reward: {best_reward:.2f}")
    
    # Test the trained policy
    print("\n=== Testing Trained Policy ===")
    test_episodes = 5
    test_rewards = []
    
    for test_ep in range(test_episodes):
        states = env.reset()
        episode_reward = 0
        
        for step in range(max_steps):
            with torch.no_grad():
                actions, _ = agent.select_action(states)
            
            states, rewards, dones = env.step(actions)
            episode_reward += rewards.mean().item()
            
            if dones.any():
                break
        
        test_rewards.append(episode_reward)
        print(f"Test episode {test_ep + 1}: {episode_reward:.2f}")
    
    print(f"\nAverage test reward: {np.mean(test_rewards):.2f} ± {np.std(test_rewards):.2f}")
    
    # Close environments
    env.close()

if __name__ == "__main__":
    main()

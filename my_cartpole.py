import torch 
import env2 
import torch.nn as nn 
import torch.nn.functional as F 
import torch.optim as optim

state_dim = 4
action_dim = 2
hidden_dim = 8
batch_size = 10  # Number of environments to run in parallel

class Policy(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(Policy, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, action_dim)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = F.softmax(self.fc2(x), dim=-1)
        return x
    
    def select_action_batch(self, states_batch):
        """Select actions for ALL environments at once (BATCHED!)"""
        # states_batch: [batch_size, 4] - process ALL environments simultaneously!
        probs = self.forward(states_batch)  # [batch_size, 2]
        
        # Sample ALL actions at once (vectorized!)
        dist = torch.distributions.Categorical(probs)
        actions = dist.sample()  # [batch_size] - all actions sampled simultaneously
        log_probs = dist.log_prob(actions)  # [batch_size] - all log probs at once
        
        return actions, log_probs

policy = Policy(state_dim, hidden_dim, action_dim).cuda()  # Move to GPU
optimizer = optim.Adam(policy.parameters(), lr=0.001)

num_episodes = 1000

for episode in range(num_episodes):
    # BATCHED RESET: Initialize 4 environments simultaneously
    states_batch = env2.env_reset(batch_size)  # [4, 4] tensor - 4 environments, 4 state variables each
    print(f"Episode {episode + 1}: Initial states shape: {states_batch.shape}")
    
    # Episode data storage for each environment
    episode_states = [[] for _ in range(batch_size)]
    episode_actions = [[] for _ in range(batch_size)]
    episode_log_probs = [[] for _ in range(batch_size)]
    episode_rewards = [[] for _ in range(batch_size)]
    
    # Track which environments are still active
    active_environments = torch.ones(batch_size, dtype=torch.bool).cuda()
    
    step = 0
    while active_environments.any():  # Continue while any environment is active
        step += 1
        
        # BATCHED ACTION SELECTION: Process ALL environments simultaneously!
        actions_batch, log_probs_batch = policy.select_action_batch(states_batch)
        
        # BATCHED ENVIRONMENT STEP: Simulate ALL environments simultaneously!
        next_states_batch, rewards_batch, dones_batch = env2.env_step(states_batch, actions_batch.float())
        
        # Store data for each environment separately
        for env_idx in range(batch_size):
            if active_environments[env_idx]:
                episode_states[env_idx].append(states_batch[env_idx:env_idx+1])
                episode_actions[env_idx].append(actions_batch[env_idx:env_idx+1])
                episode_log_probs[env_idx].append(log_probs_batch[env_idx:env_idx+1])
                episode_rewards[env_idx].append(rewards_batch[env_idx:env_idx+1])
                
                # Mark environment as done if episode terminated
                if dones_batch[env_idx]:
                    active_environments[env_idx] = False
        
        # Update states for next step
        states_batch = next_states_batch
        
        if step < 5:  # Show first few steps
            print(f"Step {step}: Actions={actions_batch}, Rewards={rewards_batch}, Dones={dones_batch}")
            print(f"Active envs: {active_environments.sum().item()}/{batch_size}")
        
        # Safety check to prevent infinite loops
        if step > 200:
            print("Max steps reached, ending episode")
            break
    
    # Compute episode statistics
    total_rewards = []
    for env_idx in range(batch_size):
        if len(episode_rewards[env_idx]) > 0:
            total_reward = sum(episode_rewards[env_idx]).item()
            total_rewards.append(total_reward)
    
    if total_rewards:
        avg_reward = sum(total_rewards) / len(total_rewards)
        print(f"Episode {episode + 1} completed in {step} steps, Avg Reward: {avg_reward:.2f}")
    
    # TODO: Add policy update logic here using the collected episode data
    # This would involve computing returns and updating the policy network






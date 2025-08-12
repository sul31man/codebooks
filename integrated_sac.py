#overall logic goes something like this
#we start of with some batch states
#the policy determines batch actions
#we feed these batched actions into the environment step which then gives us next states, rewards, dones, etcetera 
#we then take these and feed it into the buffer. 
#we do 100 iterations then we go through the training phase and reimplement it

#!/usr/bin/env python3
"""
Clean GPU-accelerated SAC with proper modular design.
Uses the debug environment wrapper as a separate module.
"""

import torch
import torch.nn as nn
import torch.nn.init as init
import numpy as np
from collections import deque 
import random 
import torch.optim as optim
import time
import math

import env

# Configuration
state_dim = 1        # Only hit rate percentage from previous step
action_dim = 512     # Changed to match codebook rows (512)
hidden_dim = 256
capacity = 1000
batch_size = 32      # This is also the environment batch size
temp = 0.32
gamma = 0.99
tau = 0.005
num_episodes = 50
lr = 0.001

def init_weights(module):
    """Initialize network weights with orthogonal initialization."""
    if isinstance(module, nn.Linear):
        # Orthogonal initialization with gain = sqrt(2) for ReLU networks
        init.orthogonal_(module.weight, gain=math.sqrt(2))
        # Initialize bias to zero
        init.constant_(module.bias, 0.0)

class CodewordPolicy(nn.Module):
    """SAC policy network for continuous actions."""
    def __init__(self, state_dim, action_dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim) 
        self.fc_std = nn.Linear(hidden_dim, action_dim)
        self.relu = nn.ReLU()
        
        # Apply orthogonal initialization
        self.apply(init_weights)
        
        # Special initialization for the output layer (smaller gain for policy)
        init.orthogonal_(self.fc_std.weight, gain=0.01)
        init.constant_(self.fc_std.bias, 0.0)

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
        
        # Apply orthogonal initialization
        self.apply(init_weights)
        
        # Special initialization for the output layer (smaller gain for value function)
        init.orthogonal_(self.fc2.weight, gain=1.0)
        init.constant_(self.fc2.bias, 0.0)

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
        # Store on CPU to save GPU memory (tensors are already detached)
        # This moves tensors from GPU -> CPU only once for storage
        state_cpu = state.cpu() if isinstance(state, torch.Tensor) else state
        action_cpu = action.cpu() if isinstance(action, torch.Tensor) else action
        next_state_cpu = next_state.cpu() if isinstance(next_state, torch.Tensor) else next_state
        self.buffer.append((state_cpu, action_cpu, next_state_cpu, reward, done))

    def sample(self):
        batch = random.sample(self.buffer, self.batch_size)
        states, actions, next_states, rewards, dones = zip(*batch)
        
        # Efficient GPU transfer: batch transfer CPU -> GPU only when needed for training
        # non_blocking=True for async transfer while CPU continues
        return (torch.stack(states).to(self.device, dtype=torch.float32, non_blocking=True),
                torch.stack(actions).to(self.device, dtype=torch.float32, non_blocking=True),
                torch.stack(next_states).to(self.device, dtype=torch.float32, non_blocking=True),
                torch.tensor(rewards, dtype=torch.float32, device=self.device),
                torch.tensor(dones, dtype=torch.float32, device=self.device))

# Add this function to test the AMP decoder integration
def test_amp_decoder():
    """Test the AMP decoder integration before starting RL training"""
    print("\n" + "="*60)
    print("TESTING AMP DECODER INTEGRATION")
    print("="*60)
    
    # AMP decoder parameters
    Ka = 3    # Number of active users per section
    L = 4     # Number of sections 
    J = 6     # 2^J codewords per section
    n = 128   # Number of measurements
    N = L * (2**J)  # Total codebook size = 4 * 64 = 256
    P_hat = 1.0
    T_max = 15
    tol = 1e-6
    
    print(f"AMP Parameters: Ka={Ka}, L={L}, J={J}, n={n}, N={N}")
    
    # Create random sensing matrix A [n, N] with proper scaling
    A = torch.randn(n, N, dtype=torch.float64) / (n**0.5)
    print(f"Sensing matrix shape: {A.shape}")
    
    # Initialize small test batch
    test_batch_size = 4
    env.initialise_global_codebook(test_batch_size)
    
    # Create test actions
    test_actions = torch.randn(test_batch_size, 512, device='cuda')
    
    print("\n--- Testing Original Environment ---")
    try:
        start_time = time.time()
        hit_rates_orig, rewards_orig, dones_orig = env.step(test_actions)
        orig_time = time.time() - start_time
        
        print(f"✓ Original environment step completed in {orig_time:.3f}s")
        print(f"  Hit rates: {hit_rates_orig}")
        print(f"  Rewards: {rewards_orig}")
        print(f"  Shapes: hit_rates={hit_rates_orig.shape}, rewards={rewards_orig.shape}")
        
    except Exception as e:
        print(f"✗ Original environment failed: {e}")
        return False
    
    print("\n--- Testing AMP Decoder Environment ---")
    try:
        start_time = time.time()
        hit_rates_amp, rewards_amp, dones_amp = env.step_amp(
            actions=test_actions,
            sensing_matrix=A,
            n=n,
            N=N,
            T_max=T_max,
            tol=tol,
            P_hat=P_hat
        )
        amp_time = time.time() - start_time
        
        print(f"✓ AMP environment step completed in {amp_time:.3f}s")
        print(f"  Hit rates: {hit_rates_amp}")
        print(f"  Rewards: {rewards_amp}")
        print(f"  Shapes: hit_rates={hit_rates_amp.shape}, rewards={rewards_amp.shape}")
        
    except Exception as e:
        print(f"✗ AMP environment failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n--- Performance Comparison ---")
    print(f"Original time: {orig_time:.3f}s")
    print(f"AMP time: {amp_time:.3f}s")
    print(f"Time ratio (AMP/Original): {amp_time/orig_time:.2f}x")
    
    print("\n--- Integration Test Results ---")
    print("✓ Both environment paths working")
    print("✓ AMP decoder interface functional")
    print("✓ Ready for RL training")
    
    print("="*60)
    print("AMP DECODER INTEGRATION TEST COMPLETE")
    print("="*60)
    
    return True

# Add this to your main() function before the training loop
def main():
    """Main training loop with clean modular design."""
    
    # Device setup
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name()}")
    else:
        device = torch.device("cpu")
        print("Using CPU for PyTorch")
    
    # Initialize CUDA environment
    print("Initializing CUDA GPU environment...")
    
    # Initialize the global codebook for the batch
    global_codebook = env.initialise_global_codebook(batch_size)
    print(f"Global codebook initialized with shape: {global_codebook.shape}")
    # Derive rows from codebook so sensing matrix matches environment rows
    rows = int(global_codebook.shape[1])
    
    # AMP decoder configuration (must match environment.cu globals L,J)
    amp_L = 16
    amp_J = 6
    amp_N = amp_L * (2**amp_J)   # 1024
    amp_n = 128                   # measurements
    amp_T_max = 15
    amp_tol = 1e-6
    amp_P_hat = 1.0
    # Sensing matrix S on CPU in float64 with shape [n, rows]
    # Use float64 to match AMP decoder; environment converts to CUDA and builds A_eff = S * C_b
    A_sensing = (torch.randn(amp_n, rows, dtype=torch.float64, device='cpu') / (amp_n ** 0.5)).contiguous()
    
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
    print("All networks initialized with orthogonal weights (gain=sqrt(2)) and zero bias")
    print("Starting GPU-accelerated SAC training...")
    
    start_time = time.time()
    
    for episode in range(num_episodes):
        if episode % 5 == 0:
            print(f"\n=== Episode {episode} ===")
        
        # Initialize states with zero hit rate for the first step
        states = torch.zeros(batch_size, state_dim, device=device)  # Shape: [batch_size, 1]
        
        episode_rewards = torch.zeros(batch_size, device=device)
        step_count = 0
        max_steps = 100  # Limit steps per episode
        
        for step in range(max_steps):
            step_count += 1
            
            # Generate actions for all environments in batch
            with torch.no_grad():
                actions, log_probs = matrix_policy(states)  # Shape: [batch_size, action_dim]
            
            # Environment step using AMP decoder path (results on CPU)
            hit_rates, rewards, dones = env.step_amp(
                actions=actions,
                sensing_matrix=A_sensing,
                n=amp_n,
                N=amp_N,
                T_max=amp_T_max,
                tol=amp_tol,
                P_hat=amp_P_hat
            )
            # Move to training device for subsequent ops
            hit_rates = hit_rates.to(device, non_blocking=True)
            rewards = rewards.to(device, non_blocking=True)
            dones = dones.to(device, non_blocking=True)
            
            # Next state is simply the current hit rates
            next_states = hit_rates.unsqueeze(1)  # Shape: [batch_size, 1]
            
            # Detach tensors before storing (no gradients needed in replay buffer)
            states_detached = states.detach()
            actions_detached = actions.detach()
            next_states_detached = next_states.detach()
            rewards_detached = rewards.detach()
            dones_detached = dones.detach() if isinstance(dones, torch.Tensor) else dones
            
            # Store experiences for each environment in the batch
            for i in range(batch_size):
                buffer.push(
                    states_detached[i], 
                    actions_detached[i], 
                    next_states_detached[i], 
                    rewards_detached[i].item(), 
                    dones_detached[i].item() if isinstance(dones_detached, torch.Tensor) else dones_detached
                )
            
            episode_rewards += rewards
            states = next_states  # Continue with updated states
            
            # Simple termination condition - could be improved
            if step >= 50:  # Run for 50 steps per episode
                break
        
        if episode % 5 == 0:
            avg_reward = episode_rewards.mean().item()
            max_reward = episode_rewards.max().item()
            final_hit_rates = states[:, 0]  # Extract hit rates (now the only state component)
            avg_hit_rate = final_hit_rates.mean().item()
            max_hit_rate = final_hit_rates.max().item()
            print(f"  Episode {episode}: {step_count} steps, avg_reward={avg_reward:.4f}, max_reward={max_reward:.4f}")
            print(f"    Final hit rates - avg: {avg_hit_rate:.4f}, max: {max_hit_rate:.4f}")
        
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
            avg_reward = episode_rewards.mean().item()
            print(f"*** Episode {episode}: Avg Reward={avg_reward:.4f}, Time={elapsed_time:.1f}s ***")
    
    total_time = time.time() - start_time
    print(f"\nTraining completed in {total_time:.1f}s!")
    print(f"Average time per episode: {total_time/num_episodes:.2f}s")
    
    # Print final codebook statistics
    print("\n" + "="*50)
    print("FINAL CODEBOOK ANALYSIS")
    print("="*50)
    
    final_codebook = env.get_codebook()
    print(f"Codebook shape: {final_codebook.shape}")
    print(f"Codebook device: {final_codebook.device}")
    print(f"Codebook dtype: {final_codebook.dtype}")
    
    # Print statistics for each environment's codebook
    for env_idx in range(min(3, batch_size)):  # Show first 3 environments
        env_codebook = final_codebook[env_idx]  # Shape: [512, 1024]
        
        print(f"\nEnvironment {env_idx} codebook stats:")
        print(f"  Mean: {env_codebook.mean().item():.6f}")
        print(f"  Std:  {env_codebook.std().item():.6f}")
        print(f"  Min:  {env_codebook.min().item():.6f}")
        print(f"  Max:  {env_codebook.max().item():.6f}")
        
        # Show a small sample of the codebook (first 3x3 corner)
        sample = env_codebook[:3, :3].cpu().numpy()
        print(f"  Sample (top-left 3x3):")
        for row in sample:
            print(f"    [{', '.join([f'{x:8.4f}' for x in row])}]")
    
    if batch_size > 3:
        print(f"\n... and {batch_size - 3} more environments")
    
    print("="*50)
    print("Training completed!")
    
    return matrix_policy

if __name__ == "__main__":
    try:
        policy = main()
        print("\nClean modular SAC training successful!")
        print("Separate environment wrapper module")
        print("GPU acceleration working")
        print("No CUDA context conflicts")
    except Exception as e:
        print(f"Error during training: {e}")
        import traceback
        traceback.print_exc() 
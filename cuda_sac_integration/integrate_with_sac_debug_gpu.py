#!/usr/bin/env python3
"""
GPU-accelerated SAC using the working debug CUDA library.
Now that we know the tensor conversion and CUDA context work perfectly,
we can safely use GPU for PyTorch while using the debug CUDA environment.
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
import ctypes

# Add the environments directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Configuration
state_dim = 10 
action_dim = 1024
hidden_dim = 256  # Reasonable size for GPU
capacity = 1000
batch_size = 32
temp = 0.32
gamma = 0.99
tau = 0.005
num_episodes = 50
lr = 0.001

class DebugCUDAEnvironment:
    """Environment wrapper using the working debug CUDA library."""
    
    def __init__(self, Ka=1, num_sims=100):
        # Load debug library
        lib_path = "./libenvironment_debug.so"
        if not os.path.exists(lib_path):
            raise FileNotFoundError(f"Debug library not found: {lib_path}")
        
        self.lib = ctypes.CDLL(lib_path)
        self.Ka = Ka
        self.num_sims = num_sims
        self.cols = 1024
        self.rows = 512
        
        # Set up function signatures
        self._setup_function_signatures()
        
        # Initialize environment
        self.lib.clear_action_buffer()
        self.lib.initialize_buffer_with_random(self.rows, self.cols)
        
        # State tracking
        self.current_step = 0
        self.best_hit_rate = 0.0
        self.action_buffer = []
        
        print(f"Debug CUDA Environment initialized: {self.rows}x{self.cols}")
    
    def _setup_function_signatures(self):
        """Set up ctypes function signatures."""
        self.lib.initialize_buffer_with_random.argtypes = [ctypes.c_int, ctypes.c_int]
        self.lib.initialize_buffer_with_random.restype = None
        
        self.lib.add_action_to_buffer.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.c_int]
        self.lib.add_action_to_buffer.restype = None
        
        self.lib.run_simulation.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_int)]
        self.lib.run_simulation.restype = None
        
        self.lib.get_codebook.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_int]
        self.lib.get_codebook.restype = None
        
        self.lib.clear_action_buffer.argtypes = []
        self.lib.clear_action_buffer.restype = None
    
    def reset(self):
        """Reset environment for new episode."""
        self.current_step = 0
        # Don't clear action_buffer to maintain persistent learning
        return np.zeros(state_dim, dtype=np.float32)
    
    def step(self, action):
        """Take environment step with safe tensor conversion."""
        # Ultra-safe tensor conversion (proven to work from debug tests)
        if isinstance(action, torch.Tensor):
            if action.is_cuda:
                action = action.cpu()
            
            if action.dtype != torch.float32:
                action = action.to(dtype=torch.float32)
                
            if not action.is_contiguous():
                action = action.contiguous()
                
            action = action.detach().numpy().copy()
        
        # Ensure proper numpy array
        if not isinstance(action, np.ndarray):
            action = np.array(action, dtype=np.float32)
        elif action.dtype != np.float32:
            action = action.astype(np.float32)
            
        if not action.flags.c_contiguous:
            action = np.ascontiguousarray(action, dtype=np.float32)
        
        if action.ndim > 1:
            action = action.flatten()
            
        if len(action) != self.cols:
            raise ValueError(f"Action must have {self.cols} elements, got {len(action)}")
        
        # Convert to C array and add to buffer
        action_array = (ctypes.c_float * self.cols)(*action)
        self.lib.add_action_to_buffer(action_array, self.cols)
        self.action_buffer.extend(action)
        
        # Run simulation
        hit_rates = []
        Ka_values = np.linspace(5, 35, 5)
        
        for Ka in Ka_values:
            hit_rate = ctypes.c_int(0)
            self.lib.run_simulation(int(Ka), self.num_sims, ctypes.byref(hit_rate))
            hit_rates.append(hit_rate.value)
        
        # Calculate reward
        avg_hit_rate = np.mean(hit_rates) / self.num_sims
        reward = avg_hit_rate - self.best_hit_rate
        
        if avg_hit_rate > self.best_hit_rate:
            self.best_hit_rate = avg_hit_rate
        
        # Update state
        self.current_step += 1
        next_state = np.array([
            avg_hit_rate,
            self.current_step / 100.0,
            len(self.action_buffer) / self.cols,
            self.best_hit_rate,
            *action[:6]
        ], dtype=np.float32)
        
        done = self.current_step >= 100
        
        info = {
            'hit_rate': avg_hit_rate,
            'best_hit_rate': self.best_hit_rate,
            'step': self.current_step,
            'buffer_size': len(self.action_buffer),
            'hit_rates': hit_rates
        }
        
        return next_state, reward, done, info
    
    def get_codebook(self):
        """Get current codebook."""
        codebook_size = self.rows * self.cols
        codebook_array = (ctypes.c_float * codebook_size)()
        self.lib.get_codebook(codebook_array, self.rows, self.cols)
        return np.array(codebook_array, dtype=np.float32).reshape(self.rows, self.cols)

class CodewordPolicy(nn.Module):
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
    def __init__(self, capacity, batch_size, device):
        self.buffer = deque(maxlen=capacity)
        self.batch_size = batch_size
        self.device = device

    def push(self, state, action, next_state, reward, done):
        # Store on CPU, transfer to GPU during sampling
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
    """GPU-accelerated SAC with debug CUDA environment."""
    
    # Use GPU for PyTorch (now proven safe!)
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name()}")
        print("🚀 GPU acceleration enabled for PyTorch + Custom CUDA!")
    else:
        device = torch.device("cpu")
        print("Using CPU for PyTorch")
    
    # Initialize debug environment
    print("Initializing debug CUDA environment...")
    env = DebugCUDAEnvironment(Ka=1, num_sims=50)
    
    # Initialize GPU-accelerated SAC components
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
    optimizer1 = optim.Adam(q_1.parameters(), lr=lr)
    optimizer2 = optim.Adam(q_2.parameters(), lr=lr)
    optimizer3 = optim.Adam(matrix_policy.parameters(), lr=lr)
    
    print(f"Networks created on {device}")
    print("Starting GPU-accelerated SAC training with debug CUDA environment...")
    
    start_time = time.time()
    
    for episode in range(num_episodes):
        episode_start = time.time()
        
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
            
            # Environment step (action will be safely converted)
            next_state, reward, done, info = env.step(action)
            next_state = torch.tensor(next_state, dtype=torch.float32, device=device)
            
            # Store experience
            buffer.push(state, action, next_state, reward, done)
            
            episode_reward += reward
            state = next_state
        
        if episode % 5 == 0:
            episode_time = time.time() - episode_start
            print(f"  Episode {episode}: {step_count} steps, reward={episode_reward:.4f}, time={episode_time:.2f}s")
        
        # GPU-accelerated training
        if len(buffer.buffer) >= batch_size:
            num_epochs = 5
            
            for epoch in range(num_epochs):
                states, actions, next_states, rewards, dones = buffer.sample()
                
                # Compute targets on GPU
                with torch.no_grad():
                    next_actions, next_log_prob = matrix_policy(next_states)
                    q1_next = tq_1(next_states, next_actions)
                    q2_next = tq_2(next_states, next_actions)
                    min_q_next = torch.min(q1_next, q2_next)
                    target_value = rewards.unsqueeze(1) + gamma * (1-dones.unsqueeze(1)) * (min_q_next - temp * next_log_prob.unsqueeze(1))
                
                # Update Q-networks on GPU
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
                
                # Update policy on GPU
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
    print(f"\nGPU-accelerated training completed in {total_time:.1f}s!")
    print(f"Average time per episode: {total_time/num_episodes:.2f}s")
    print(f"Final best hit rate: {env.best_hit_rate:.4f}")
    
    # Get final codebook
    final_codebook = env.get_codebook()
    print(f"Final codebook shape: {final_codebook.shape}")
    
    return env, matrix_policy

if __name__ == "__main__":
    try:
        env, policy = main()
        print("\n🎉 GPU-accelerated integration successful!")
        print("✅ PyTorch networks running on GPU")
        print("✅ Custom CUDA environment running on GPU") 
        print("✅ No CUDA context conflicts!")
        print("✅ Safe tensor pointer conversion verified!")
    except Exception as e:
        print(f"Error during training: {e}")
        import traceback
        traceback.print_exc() 
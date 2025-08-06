#!/usr/bin/env python3
"""
Simple test to isolate CUDA issues from SAC training complexity.
"""

import torch
import numpy as np
from environment_wrapper import CUDAEnvironment

def test_simple_training():
    """Test simple environment interaction without SAC complexity."""
    
    print("Testing simple CUDA environment interaction...")
    
    # Initialize environment
    env = CUDAEnvironment(Ka=1, num_sims=100)
    
    # Simple interaction loop
    state = env.reset()
    print(f"Initial state shape: {state.shape}")
    
    total_reward = 0.0
    
    for step in range(10):
        # Generate random action
        action = torch.randn(1024)  # 1024-dimensional action
        
        # Take step
        next_state, reward, done, info = env.step(action)
        
        print(f"Step {step + 1}:")
        print(f"  Action shape: {action.shape}")
        print(f"  Reward: {reward:.4f}")
        print(f"  Hit rate: {info['hit_rate']:.4f}")
        print(f"  Done: {done}")
        
        total_reward += reward
        state = next_state
        
        if done:
            break
    
    print(f"\nTest completed!")
    print(f"Total reward: {total_reward:.4f}")
    print(f"Best hit rate: {env.best_hit_rate:.4f}")
    
    # Get final codebook
    codebook = env.get_codebook()
    print(f"Final codebook shape: {codebook.shape}")
    
    return env

if __name__ == "__main__":
    try:
        env = test_simple_training()
        print("\n✅ Simple test completed successfully!")
    except Exception as e:
        print(f"❌ Error during simple test: {e}")
        print("Please check CUDA installation and library compilation.") 
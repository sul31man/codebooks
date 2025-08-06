#!/usr/bin/env python3
"""
Test script to verify the CUDA environment works correctly.
"""

import numpy as np
import torch
import sys
import os

# Add the environments directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from environment_wrapper import CUDAEnvironment

def test_environment():
    """Test the CUDA environment functionality."""
    
    print("Testing CUDA Environment...")
    
    try:
        # Initialize environment with hardcoded dimensions
        env = CUDAEnvironment(Ka=1, num_sims=100)
        
        print("✓ Environment initialized successfully")
        print(f"  Dimensions: {env.rows} rows x {env.cols} cols")
        
        # Test reset
        state = env.reset()
        print(f"✓ Reset successful, state shape: {state.shape}")
        
        # Test getting initial codebook
        codebook = env.get_codebook()
        print(f"✓ Initial codebook shape: {codebook.shape}")
        print(f"  Codebook range: [{codebook.min():.3f}, {codebook.max():.3f}]")
        
        # Test a few steps
        for i in range(3):
            # Generate random action
            action = torch.randn(env.cols)  # Use environment cols
            
            # Take step
            next_state, reward, done, info = env.step(action)
            
            print(f"✓ Step {i+1}:")
            print(f"  Action shape: {action.shape}")
            print(f"  Reward: {reward:.4f}")
            print(f"  Hit rate: {info['hit_rate']:.4f}")
            print(f"  Done: {done}")
            print(f"  Buffer size: {info['buffer_size']}")
            
            if done:
                break
        
        # Test final codebook
        final_codebook = env.get_codebook()
        print(f"✓ Final codebook shape: {final_codebook.shape}")
        
        print("\n🎉 All tests passed! Environment is working correctly.")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_environment()
    if success:
        print("\n✅ Environment is ready for SAC training!")
    else:
        print("\n❌ Environment needs fixes before training.") 
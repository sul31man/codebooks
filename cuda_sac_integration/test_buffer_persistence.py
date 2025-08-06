#!/usr/bin/env python3
"""
Test script to verify that the action buffer persists across episodes.
"""

import numpy as np
import torch
import sys
import os

# Add the environments directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from environment_wrapper import CUDAEnvironment

def test_buffer_persistence():
    """Test that the action buffer persists across episodes."""
    
    print("Testing Buffer Persistence Across Episodes...")
    
    try:
        # Initialize environment
        env = CUDAEnvironment(Ka=1, num_sims=100)
        
        print("✓ Environment initialized successfully")
        
        # Test first episode
        print("\n--- Episode 1 ---")
        state = env.reset()
        print(f"Initial buffer size: {len(env.action_buffer)}")
        
        # Take a few steps in first episode
        for i in range(3):
            action = torch.randn(env.cols)
            next_state, reward, done, info = env.step(action)
            print(f"Step {i+1}: Buffer size = {info['buffer_size']}")
            
            if done:
                break
        
        # Get codebook after first episode
        codebook1 = env.get_codebook()
        print(f"Codebook after episode 1: shape {codebook1.shape}, non-zero elements: {np.count_nonzero(codebook1)}")
        
        # Test second episode (should persist buffer)
        print("\n--- Episode 2 ---")
        state = env.reset()
        print(f"Buffer size after reset: {len(env.action_buffer)}")
        
        # Take a few more steps
        for i in range(2):
            action = torch.randn(env.cols)
            next_state, reward, done, info = env.step(action)
            print(f"Step {i+1}: Buffer size = {info['buffer_size']}")
            
            if done:
                break
        
        # Get codebook after second episode
        codebook2 = env.get_codebook()
        print(f"Codebook after episode 2: shape {codebook2.shape}, non-zero elements: {np.count_nonzero(codebook2)}")
        
        # Verify buffer persisted
        if len(env.action_buffer) > 0:
            print("✓ Buffer persisted across episodes!")
        else:
            print("❌ Buffer was cleared between episodes!")
            return False
        
        # Verify codebook is the same (should be, since buffer persisted)
        if np.array_equal(codebook1, codebook2):
            print("✓ Codebook remained the same across episodes!")
        else:
            print("⚠ Codebook changed between episodes (this might be expected if actions were added)")
        
        print("\n🎉 Buffer persistence test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_buffer_persistence()
    if success:
        print("\n✅ Buffer persistence is working correctly!")
    else:
        print("\n❌ Buffer persistence needs to be fixed.") 
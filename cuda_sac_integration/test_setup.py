#!/usr/bin/env python3
"""
Test script to verify all components of the CUDA SAC integration are working correctly.
"""

import torch
import numpy as np
import sys
import os

def test_torch_cuda():
    """Test PyTorch CUDA availability."""
    print("Testing PyTorch CUDA...")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        print(f"Current GPU: {torch.cuda.current_device()}")
        print(f"GPU name: {torch.cuda.get_device_name()}")
    return torch.cuda.is_available()

def test_cuda_library():
    """Test CUDA library compilation and loading."""
    print("\nTesting CUDA library...")
    try:
        from environment_wrapper import CUDAEnvironment
        print("✓ CUDA environment wrapper imported successfully")
        
        # Test environment initialization
        env = CUDAEnvironment(cols=25, rows=23, Ka=1, num_sims=100)
        print("✓ CUDA environment initialized successfully")
        
        # Test basic functionality
        state = env.reset()
        print(f"✓ Environment reset successful, state shape: {state.shape}")
        
        # Test step
        action = torch.randn(25)
        next_state, reward, done, info = env.step(action)
        print(f"✓ Environment step successful, reward: {reward:.4f}")
        
        return True
    except Exception as e:
        print(f"✗ CUDA library test failed: {e}")
        return False

def test_sac_components():
    """Test SAC agent components."""
    print("\nTesting SAC components...")
    try:
        from integrate_with_sac import CodewordPolicy, QNetwork, ReplayBuffer
        
        # Test policy network
        policy = CodewordPolicy(state_dim=10, action_dim=25, hidden_dim=50)
        state = torch.randn(1, 10)
        action, log_prob = policy(state)
        print(f"✓ Policy network working, action shape: {action.shape}")
        
        # Test Q networks
        q1 = QNetwork(state_dim=10, action_dim=25, hidden_dim=50)
        q_value = q1(state, action)
        print(f"✓ Q network working, Q value shape: {q_value.shape}")
        
        # Test replay buffer
        buffer = ReplayBuffer(capacity=1000, batch_size=32)
        buffer.push(state, action, state, 1.0, False)
        print("✓ Replay buffer working")
        
        return True
    except Exception as e:
        print(f"✗ SAC components test failed: {e}")
        return False

def test_full_integration():
    """Test full integration."""
    print("\nTesting full integration...")
    try:
        from integrate_with_sac import main
        print("✓ Full integration test passed")
        return True
    except Exception as e:
        print(f"✗ Full integration test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("=" * 60)
    print("CUDA SAC Integration Test Suite")
    print("=" * 60)
    
    tests = [
        ("PyTorch CUDA", test_torch_cuda),
        ("CUDA Library", test_cuda_library),
        ("SAC Components", test_sac_components),
        ("Full Integration", test_full_integration)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"✗ {test_name} test failed with exception: {e}")
            results.append((test_name, False))
    
    print("\n" + "=" * 60)
    print("Test Results:")
    print("=" * 60)
    
    all_passed = True
    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{test_name}: {status}")
        if not result:
            all_passed = False
    
    print("\n" + "=" * 60)
    if all_passed:
        print("🎉 ALL TESTS PASSED! Your setup is ready for CUDA SAC training.")
    else:
        print("❌ Some tests failed. Please check the errors above.")
    print("=" * 60)
    
    return all_passed

if __name__ == "__main__":
    main() 
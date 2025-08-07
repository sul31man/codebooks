#!/usr/bin/env python3
"""
Parallel CUDA Environment Wrapper
Processes multiple Ka values simultaneously for better performance.
"""

import numpy as np
import ctypes
import os
import torch
from typing import Tuple, Optional, Dict, Any

class ParallelCUDAEnvironment:
    """
    Environment wrapper using parallel CUDA implementation.
    Processes multiple Ka values simultaneously for better performance.
    """
    
    def __init__(self, Ka_values=None, num_sims: int = 100):
        """
        Initialize the parallel CUDA environment wrapper.
        
        Args:
            Ka_values: List of Ka values to simulate in parallel
            num_sims: Number of simulations to run for evaluation
        """
        # Default Ka values if none provided
        self.Ka_values = Ka_values or [5, 12, 20, 27, 35]
        self.num_Ka = len(self.Ka_values)
        self.num_sims = num_sims
        
        # Hardcoded dimensions matching CUDA constants
        self.cols = 1024  # L*N = 16*64
        self.rows = 512   # n = 512
        
        # Load parallel library
        lib_path = "./libenvironment_parallel.so"
        if not os.path.exists(lib_path):
            raise FileNotFoundError(f"Parallel library not found: {lib_path}")
        
        self.lib = ctypes.CDLL(lib_path)
        self._setup_function_signatures()
        
        # Initialize environment
        self.lib.clear_action_buffer()
        self.lib.initialize_buffer_with_random(self.rows, self.cols)
        
        # State tracking
        self.current_step = 0
        self.best_hit_rate = 0.0
        self.action_buffer = []
        
        print(f"Parallel CUDA Environment initialized: {self.rows}x{self.cols}")
        print(f"Processing {self.num_Ka} Ka values in parallel: {self.Ka_values}")
    
    def _setup_function_signatures(self):
        """Set up ctypes function signatures."""
        # Standard functions
        self.lib.initialize_buffer_with_random.argtypes = [ctypes.c_int, ctypes.c_int]
        self.lib.initialize_buffer_with_random.restype = None
        
        self.lib.add_action_to_buffer.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.c_int]
        self.lib.add_action_to_buffer.restype = None
        
        self.lib.get_codebook.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_int]
        self.lib.get_codebook.restype = None
        
        self.lib.clear_action_buffer.argtypes = []
        self.lib.clear_action_buffer.restype = None
        
        # New parallel simulation function
        self.lib.run_parallel_simulation.argtypes = [
            ctypes.POINTER(ctypes.c_int),  # Ka_values
            ctypes.c_int,                  # num_Ka
            ctypes.c_int,                  # num_sims
            ctypes.POINTER(ctypes.c_int)   # hit_rates
        ]
        self.lib.run_parallel_simulation.restype = None
    
    def reset(self) -> np.ndarray:
        """Reset environment for new episode."""
        self.current_step = 0
        # Don't clear action_buffer to maintain persistent learning
        return np.zeros(10, dtype=np.float32)  # State dimension = 10
    
    def _safe_tensor_conversion(self, action):
        """
        Ultra-safe tensor conversion proven to work in debug tests.
        Handles all tensor types: CPU/GPU, float32/float64, contiguous/non-contiguous.
        """
        if isinstance(action, torch.Tensor):
            # Step 1: Move to CPU if on CUDA
            if action.is_cuda:
                action = action.cpu()
            
            # Step 2: Ensure float32 dtype
            if action.dtype != torch.float32:
                action = action.to(dtype=torch.float32)
                
            # Step 3: Ensure contiguous memory layout
            if not action.is_contiguous():
                action = action.contiguous()
                
            # Step 4: Convert to numpy with copy for memory safety
            action = action.detach().numpy().copy()
        
        # Ensure proper numpy array properties
        if not isinstance(action, np.ndarray):
            action = np.array(action, dtype=np.float32)
        elif action.dtype != np.float32:
            action = action.astype(np.float32)
            
        # Ensure C-contiguous memory layout
        if not action.flags.c_contiguous:
            action = np.ascontiguousarray(action, dtype=np.float32)
        
        # Flatten if multi-dimensional
        if action.ndim > 1:
            action = action.flatten()
            
        return action
    
    def step(self, action) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """
        Take a step in the environment with parallel Ka value processing.
        
        Args:
            action: Action tensor/array from SAC agent (1024 elements)
            
        Returns:
            Tuple of (next_state, reward, done, info)
        """
        # Safe tensor conversion
        action = self._safe_tensor_conversion(action)
            
        # Validate action dimensions
        if len(action) != self.cols:
            raise ValueError(f"Action must have {self.cols} elements, got {len(action)}")
        
        # Convert to C array and add to buffer
        action_array = (ctypes.c_float * self.cols)(*action)
        self.lib.add_action_to_buffer(action_array, self.cols)
        self.action_buffer.extend(action)
        
        # Run parallel simulation for all Ka values
        Ka_array = (ctypes.c_int * self.num_Ka)(*self.Ka_values)
        hit_rates = (ctypes.c_int * self.num_Ka)()
        
        self.lib.run_parallel_simulation(
            Ka_array,
            ctypes.c_int(self.num_Ka),
            ctypes.c_int(self.num_sims),
            hit_rates
        )
        
        # Convert hit rates to numpy array
        hit_rates = np.array([hit_rates[i] for i in range(self.num_Ka)])
        
        # Calculate reward based on average hit rate
        avg_hit_rate = np.mean(hit_rates) / self.num_sims
        reward = avg_hit_rate - self.best_hit_rate
        
        # Update best performance
        if avg_hit_rate > self.best_hit_rate:
            self.best_hit_rate = avg_hit_rate
        
        # Update state
        self.current_step += 1
        next_state = np.array([
            avg_hit_rate,                           # Current performance
            self.current_step / 100.0,              # Progress through episode
            len(self.action_buffer) / self.cols,    # Buffer fullness
            self.best_hit_rate,                     # Best performance so far
            *action[:6]                             # First 6 action components
        ], dtype=np.float32)
        
        # Episode termination
        done = self.current_step >= 100
        
        # Info dictionary
        info = {
            'hit_rate': avg_hit_rate,
            'best_hit_rate': self.best_hit_rate,
            'step': self.current_step,
            'buffer_size': len(self.action_buffer),
            'hit_rates': hit_rates,
            'Ka_values': self.Ka_values
        }
        
        return next_state, reward, done, info
    
    def get_codebook(self) -> np.ndarray:
        """Get the current codebook as a numpy array."""
        codebook_size = self.rows * self.cols
        codebook_array = (ctypes.c_float * codebook_size)()
        self.lib.get_codebook(codebook_array, self.rows, self.cols)
        return np.array(codebook_array, dtype=np.float32).reshape(self.rows, self.cols)
    
    def clear_buffer(self):
        """Clear the action buffer."""
        self.lib.clear_action_buffer()
        self.action_buffer = []
        self.current_step = 0
        self.best_hit_rate = 0.0

# Test the parallel environment
if __name__ == "__main__":
    """Test the parallel environment wrapper."""
    try:
        print("Testing Parallel CUDA Environment...")
        env = ParallelCUDAEnvironment(Ka_values=[5, 12, 20, 27, 35], num_sims=50)
        
        # Test reset
        state = env.reset()
        print(f"✅ Reset successful: state shape = {state.shape}")
        
        # Test different tensor types
        test_actions = [
            torch.randn(1024, dtype=torch.float32),  # CPU float32
            torch.randn(1024, dtype=torch.float64),  # CPU float64
        ]
        
        if torch.cuda.is_available():
            test_actions.extend([
                torch.randn(1024, dtype=torch.float32, device='cuda'),  # GPU float32
                torch.randn(1024, dtype=torch.float64, device='cuda'),  # GPU float64
            ])
        
        for i, action in enumerate(test_actions):
            print(f"\nTesting action {i+1}: {action.dtype} on {action.device}")
            next_state, reward, done, info = env.step(action)
            print(f"✅ Step successful: reward={reward:.4f}")
            print(f"Hit rates for Ka values {env.Ka_values}:")
            print(info['hit_rates'])
        
        # Test codebook retrieval
        codebook = env.get_codebook()
        print(f"\n✅ Codebook retrieved: shape = {codebook.shape}")
        
        print("\n🎉 Parallel environment test successful!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc() 
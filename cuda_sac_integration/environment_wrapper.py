import numpy as np
import ctypes
import os
import torch
import subprocess
import sys
from typing import Tuple, Optional, Dict, Any

class CUDAEnvironment:
    """
    Python wrapper for the CUDA environment that maintains a persistent action buffer
    and provides a gym-like interface for the SAC agent.
    """
    
    def __init__(self, Ka: int = 1, num_sims: int = 1000):
        """
        Initialize the CUDA environment wrapper.
        
        Args:
            Ka: Number of active users
            num_sims: Number of simulations to run for evaluation
        """
        # Hardcoded dimensions from CUDA constants
        self.cols = 16 * 64  # L*N = 1024
        self.rows = 512      # n = 512
        self.Ka = Ka
        self.num_sims = num_sims
        
        # Load the CUDA library
        self._load_cuda_library()
        
        # Initialize the environment
        self._initialize_environment()
        
        # State tracking
        self.current_step = 0
        self.total_reward = 0.0
        self.best_hit_rate = 0.0
        
    def _compile_library(self):
        """Compile the CUDA library if it doesn't exist."""
        lib_path = os.path.join(os.path.dirname(__file__), "libenvironment.so")
        if not os.path.exists(lib_path):
            print("CUDA library not found. Compiling...")
            try:
                # Change to the environments directory
                current_dir = os.getcwd()
                env_dir = os.path.dirname(__file__)
                os.chdir(env_dir)
                
                # Run make
                result = subprocess.run(['make', 'clean', 'all'], 
                                      capture_output=True, text=True)
                
                if result.returncode != 0:
                    print(f"Compilation failed: {result.stderr}")
                    raise RuntimeError("Failed to compile CUDA library")
                
                print("Compilation successful!")
                os.chdir(current_dir)
                
            except Exception as e:
                print(f"Failed to compile library: {e}")
                raise
    
    def _load_cuda_library(self):
        """Load the compiled CUDA library."""
        try:
            # First try to compile if library doesn't exist
            self._compile_library()
            
            # Try to load the compiled library
            lib_path = os.path.join(os.path.dirname(__file__), "libenvironment.so")
            if os.path.exists(lib_path):
                self.lib = ctypes.CDLL(lib_path)
            else:
                raise FileNotFoundError(f"Library not found at {lib_path}")
                
        except Exception as e:
            print(f"Failed to load CUDA library: {e}")
            print("Please ensure CUDA is installed and nvcc is in your PATH")
            raise
    
    def _initialize_environment(self):
        """Initialize the CUDA environment."""
        # Set up function signatures
        self.lib.add_action_to_buffer.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.c_int]
        self.lib.add_action_to_buffer.restype = None
        
        self.lib.get_codebook.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_int]
        self.lib.get_codebook.restype = None
        
        # Add new function signatures
        self.lib.initialize_buffer_with_random.argtypes = [ctypes.c_int, ctypes.c_int]
        self.lib.initialize_buffer_with_random.restype = None
        
        self.lib.run_simulation.argtypes = [
            ctypes.c_int,  # Ka
            ctypes.c_int,  # num_sims
            ctypes.POINTER(ctypes.c_int)  # hit_rate
        ]
        self.lib.run_simulation.restype = None
        
        # Initialize buffer with random matrix using hardcoded dimensions
        self._initialize_random_buffer()
        
        # Initialize buffer
        self.action_buffer = []
    
    def _initialize_random_buffer(self):
        """Initialize the action buffer with random actions using hardcoded dimensions."""
        # Use hardcoded dimensions from CUDA: n=512 rows, L*N=16*64=1024 cols
        self.lib.initialize_buffer_with_random(ctypes.c_int(512), ctypes.c_int(16*64))
    
    def reset(self) -> np.ndarray:
        """
        Reset the environment and return initial state.
        Note: This does NOT reset the action buffer - it persists across episodes.
        
        Returns:
            Initial state as numpy array
        """
        self.current_step = 0
        self.total_reward = 0.0
        self.best_hit_rate = 0.0
        
        # DO NOT reinitialize buffer - it should persist across episodes
        # The buffer was already initialized once in _initialize_environment()
        
        # Return initial state (you can modify this based on your state representation)
        initial_state = np.zeros(10, dtype=np.float32)  # Example state dimension
        return initial_state
    
    def step(self, action: torch.Tensor) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """
        Take a step in the environment.
        
        Args:
            action: Action tensor from the SAC agent (should be 1D with cols elements)
            
        Returns:
            Tuple of (next_state, reward, done, info)
        """
        # Convert action to numpy and ensure it's 1D
        if isinstance(action, torch.Tensor):
            action = action.detach().cpu().numpy()
        
        if action.ndim > 1:
            action = action.flatten()
        
        # Ensure action has the correct size
        if len(action) != self.cols:
            raise ValueError(f"Action must have {self.cols} elements, got {len(action)}")
        
        # Convert action to C array
        action_array = (ctypes.c_float * self.cols)(*action)
        
        # Add action to the buffer
        self.lib.add_action_to_buffer(action_array, ctypes.c_int(self.cols))
        self.action_buffer.extend(action)
        
        # Run simulation to get performance (the CUDA function gets codebook internally)
        hit_rates = []
        upper_val = 35
        Ka_values = np.linspace(5, upper_val, 5)
        
        for Ka in Ka_values:  # Fixed: iterate over actual values, not range
            hit_rate = ctypes.c_int(0)
            
            # Fixed: call with correct parameters (Ka, num_sims, hit_rate)
            self.lib.run_simulation(
                ctypes.c_int(int(Ka)),  # Convert to int
                ctypes.c_int(self.num_sims),
                ctypes.byref(hit_rate)
                #not inputting the power yet but will do soon. 
            )
            hit_rates.append(hit_rate.value)  # Fixed: append the value, not the ctypes object
        
        # Calculate reward based on average hit rate
        hit_rates = np.array(hit_rates)
        avg_hit_rate = np.mean(hit_rates) / self.num_sims  # Normalize by num_sims
        
        current_hit_rate = avg_hit_rate
        reward = current_hit_rate - self.best_hit_rate  # Reward improvement
        
        # Update best hit rate
        if current_hit_rate > self.best_hit_rate:
            self.best_hit_rate = current_hit_rate
        
        # Update state
        self.current_step += 1
        
        # Create next state (modify based on your state representation)
        next_state = np.array([
            current_hit_rate,
            self.current_step / 100.0,  # Normalized step
            len(self.action_buffer) / self.cols,  # Buffer fullness
            self.best_hit_rate,
            *action[:6]  # First 6 action components as state
        ], dtype=np.float32)
        
        # Determine if episode is done
        done = self.current_step >= 100  # Example: 100 steps per episode
        
        # Info dictionary
        info = {
            'hit_rate': current_hit_rate,
            'best_hit_rate': self.best_hit_rate,
            'step': self.current_step,
            'buffer_size': len(self.action_buffer),
            'hit_rates': hit_rates  # Add the individual hit rates to info
        }
        
        return next_state, reward, done, info
    
    def get_codebook(self) -> np.ndarray:
        """Get the current codebook as a numpy array."""
        codebook_size = 512 * (16 * 64)  # n * (L*N)
        codebook_array = (ctypes.c_float * codebook_size)()
        self.lib.get_codebook(codebook_array, ctypes.c_int(512), ctypes.c_int(16*64))
        return np.array(codebook_array, dtype=np.float32).reshape(512, 16*64)
    
    def get_action_buffer(self) -> np.ndarray:
        """Get the current action buffer as a numpy array."""
        return np.array(self.action_buffer, dtype=np.float32)

# Example usage and testing
if __name__ == "__main__":
    # Test the environment
    try:
        env = CUDAEnvironment(Ka=1, num_sims=100)
        
        # Reset environment
        state = env.reset()
        print(f"Initial state: {state}")
        
        # Take a few steps
        for i in range(5):
            # Generate random action
            action = torch.randn(env.cols)  # Use environment cols
            
            # Take step
            next_state, reward, done, info = env.step(action)
            
            print(f"Step {i+1}:")
            print(f"  Action shape: {action.shape}")
            print(f"  Reward: {reward:.4f}")
            print(f"  Hit rate: {info['hit_rate']:.4f}")
            print(f"  Done: {done}")
            print(f"  Buffer size: {info['buffer_size']}")
            
            if done:
                break
        
        # Get final codebook
        codebook = env.get_codebook()
        print(f"Final codebook shape: {codebook.shape}")
        
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure CUDA is installed and the library can be compiled.") 
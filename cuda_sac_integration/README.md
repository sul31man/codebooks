# CUDA Environment Integration with SAC Agent

This directory contains the CUDA environment implementation and Python wrapper for integrating with your SAC agent.

## Overview

The CUDA environment provides:
- **Persistent Action Buffer**: Maintains a buffer of the last `cols` actions (forming the codebook)
- **High-Performance Simulation**: Runs parallel simulations on GPU for fast evaluation
- **Gym-like Interface**: Standard `reset()` and `step()` methods for easy integration

## Files

- `environment.cu`: CUDA implementation of the environment with persistent buffer
- `environment_wrapper.py`: Python wrapper that provides gym-like interface
- `Makefile`: Compilation script for the CUDA library
- `sac_integration_example.py`: Example showing how to integrate with SAC
- `integrate_with_sac.py`: Complete example with SAC training loop

## Quick Start

### 1. Compile the CUDA Library

```bash
cd environments
make clean all
```

This will create `libenvironment.so` that can be loaded by Python.

### 2. Test the Environment

```bash
python environment_wrapper.py
```

This will test the basic functionality of the environment.

### 3. Integrate with SAC Agent

```python
# In your SAC.py file, add these imports
import sys
import os
sys.path.append('environments')  # Add environments directory to path
from environment_wrapper import CUDAEnvironment

# Initialize environment
env = CUDAEnvironment(cols=25, rows=23, Ka=1, num_sims=1000)

# In your training loop, replace the environment interaction:
state = env.reset()
state = torch.tensor(state, dtype=torch.float32)

while not done:
    # Generate action using your SAC policy
    action, log_prob = matrix_policy(state)
    
    # Take step in CUDA environment
    next_state, reward, done, info = env.step(action)
    next_state = torch.tensor(next_state, dtype=torch.float32)
    
    # Store experience in buffer
    buffer.push(state, action, next_state, reward, done)
    
    state = next_state
```

## Environment Interface

### Initialization

```python
env = CUDAEnvironment(
    cols=25,      # Number of columns in codebook (action dimension)
    rows=23,      # Number of rows in codebook
    Ka=1,         # Number of active users
    num_sims=1000 # Number of simulations for evaluation
)
```

### Methods

#### `reset() -> np.ndarray`
Reset the environment and return initial state.

#### `step(action: torch.Tensor) -> Tuple[np.ndarray, float, bool, Dict]`
Take a step in the environment.

**Args:**
- `action`: Action tensor from SAC agent (1D with `cols` elements)

**Returns:**
- `next_state`: Next state as numpy array
- `reward`: Reward for this step
- `done`: Whether episode is finished
- `info`: Additional information (hit_rate, best_hit_rate, step, buffer_size)

#### `get_codebook() -> np.ndarray`
Get the current codebook as a numpy array (shape: `rows × cols`).

#### `get_action_buffer() -> np.ndarray`
Get the current action buffer as a numpy array.

## Key Features

### Persistent Action Buffer

The environment maintains a persistent buffer of actions across episodes. Each action is added as a new row to the codebook, and the buffer automatically manages overflow by shifting older actions out.

### High-Performance Simulation

The CUDA implementation runs parallel simulations on the GPU, making it much faster than CPU-based alternatives. The simulation:
1. Generates random message superpositions
2. Adds Gaussian noise
3. Decodes using greedy algorithm
4. Calculates hit rate based on successful decodings

### Reward Structure

The reward is based on the improvement in hit rate:
```python
reward = current_hit_rate - best_hit_rate
```

This encourages the agent to continuously improve the codebook performance.

## State Representation

The state includes:
- Current hit rate
- Normalized step count
- Buffer fullness
- Best hit rate achieved
- First 6 action components

You can modify the state representation in `environment_wrapper.py` to match your needs.

## Troubleshooting

### CUDA Not Found
If you get CUDA-related errors:
1. Ensure CUDA is installed: `nvcc --version`
2. Check GPU availability: `nvidia-smi`
3. Make sure `nvcc` is in your PATH

### Compilation Errors
If compilation fails:
1. Check CUDA installation
2. Ensure you have write permissions in the environments directory
3. Try running `make clean` before `make all`

### Library Loading Errors
If Python can't load the library:
1. Check that `libenvironment.so` was created
2. Ensure the library is in the correct directory
3. Check file permissions

## Example Usage

See `sac_integration_example.py` for a complete example of how to integrate with your SAC agent.

## Performance Notes

- The environment is designed for high-throughput training
- GPU memory usage scales with `num_sims`
- Consider reducing `num_sims` if you run into memory issues
- The action buffer persists across episodes, so the codebook builds up over time

## Customization

You can customize:
- State representation in `environment_wrapper.py`
- Reward function in `environment_wrapper.py`
- Simulation parameters in `environment.cu`
- Buffer management in `environment.cu` 
# CUDA SAC Integration Setup Guide

This guide will help you set up and run the CUDA SAC integration for codebook generation.

## Prerequisites

### System Requirements
- **OS**: Linux (Ubuntu 20.04+ recommended)
- **GPU**: NVIDIA GPU with CUDA support
- **CUDA**: Version 11.0 or higher
- **Python**: 3.8 or higher

### Hardware Check
```bash
# Check CUDA installation
nvcc --version

# Check GPU availability
nvidia-smi

# Check Python version
python3 --version
```

## Installation Steps

### 1. Create Virtual Environment
```bash
# Create virtual environment
python3 -m venv cuda_sac_env

# Activate virtual environment
source cuda_sac_env/bin/activate
```

### 2. Install PyTorch with CUDA Support
```bash
# Install PyTorch with CUDA 12.1 support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### 3. Install Additional Dependencies
```bash
# Install other required packages
pip install numpy matplotlib gym gymnasium
```

### 4. Compile CUDA Library
```bash
# Navigate to the cuda_sac_integration directory
cd cuda_sac_integration

# Compile the CUDA library
make clean all
```

## Verification

### Run Test Suite
```bash
# Run comprehensive test suite
python test_setup.py
```

Expected output:
```
🎉 ALL TESTS PASSED! Your setup is ready for CUDA SAC training.
```

### Test Basic Functionality
```bash
# Test environment wrapper
python environment_wrapper.py

# Test SAC integration example
python sac_integration_example.py
```

## Usage

### Quick Start
```bash
# Run the complete SAC training with CUDA environment
python integrate_with_sac.py
```

### Custom Training
```python
from environment_wrapper import CUDAEnvironment
from integrate_with_sac import CodewordPolicy, QNetwork, ReplayBuffer

# Initialize environment
env = CUDAEnvironment(cols=25, rows=23, Ka=1, num_sims=1000)

# Initialize SAC components
policy = CodewordPolicy(state_dim=10, action_dim=25, hidden_dim=50)
q1 = QNetwork(state_dim=10, action_dim=25, hidden_dim=50)
q2 = QNetwork(state_dim=10, action_dim=25, hidden_dim=50)
buffer = ReplayBuffer(capacity=10000, batch_size=32)

# Training loop
for episode in range(1000):
    state = env.reset()
    done = False
    
    while not done:
        action, log_prob = policy(state)
        next_state, reward, done, info = env.step(action)
        
        # Store experience and train...
```

## Configuration

### Environment Parameters
- `cols`: Number of columns in codebook (action dimension)
- `rows`: Number of rows in codebook
- `Ka`: Number of active users
- `num_sims`: Number of simulations for evaluation

### SAC Hyperparameters
- `state_dim`: State dimension (default: 10)
- `action_dim`: Action dimension (default: 25)
- `hidden_dim`: Hidden layer dimension (default: 50)
- `lr`: Learning rate (default: 0.0001)
- `gamma`: Discount factor (default: 0.99)
- `tau`: Soft update coefficient (default: 0.005)
- `temp`: Temperature parameter (default: 0.32)

## Troubleshooting

### Common Issues

1. **CUDA Not Found**
   ```bash
   # Ensure CUDA is in PATH
   export PATH=/usr/local/cuda/bin:$PATH
   export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
   ```

2. **Compilation Errors**
   ```bash
   # Clean and recompile
   make clean
   make all
   ```

3. **Library Loading Errors**
   ```bash
   # Check library exists
   ls -la libenvironment.so
   
   # Check permissions
   chmod +x libenvironment.so
   ```

4. **PyTorch CUDA Issues**
   ```bash
   # Verify PyTorch CUDA support
   python -c "import torch; print(torch.cuda.is_available())"
   ```

### Performance Optimization

1. **GPU Memory**: Adjust `num_sims` based on GPU memory
2. **Batch Size**: Increase for faster training (if memory allows)
3. **Simulation Count**: Higher values give more accurate evaluation

## File Structure

```
cuda_sac_integration/
├── environment.cu              # CUDA environment implementation
├── environment_wrapper.py      # Python wrapper for CUDA environment
├── SAC.py                     # SAC agent implementation
├── integrate_with_sac.py      # Complete integration example
├── sac_integration_example.py # Basic integration example
├── test_setup.py              # Comprehensive test suite
├── requirements.txt            # Python dependencies
├── Makefile                   # CUDA compilation script
├── libenvironment.so          # Compiled CUDA library
└── SETUP_GUIDE.md            # This guide
```

## Performance Notes

- The CUDA environment provides significant speedup over CPU-based alternatives
- GPU memory usage scales with `num_sims` parameter
- The action buffer persists across episodes, building the codebook over time
- Monitor GPU memory usage with `nvidia-smi` during training

## Support

If you encounter issues:
1. Run `python test_setup.py` to identify the problem
2. Check the troubleshooting section above
3. Verify CUDA and PyTorch installations
4. Ensure all dependencies are installed in the virtual environment 
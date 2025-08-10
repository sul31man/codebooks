import torch
import env2

def test_env_reset():
    print("=== Testing GPU Environment Reset ===")
    
    # Test different batch sizes
    batch_sizes = [1, 4, 16, 100]
    
    for batch_size in batch_sizes:
        print(f"\nBatch size: {batch_size}")
        
        # Reset environments
        states = env2.env_reset(batch_size)
        
        print(f"States shape: {states.shape}")
        print(f"States device: {states.device}")
        print(f"States dtype: {states.dtype}")
        
        # Check the range is correct [-0.05, 0.05]
        min_val = states.min().item()
        max_val = states.max().item()
        print(f"Value range: [{min_val:.4f}, {max_val:.4f}]")
        
        # Show first few states
        if batch_size <= 4:
            print(f"States:\n{states}")
        else:
            print(f"First 3 states:\n{states[:3]}")
        
        # Verify all values are in correct range
        in_range = torch.all((states >= -0.05) & (states <= 0.05))
        print(f"All values in [-0.05, 0.05]: {in_range.item()}")

if __name__ == "__main__":
    test_env_reset()

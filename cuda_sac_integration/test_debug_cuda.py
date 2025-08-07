#!/usr/bin/env python3
"""
Comprehensive CUDA debug test script.
Tests each hypothesis for the CUDA memory errors systematically.
"""

import sys
import os
import ctypes
import numpy as np
import torch
import time

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_library_loading():
    """Test 1: Basic library loading and function availability."""
    print("🔍 Test 1: Library Loading")
    print("-" * 50)
    
    try:
        # Load debug library
        lib_path = "./libenvironment_debug.so"
        if not os.path.exists(lib_path):
            print(f"❌ Debug library not found: {lib_path}")
            return False
            
        lib = ctypes.CDLL(lib_path)
        print(f"✅ Debug library loaded: {lib_path}")
        
        # Test function availability
        functions = [
            'initialize_buffer_with_random',
            'add_action_to_buffer', 
            'get_codebook',
            'run_simulation',
            'clear_action_buffer'
        ]
        
        for func_name in functions:
            try:
                func = getattr(lib, func_name)
                print(f"✅ Function found: {func_name}")
            except AttributeError:
                print(f"❌ Function missing: {func_name}")
                return False
        
        return lib
        
    except Exception as e:
        print(f"❌ Library loading failed: {e}")
        return False

def test_cuda_device_availability():
    """Test 2: CUDA device and PyTorch CUDA availability."""
    print("\n🔍 Test 2: CUDA Device Availability")
    print("-" * 50)
    
    # Check PyTorch CUDA
    print(f"PyTorch CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU device: {torch.cuda.get_device_name()}")
        print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory // 1024**3} GB")
        
        # Test basic tensor creation
        try:
            test_tensor = torch.randn(10, device='cuda')
            print(f"✅ PyTorch CUDA tensor creation successful")
            del test_tensor
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"❌ PyTorch CUDA tensor creation failed: {e}")
            return False
    else:
        print("❌ PyTorch CUDA not available")
        return False
    
    return True

def test_buffer_initialization(lib):
    """Test 3: Buffer initialization with debug output."""
    print("\n🔍 Test 3: Buffer Initialization")
    print("-" * 50)
    
    try:
        # Set up function signatures
        lib.initialize_buffer_with_random.argtypes = [ctypes.c_int, ctypes.c_int]
        lib.initialize_buffer_with_random.restype = None
        
        lib.clear_action_buffer.argtypes = []
        lib.clear_action_buffer.restype = None
        
        # Clear any existing buffer
        print("Clearing existing buffer...")
        lib.clear_action_buffer()
        
        # Test buffer initialization
        rows, cols = 512, 1024
        print(f"Initializing buffer: {rows} x {cols}")
        lib.initialize_buffer_with_random(rows, cols)
        print("✅ Buffer initialization completed")
        
        return True
        
    except Exception as e:
        print(f"❌ Buffer initialization failed: {e}")
        return False

def test_action_tensor_conversion(lib):
    """Test 4: Tensor to C array conversion (your hypothesis #1)."""
    print("\n🔍 Test 4: Tensor Pointer Conversion")
    print("-" * 50)
    
    try:
        # Set up function signature
        lib.add_action_to_buffer.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.c_int]
        lib.add_action_to_buffer.restype = None
        
        # Test different tensor types and conversions
        test_cases = [
            ("CPU Float32 Contiguous", torch.randn(1024, dtype=torch.float32)),
            ("CPU Float32 Non-contiguous", torch.randn(1024, dtype=torch.float32).t().t()),
            ("CPU Float64", torch.randn(1024, dtype=torch.float64)),
        ]
        
        if torch.cuda.is_available():
            test_cases.extend([
                ("CUDA Float32", torch.randn(1024, dtype=torch.float32, device='cuda')),
                ("CUDA Float64", torch.randn(1024, dtype=torch.float64, device='cuda')),
            ])
        
        for case_name, tensor in test_cases:
            print(f"\nTesting: {case_name}")
            print(f"  Original - Shape: {tensor.shape}, Dtype: {tensor.dtype}, Device: {tensor.device}")
            print(f"  Contiguous: {tensor.is_contiguous()}")
            
            try:
                # Safe conversion process
                if tensor.is_cuda:
                    tensor = tensor.cpu()
                    print(f"  Moved to CPU")
                
                if tensor.dtype != torch.float32:
                    tensor = tensor.to(dtype=torch.float32)
                    print(f"  Converted to float32")
                
                if not tensor.is_contiguous():
                    tensor = tensor.contiguous()
                    print(f"  Made contiguous")
                
                # Convert to numpy with copy
                action_np = tensor.detach().numpy().copy()
                print(f"  NumPy - Shape: {action_np.shape}, Dtype: {action_np.dtype}")
                print(f"  C_contiguous: {action_np.flags.c_contiguous}")
                
                # Test for invalid values
                if not np.all(np.isfinite(action_np)):
                    print(f"  ❌ Contains NaN/inf values")
                    continue
                
                # Convert to C array
                action_array = (ctypes.c_float * len(action_np))(*action_np)
                print(f"  ✅ C array created successfully")
                
                # Call C function
                lib.add_action_to_buffer(action_array, len(action_np))
                print(f"  ✅ Buffer addition successful")
                
            except Exception as e:
                print(f"  ❌ Conversion failed: {e}")
                
        return True
        
    except Exception as e:
        print(f"❌ Tensor conversion test failed: {e}")
        return False

def test_cuda_simulation(lib):
    """Test 5: CUDA simulation with comprehensive error checking."""
    print("\n🔍 Test 5: CUDA Simulation")
    print("-" * 50)
    
    try:
        # Set up function signatures
        lib.run_simulation.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_int)]
        lib.run_simulation.restype = None
        
        lib.get_codebook.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_int]
        lib.get_codebook.restype = None
        
        # Test parameters
        Ka = 1
        num_sims = 100  # Small number for testing
        hit_rate = ctypes.c_int(0)
        
        print(f"Running simulation: Ka={Ka}, num_sims={num_sims}")
        print("This will show detailed debug output from CUDA code...")
        
        # Run simulation
        lib.run_simulation(Ka, num_sims, ctypes.byref(hit_rate))
        
        print(f"✅ Simulation completed: hit_rate={hit_rate.value}")
        
        # Test codebook retrieval
        rows, cols = 512, 1024
        codebook_size = rows * cols
        codebook_array = (ctypes.c_float * codebook_size)()
        
        print(f"Retrieving codebook: {rows} x {cols}")
        lib.get_codebook(codebook_array, rows, cols)
        
        # Convert to numpy for analysis
        codebook_np = np.array(codebook_array, dtype=np.float32).reshape(rows, cols)
        print(f"✅ Codebook retrieved: shape={codebook_np.shape}")
        print(f"  Stats - Min: {codebook_np.min():.4f}, Max: {codebook_np.max():.4f}, Mean: {codebook_np.mean():.4f}")
        print(f"  Finite values: {np.sum(np.isfinite(codebook_np))}/{codebook_np.size}")
        
        return True
        
    except Exception as e:
        print(f"❌ CUDA simulation failed: {e}")
        return False

def test_pytorch_cuda_conflict(lib):
    """Test 6: PyTorch + Custom CUDA context conflict."""
    print("\n🔍 Test 6: PyTorch CUDA Context Conflict")
    print("-" * 50)
    
    if not torch.cuda.is_available():
        print("❌ CUDA not available, skipping conflict test")
        return False
    
    try:
        # Create PyTorch CUDA tensors
        print("Creating PyTorch CUDA tensors...")
        tensor1 = torch.randn(1000, 1000, device='cuda')
        tensor2 = torch.randn(1000, 1000, device='cuda')
        result = torch.matmul(tensor1, tensor2)
        print(f"✅ PyTorch CUDA operation successful: {result.shape}")
        
        # Run custom CUDA simulation
        print("Running custom CUDA simulation...")
        Ka, num_sims = 1, 50
        hit_rate = ctypes.c_int(0)
        lib.run_simulation(Ka, num_sims, ctypes.byref(hit_rate))
        print(f"✅ Custom CUDA operation successful: hit_rate={hit_rate.value}")
        
        # Try PyTorch operation again
        print("Testing PyTorch CUDA after custom CUDA...")
        tensor3 = torch.randn(1000, 1000, device='cuda')
        result2 = torch.matmul(result, tensor3)
        print(f"✅ PyTorch CUDA still works: {result2.shape}")
        
        # Clean up
        del tensor1, tensor2, tensor3, result, result2
        torch.cuda.empty_cache()
        
        return True
        
    except Exception as e:
        print(f"❌ CUDA context conflict detected: {e}")
        return False

def main():
    """Run comprehensive CUDA debugging tests."""
    print("🚀 CUDA Debugging Test Suite")
    print("=" * 60)
    print("Testing your hypotheses:")
    print("1. CPU pointer + CUDA kernel issues")
    print("2. Unaligned memory or wrong dtype") 
    print("3. Missing CUDA API error checking")
    print("4. Out-of-bounds memory access")
    print("5. PyTorch CUDA context conflicts")
    print("=" * 60)
    
    # Test 1: Library loading
    lib = test_library_loading()
    if not lib:
        print("\n❌ Cannot proceed without library")
        return
    
    # Test 2: CUDA device
    if not test_cuda_device_availability():
        print("\n❌ CUDA device issues detected")
        return
    
    # Test 3: Buffer initialization
    if not test_buffer_initialization(lib):
        print("\n❌ Buffer initialization issues detected")
        return
    
    # Test 4: Tensor conversion (Hypothesis #1)
    if not test_action_tensor_conversion(lib):
        print("\n❌ Tensor conversion issues detected")
        return
    
    # Test 5: CUDA simulation (Hypotheses #2, #3, #4)
    print("\n" + "=" * 60)
    print("🔥 CRITICAL TEST: Running CUDA simulation with debug output")
    print("This will reveal memory access, alignment, and API errors...")
    print("=" * 60)
    
    if not test_cuda_simulation(lib):
        print("\n❌ CUDA simulation issues detected")
        return
    
    # Test 6: PyTorch conflict (Hypothesis #5)
    if not test_pytorch_cuda_conflict(lib):
        print("\n❌ PyTorch CUDA context conflict detected")
        return
    
    print("\n" + "=" * 60)
    print("🎉 ALL TESTS PASSED!")
    print("Your CUDA implementation appears to be working correctly.")
    print("=" * 60)

if __name__ == "__main__":
    main() 
#include <stdio.h>
#include <cuda_runtime.h>

// CUDA kernel function - runs on GPU (device)
// __global__ means this function can be called from host and runs on device
__global__ void hello_world_kernel(int *d_data, int n) {
    // Get thread index - each thread has a unique ID
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Make sure we don't go out of bounds
    if (idx < n) {
        // Each thread modifies one element
        d_data[idx] = idx * idx; // Square the index
        
        // Print from GPU (limited threads to avoid spam)
        if (idx < 10) {
            printf("Hello from GPU thread %d! Setting d_data[%d] = %d\n", 
                   idx, idx, d_data[idx]);
        }
    }
}

// Host function - runs on CPU
void check_cuda_error(cudaError_t error, const char* message) {
    if (error != cudaSuccess) {
        fprintf(stderr, "CUDA Error: %s - %s\n", message, cudaGetErrorString(error));
        exit(1);
    }
}

int main() {
    printf("=== CUDA Hello World Tutorial ===\n\n");
    
    // 1. SETUP: Define problem size and allocate host memory
    const int N = 16;  // Number of elements
    const int size = N * sizeof(int);
    
    // Host (CPU) memory allocation
    int *h_data = (int*)malloc(size);
    int *h_result = (int*)malloc(size);
    
    // Initialize host data
    for (int i = 0; i < N; i++) {
        h_data[i] = i;
    }
    
    printf("1. Host data initialized:\n");
    for (int i = 0; i < N; i++) {
        printf("h_data[%d] = %d  ", i, h_data[i]);
    }
    printf("\n\n");
    
    // 2. DEVICE MEMORY: Allocate memory on GPU
    int *d_data;
    cudaError_t error;
    
    // cudaMalloc allocates memory on GPU (device)
    error = cudaMalloc((void**)&d_data, size);
    check_cuda_error(error, "cudaMalloc failed");
    
    // 3. COPY DATA: Host to Device
    error = cudaMemcpy(d_data, h_data, size, cudaMemcpyHostToDevice);
    check_cuda_error(error, "cudaMemcpy Host to Device failed");
    
    printf("2. Data copied to GPU\n\n");
    
    // 4. KERNEL LAUNCH: This is the key CUDA syntax!
    // <<<blocks, threads_per_block>>>
    int threads_per_block = 8;
    int blocks = (N + threads_per_block - 1) / threads_per_block; // Ceiling division
    
    printf("3. Launching kernel with:\n");
    printf("   - Blocks: %d\n", blocks);
    printf("   - Threads per block: %d\n", threads_per_block);
    printf("   - Total threads: %d\n\n", blocks * threads_per_block);
    
    // KERNEL LAUNCH SYNTAX: function_name<<<blocks, threads>>>(parameters)
    hello_world_kernel<<<blocks, threads_per_block>>>(d_data, N);
    
    // 5. SYNCHRONIZATION: Wait for GPU to finish
    error = cudaDeviceSynchronize();
    check_cuda_error(error, "cudaDeviceSynchronize failed");
    
    printf("4. Kernel execution completed\n\n");
    
    // 6. COPY RESULTS: Device to Host
    error = cudaMemcpy(h_result, d_data, size, cudaMemcpyDeviceToHost);
    check_cuda_error(error, "cudaMemcpy Device to Host failed");
    
    // 7. VERIFY RESULTS
    printf("5. Results copied back to host:\n");
    for (int i = 0; i < N; i++) {
        printf("h_result[%d] = %d  ", i, h_result[i]);
    }
    printf("\n\n");
    
    // 8. CLEANUP: Free memory
    free(h_data);
    free(h_result);
    cudaFree(d_data);
    
    printf("6. Memory cleaned up successfully!\n");
    
    // 9. BONUS: Show device properties
    int device_count;
    cudaGetDeviceCount(&device_count);
    printf("\n=== GPU Information ===\n");
    printf("Number of CUDA devices: %d\n", device_count);
    
    if (device_count > 0) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, 0);
        printf("Device 0: %s\n", prop.name);
        printf("Compute capability: %d.%d\n", prop.major, prop.minor);
        printf("Max threads per block: %d\n", prop.maxThreadsPerBlock);
        printf("Max blocks per grid: %d\n", prop.maxGridSize[0]);
    }
    
    return 0;
} 
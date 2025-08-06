#include <stdio.h>
#include <cuda_runtime.h>

// CUDA kernel function for vector addition
__global__ void vectorAdd(float *a, float *b, float *c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

int main() {
    const int N = 1024;
    const int size = N * sizeof(float);
    
    // Host vectors
    float *h_a, *h_b, *h_c;
    // Device vectors
    float *d_a, *d_b, *d_c;
    
    // Allocate host memory
    h_a = (float*)malloc(size);
    h_b = (float*)malloc(size);
    h_c = (float*)malloc(size);
    
    // Initialize host vectors
    for (int i = 0; i < N; i++) {
        h_a[i] = i * 1.0f;
        h_b[i] = i * 2.0f;
    }
    
    // Allocate device memory
    cudaMalloc((void**)&d_a, size);
    cudaMalloc((void**)&d_b, size);
    cudaMalloc((void**)&d_c, size);
    
    // Copy data from host to device
    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);
    
    // Launch kernel with 1D grid
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    
    printf("CUDA Setup Test - Vector Addition\n");
    printf("Vector size: %d elements\n", N);
    printf("Grid size: %d blocks\n", blocksPerGrid);
    printf("Block size: %d threads\n", threadsPerBlock);
    printf("\nLaunching kernel...\n");
    
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, N);
    
    // Check for kernel errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA kernel error: %s\n", cudaGetErrorString(err));
        return -1;
    }
    
    // Wait for kernel to finish
    cudaDeviceSynchronize();
    
    // Copy result back to host
    cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);
    
    // Verify results
    bool success = true;
    for (int i = 0; i < N; i++) {
        if (abs(h_c[i] - (h_a[i] + h_b[i])) > 1e-5) {
            success = false;
            break;
        }
    }
    
    if (success) {
        printf("SUCCESS: Vector addition completed correctly!\n");
        printf("Sample results:\n");
        for (int i = 0; i < 5; i++) {
            printf("  %.1f + %.1f = %.1f\n", h_a[i], h_b[i], h_c[i]);
        }
    } else {
        printf("ERROR: Vector addition failed!\n");
    }
    
    // Get device properties
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("\nGPU Information:\n");
    printf("  Device name: %s\n", prop.name);
    printf("  Compute capability: %d.%d\n", prop.major, prop.minor);
    printf("  Total global memory: %.1f MB\n", prop.totalGlobalMem / 1024.0 / 1024.0);
    printf("  Multiprocessors: %d\n", prop.multiProcessorCount);
    
    // Cleanup
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    free(h_a);
    free(h_b);
    free(h_c);
    
    printf("\nCUDA test completed successfully!\n");
    return 0;
}

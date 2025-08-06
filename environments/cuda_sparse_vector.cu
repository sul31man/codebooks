#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <math.h>

// CUDA kernel for sparse-dense vector multiplication
__global__ void sparse_dense_kernel(int* indices, float* values, int nnz, 
                                   float* dense_vector, float* result) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < nnz) {
        int sparse_idx = indices[idx];
        result[idx] = values[idx] * dense_vector[sparse_idx];
    }
}

// CUDA kernel for reduction (sum all partial results)
__global__ void reduction_kernel(float* input, float* output, int n) {
    extern __shared__ float sdata[];
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Load data into shared memory
    sdata[tid] = (idx < n) ? input[idx] : 0.0f;
    __syncthreads();
    
    // Reduction in shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    // Write result for this block to global memory
    if (tid == 0) {
        output[blockIdx.x] = sdata[0];
    }
}

void check_cuda_error(cudaError_t error, const char* message) {
    if (error != cudaSuccess) {
        fprintf(stderr, "CUDA Error: %s - %s\n", message, cudaGetErrorString(error));
        exit(1);
    }
}

// Host function to perform sparse-dense dot product on GPU
float cuda_sparse_dense_dot(int* h_indices, float* h_values, int nnz, 
                           float* h_dense, int dense_size) {
    
    // Device memory pointers
    int* d_indices;
    float* d_values;
    float* d_dense;
    float* d_partial_results;
    float* d_final_result;
    
    // Calculate memory sizes
    int indices_size = nnz * sizeof(int);
    int values_size = nnz * sizeof(float);
    int dense_size_bytes = dense_size * sizeof(float);
    int partial_size = nnz * sizeof(float);
    
    // Allocate device memory
    cudaMalloc(&d_indices, indices_size);
    cudaMalloc(&d_values, values_size);
    cudaMalloc(&d_dense, dense_size_bytes);
    cudaMalloc(&d_partial_results, partial_size);
    cudaMalloc(&d_final_result, sizeof(float));
    
    // Copy data to device
    cudaMemcpy(d_indices, h_indices, indices_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_values, h_values, values_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_dense, h_dense, dense_size_bytes, cudaMemcpyHostToDevice);
    
    // Configure kernel launch parameters
    int threads_per_block = 256;
    int blocks = (nnz + threads_per_block - 1) / threads_per_block;
    
    // Launch element-wise multiplication kernel
    sparse_dense_kernel<<<blocks, threads_per_block>>>(
        d_indices, d_values, nnz, d_dense, d_partial_results);
    
    // Configure reduction kernel
    int reduction_blocks = (nnz + threads_per_block - 1) / threads_per_block;
    int shared_mem_size = threads_per_block * sizeof(float);
    
    // Launch reduction kernel
    reduction_kernel<<<reduction_blocks, threads_per_block, shared_mem_size>>>(
        d_partial_results, d_final_result, nnz);
    
    // If we have multiple blocks, we need another reduction
    if (reduction_blocks > 1) {
        float* d_block_results;
        cudaMalloc(&d_block_results, reduction_blocks * sizeof(float));
        
        reduction_kernel<<<1, threads_per_block, shared_mem_size>>>(
            d_final_result, d_block_results, reduction_blocks);
        
        cudaFree(d_block_results);
    }
    
    // Copy result back to host
    float result;
    cudaMemcpy(&result, d_final_result, sizeof(float), cudaMemcpyDeviceToHost);
    
    // Cleanup
    cudaFree(d_indices);
    cudaFree(d_values);
    cudaFree(d_dense);
    cudaFree(d_partial_results);
    cudaFree(d_final_result);
    
    return result;
}

int main() {
    printf("=== CUDA Sparse Vector Example ===\n\n");
    
    // Create test data
    int nnz = 5;
    int dense_size = 10;
    
    // Sparse vector: only elements at indices [1, 3, 5, 7, 9] are non-zero
    int h_indices[] = {1, 3, 5, 7, 9};
    float h_values[] = {2.0f, 4.0f, 6.0f, 8.0f, 10.0f};
    
    // Dense vector: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    float h_dense[10];
    for (int i = 0; i < dense_size; i++) {
        h_dense[i] = (float)i;
    }
    
    // Print input data
    printf("Sparse vector (indices, values):\n");
    for (int i = 0; i < nnz; i++) {
        printf("  [%d] = %.1f\n", h_indices[i], h_values[i]);
    }
    
    printf("\nDense vector:\n");
    for (int i = 0; i < dense_size; i++) {
        printf("  [%d] = %.1f\n", i, h_dense[i]);
    }
    
    // Compute dot product using CUDA
    float cuda_result = cuda_sparse_dense_dot(h_indices, h_values, nnz, h_dense, dense_size);
    
    // Compute reference result on CPU
    float cpu_result = 0.0f;
    for (int i = 0; i < nnz; i++) {
        cpu_result += h_values[i] * h_dense[h_indices[i]];
    }
    
    printf("\nResults:\n");
    printf("CPU result:  %.1f\n", cpu_result);
    printf("CUDA result: %.1f\n", cuda_result);
    printf("Match: %s\n", (fabs(cuda_result - cpu_result) < 1e-6) ? "YES" : "NO");
    
    // Show the calculation breakdown
    printf("\nCalculation breakdown:\n");
    for (int i = 0; i < nnz; i++) {
        printf("  %.1f * %.1f = %.1f\n", 
               h_values[i], h_dense[h_indices[i]], 
               h_values[i] * h_dense[h_indices[i]]);
    }
    
    return 0;
} 
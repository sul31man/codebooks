#include <stdio.h>
#include <cuda_runtime.h>
#include <chrono>

#define BLOCK_SIZE 256
#define VECTOR_SIZE (1024 * 1024 * 16)  // 16M elements

// ============================================================================
// 🎯 YOUR CHALLENGE: IMPLEMENT THESE 4 KERNELS
// ============================================================================

// TASK 1: Perfect Coalescing (EASY)
// Hint: Each thread processes consecutive elements
__global__ void vector_add_coalesced(float *a, float *b, float *c, int n) {
    // TODO: Implement perfect coalescing pattern
    // Each thread should process: a[i] + b[i] = c[i]
    // Where i = blockIdx.x * blockDim.x + threadIdx.x
    
    // YOUR CODE HERE:

    int idx = blockIdx * blockDim + threadIdx; 

    if (idx < n){


        c[idx] = a[idx] + b[idx];
    }
    
}

// TASK 2: Terrible Coalescing (MEDIUM)
// Hint: Each thread processes elements with large stride
__global__ void vector_add_strided(float *a, float *b, float *c, int n, int stride) {
    // TODO: Implement bad coalescing with stride pattern
    // Each thread processes: a[i*stride] + b[i*stride] = c[i*stride]
    // This will have terrible memory access patterns!
    
    // YOUR CODE HERE:

    int idx = blockDim * blockIdx + threadIdx; 

    if (idx < n){


        c[i*stride] = a[i*stride] + b[i*stride];
    }
    
}

// TASK 3: Block-wise Processing (HARD)
// Hint: Each block processes a chunk, but within block threads are coalesced
__global__ void vector_add_blocked(float *a, float *b, float *c, int n, int elements_per_block) {
    // TODO: Each block processes 'elements_per_block' consecutive elements
    // But threads within block should still be coalesced
    // Block 0: elements [0 to elements_per_block-1]
    // Block 1: elements [elements_per_block to 2*elements_per_block-1]
    // etc.
    
    // YOUR CODE HERE:
    int block_start = blockIdx.x * elements_per_block;
    int idx = block_start + threadIdx.x;
    
    // Each thread processes multiple elements to handle elements_per_block
    int elements_per_thread = (elements_per_block + blockDim.x - 1) / blockDim.x;
    
    for (int i = 0; i < elements_per_thread; i++) {
        int global_idx = idx + i * blockDim.x;
        if (global_idx < block_start + elements_per_block && global_idx < n) {
            c[global_idx] = a[global_idx] + b[global_idx];
        }
    }
}



// TASK 4: Shared Memory Optimization (EXPERT)
// Hint: Use shared memory to cache data and improve reuse
__global__ void vector_add_shared_cache(float *a, float *b, float *c, int n) {
    // TODO: Use shared memory to cache portions of arrays
    // Each thread loads multiple elements into shared memory
    // Then processes them with good locality
    
    // YOUR CODE HERE:
    extern __shared__ float shared_mem[];
    float *shared_a = shared_mem;
    float *shared_b = &shared_mem[blockDim.x];
    

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Load data into shared memory with coalesced access
    if (idx < n) {
        shared_a[tid] = a[idx];
        shared_b[tid] = b[idx];
    }
    
    // Synchronize to ensure all threads have loaded their data
    __syncthreads();
    
    // Perform computation using shared memory data
    if (idx < n) {
        c[idx] = shared_a[tid] + shared_b[tid];
    }
}

// ============================================================================
// 🔬 ANALYSIS FRAMEWORK (PROVIDED)
// ============================================================================

float benchmark_kernel(void (*kernel)(float*, float*, float*, int),
                      float *d_a, float *d_b, float *d_c, int n, 
                      const char *name) {
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // Warm up
    int blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    kernel<<<blocks, BLOCK_SIZE>>>(d_a, d_b, d_c, n);
    cudaDeviceSynchronize();
    
    // Benchmark
    cudaEventRecord(start);
    for (int i = 0; i < 100; i++) {
        kernel<<<blocks, BLOCK_SIZE>>>(d_a, d_b, d_c, n);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    float avg_ms = ms / 100.0f;
    
    // Calculate bandwidth
    size_t bytes = 3 * n * sizeof(float);  // Read A, Read B, Write C
    float bandwidth = (bytes / (avg_ms / 1000.0f)) / (1024.0f * 1024.0f * 1024.0f);
    
    printf("%-25s: %8.3f ms, %8.1f GB/s\n", name, avg_ms, bandwidth);
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return avg_ms;
}

// Specialized benchmark for strided kernel
float benchmark_strided_kernel(float *d_a, float *d_b, float *d_c, int n, int stride) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    int effective_n = n / stride;
    int blocks = (effective_n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    // Warm up
    vector_add_strided<<<blocks, BLOCK_SIZE>>>(d_a, d_b, d_c, n, stride);
    cudaDeviceSynchronize();
    
    // Benchmark
    cudaEventRecord(start);
    for (int i = 0; i < 100; i++) {
        vector_add_strided<<<blocks, BLOCK_SIZE>>>(d_a, d_b, d_c, n, stride);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    float avg_ms = ms / 100.0f;
    
    size_t bytes = 3 * effective_n * sizeof(float);
    float bandwidth = (bytes / (avg_ms / 1000.0f)) / (1024.0f * 1024.0f * 1024.0f);
    
    printf("Stride %-18d: %8.3f ms, %8.1f GB/s\n", stride, avg_ms, bandwidth);
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return avg_ms;
}

void verify_result(float *a, float *b, float *c, int n, const char *kernel_name) {
    bool correct = true;
    int errors = 0;
    
    for (int i = 0; i < n && errors < 5; i++) {
        if (abs(c[i] - (a[i] + b[i])) > 1e-5) {
            if (errors == 0) {
                printf("❌ %s FAILED!\n", kernel_name);
            }
            printf("  Error at %d: %.1f + %.1f = %.1f (expected %.1f)\n", 
                   i, a[i], b[i], c[i], a[i] + b[i]);
            errors++;
            correct = false;
        }
    }
    
    if (correct) {
        printf("✅ %s verification PASSED!\n", kernel_name);
    }
}

// ============================================================================
// 🎯 MAIN CHALLENGE
// ============================================================================

int main() {
    printf("🎯 VECTOR COALESCING CHALLENGE\n");
    printf("===============================\n\n");
    
    // Allocate memory
    size_t size = VECTOR_SIZE * sizeof(float);
    float *h_a = (float*)malloc(size);
    float *h_b = (float*)malloc(size);
    float *h_c = (float*)malloc(size);
    float *d_a, *d_b, *d_c;
    
    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c, size);
    
    // Initialize data
    for (int i = 0; i < VECTOR_SIZE; i++) {
        h_a[i] = (float)(i % 100);
        h_b[i] = (float)((i + 50) % 100);
    }
    
    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);
    
    printf("🔬 TESTING COALESCING PATTERNS\n");
    printf("===============================\n");
    printf("Vector size: %d elements (%.1f MB per array)\n\n", 
           VECTOR_SIZE, size / (1024.0f * 1024.0f));
    
    // Test 1: Perfect Coalescing
    printf("📊 TASK 1: Perfect Coalescing\n");
    printf("=============================\n");
    float coalesced_time = benchmark_kernel(vector_add_coalesced, d_a, d_b, d_c, VECTOR_SIZE, "Coalesced Access");
    cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);
    verify_result(h_a, h_b, h_c, VECTOR_SIZE, "Coalesced kernel");
    printf("\n");
    
    // Test 2: Stride Patterns (showing how bad it gets)
    printf("📊 TASK 2: Stride Patterns (Bad Coalescing)\n");
    printf("============================================\n");
    int strides[] = {1, 2, 4, 8, 16, 32, 64};
    for (int i = 0; i < 7; i++) {
        benchmark_strided_kernel(d_a, d_b, d_c, VECTOR_SIZE, strides[i]);
    }
    printf("\n");
    
    // Test 3: Block Processing
    printf("📊 TASK 3: Block-wise Processing\n");
    printf("=================================\n");
    // You'll need to implement this
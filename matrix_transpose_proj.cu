#include <stdio.h>
#include <cuda_runtime.h>
#include <chrono>

#define TILE_SIZE 32

// ============================================================================
// PROJECT PART 1: NAIVE TRANSPOSE (BAD COALESCING)
// ============================================================================

__global__ void naive_transpose(float *input, float *output, int rows, int cols) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (idx < cols && idy < rows) {
        // TODO: Implement naive transpose
        // HINT: output[new_row][new_col] = input[old_row][old_col]
        // Remember: matrix[row][col] = matrix[row * width + col] in 1D array
        
        // YOUR CODE HERE:
        output[idx * rows + idy] = input[idy * cols + idx];
        
        // ANALYSIS: Why is this bad for coalescing?
        // - Reading input: threads in same warp read input[0*cols+0], input[0*cols+1], ... (GOOD!)
        // - Writing output: threads in same warp write output[0*rows+0], output[1*rows+0], ... (BAD!)
        // Writing has stride of 'rows' - terrible coalescing!
    }
}

// ============================================================================
// PROJECT PART 2: SHARED MEMORY TRANSPOSE (GOOD COALESCING)
// ============================================================================

__global__ void shared_memory_transpose(float *input, float *output, int rows, int cols) {
    // Shared memory tile for staging
    __shared__ float tile[TILE_SIZE][TILE_SIZE];
    
    // Calculate global positions
    int x = blockIdx.x * TILE_SIZE + threadIdx.x;
    int y = blockIdx.y * TILE_SIZE + threadIdx.y;
    
    // TODO: Implement shared memory transpose
    
    // STEP 1: Cooperatively load input tile with COALESCED reads
    if (x < cols && y < rows) {
        // YOUR CODE HERE: Load from input into shared memory
        tile[threadIdx.y][threadIdx.x] = input[y * cols + x];
    }
    
    // STEP 2: Synchronize - wait for all threads to finish loading
    // YOUR CODE HERE: Add synchronization
    __syncthreads();
    
    // STEP 3: Calculate transposed positions
    int trans_x = blockIdx.y * TILE_SIZE + threadIdx.x;
    int trans_y = blockIdx.x * TILE_SIZE + threadIdx.y;
    
    // STEP 4: Cooperatively store to output with COALESCED writes
    if (trans_x < rows && trans_y < cols) {
        // YOUR CODE HERE: Store from shared memory to output
        // HINT: Notice the index swap in shared memory access
        output[trans_y * rows + trans_x] = tile[threadIdx.x][threadIdx.y];
    }
}

// ============================================================================
// PROJECT PART 3: OPTIMIZED WITH PADDING (ADVANCED)
// ============================================================================

__global__ void optimized_transpose(float *input, float *output, int rows, int cols) {
    // TODO: Add padding to avoid bank conflicts
    // HINT: __shared__ float tile[TILE_SIZE][TILE_SIZE + 1];
    __shared__ float tile[TILE_SIZE][TILE_SIZE + 1];  // +1 for padding
    
    int x = blockIdx.x * TILE_SIZE + threadIdx.x;
    int y = blockIdx.y * TILE_SIZE + threadIdx.y;
    
    // Load with coalesced reads
    if (x < cols && y < rows) {
        tile[threadIdx.y][threadIdx.x] = input[y * cols + x];
    }
    
    __syncthreads();
    
    // Store with coalesced writes and no bank conflicts
    int trans_x = blockIdx.y * TILE_SIZE + threadIdx.x;
    int trans_y = blockIdx.x * TILE_SIZE + threadIdx.y;
    
    if (trans_x < rows && trans_y < cols) {
        output[trans_y * rows + trans_x] = tile[threadIdx.x][threadIdx.y];
    }
}

// ============================================================================
// TESTING AND BENCHMARKING FRAMEWORK
// ============================================================================

void verify_transpose(float *original, float *transposed, int rows, int cols) {
    bool correct = true;
    int errors = 0;
    
    for (int i = 0; i < rows && errors < 10; i++) {
        for (int j = 0; j < cols && errors < 10; j++) {
            float expected = original[i * cols + j];
            float actual = transposed[j * rows + i];
            
            if (abs(expected - actual) > 1e-5) {
                if (errors == 0) {
                    printf("❌ TRANSPOSE VERIFICATION FAILED!\n");
                    printf("First few errors:\n");
                }
                printf("  [%d][%d]: expected %.1f, got %.1f\n", i, j, expected, actual);
                errors++;
                correct = false;
            }
        }
    }
    
    if (correct) {
        printf("✅ Transpose verification PASSED!\n");
    } else {
        printf("❌ Found %s errors in transpose\n", errors >= 10 ? "10+" : "some");
    }
}

float benchmark_transpose(void (*transpose_func)(float*, float*, int, int),
                         float *d_input, float *d_output, int rows, int cols,
                         const char *method_name) {
    
    // Clear output
    cudaMemset(d_output, 0, rows * cols * sizeof(float));
    
    // Setup timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // Calculate grid dimensions
    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid((cols + TILE_SIZE - 1) / TILE_SIZE, (rows + TILE_SIZE - 1) / TILE_SIZE);
    
    // Warm up
    transpose_func<<<grid, block>>>(d_input, d_output, rows, cols);
    cudaDeviceSynchronize();
    
    // Benchmark
    cudaEventRecord(start);
    for (int i = 0; i < 100; i++) {
        transpose_func<<<grid, block>>>(d_input, d_output, rows, cols);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    float avg_ms = ms / 100.0f;
    
    // Calculate effective bandwidth
    size_t bytes_transferred = 2 * rows * cols * sizeof(float);  // Read + Write
    float bandwidth_gb_s = (bytes_transferred / (avg_ms / 1000.0f)) / (1024.0f * 1024.0f * 1024.0f);
    
    printf("%-25s: %8.3f ms, %8.1f GB/s\n", method_name, avg_ms, bandwidth_gb_s);
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    return avg_ms;
}

void print_coalescing_analysis(int rows, int cols) {
    printf("🔬 MEMORY COALESCING ANALYSIS\n");
    printf("=============================\n");
    printf("Matrix: %d × %d = %d elements (%.1f MB)\n", 
           rows, cols, rows * cols, (rows * cols * sizeof(float)) / (1024.0f * 1024.0f));
    
    printf("\n📖 NAIVE TRANSPOSE PATTERN:\n");
    printf("Input access:  input[0*%d+0], input[0*%d+1], input[0*%d+2], ... (stride 1) ✅\n", cols, cols, cols);
    printf("Output access: output[0*%d+0], output[1*%d+0], output[2*%d+0], ... (stride %d) ❌\n", rows, rows, rows, rows);
    printf("Coalescing efficiency: ~%.1f%% (terrible!)\n", 100.0f / rows);
    
    printf("\n🧠 SHARED MEMORY PATTERN:\n");
    printf("Load phase:  input[y*%d+x] - coalesced reads ✅\n", cols);
    printf("Store phase: output[y*%d+x] - coalesced writes ✅\n", rows);
    printf("Coalescing efficiency: ~100%% (excellent!)\n");
    
    printf("\n⚡ PERFORMANCE EXPECTATIONS:\n");
    printf("Shared memory should be %.1fx faster than naive\n", (float)rows / 32.0f);
    printf("With optimizations, can reach 80-90%% of memory bandwidth\n\n");
}

// ============================================================================
// PROJECT MAIN - YOUR ASSIGNMENT
// ============================================================================

int main() {
    printf("🎯 MATRIX TRANSPOSE COALESCING PROJECT\n");
    printf("======================================\n\n");
    
    // Test different matrix sizes to see scaling
    int test_sizes[][2] = {
        {1024, 1024},   // Square matrix
        {2048, 1024},   // Rectangular matrix  
        {4096, 2048},   // Large matrix
    };
    
    int num_tests = sizeof(test_sizes) / sizeof(test_sizes[0]);
    
    for (int test = 0; test < num_tests; test++) {
        int rows = test_sizes[test][0];
        int cols = test_sizes[test][1];
        
        printf("🔧 TEST %d: %d × %d MATRIX\n", test + 1, rows, cols);
        printf("=========================\n");
        
        size_t size = rows * cols * sizeof(float);
        
        // Allocate memory
        float *h_input = (float*)malloc(size);
        float *h_output = (float*)malloc(size);
        float *d_input, *d_output;
        cudaMalloc(&d_input, size);
        cudaMalloc(&d_output, size);
        
        // Initialize test matrix
        for (int i = 0; i < rows * cols; i++) {
            h_input[i] = (float)(i % 100);  // Pattern to make verification easy
        }
        cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice);
        
        print_coalescing_analysis(rows, cols);
        
        // Benchmark all methods
        printf("📊 PERFORMANCE RESULTS:\n");
        printf("=======================\n");
        
        float naive_time = benchmark_transpose(naive_transpose, d_input, d_output, rows, cols, "Naive Transpose");
        
        // Verify naive result
        cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost);
        verify_transpose(h_input, h_output, rows, cols);
        printf("\n");
        
        float shared_time = benchmark_transpose(shared_memory_transpose, d_input, d_output, rows, cols, "Shared Memory");
        
        // Verify shared memory result
        cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost);
        verify_transpose(h_input, h_output, rows, cols);
        printf("\n");
        
        float optimized_time = benchmark_transpose(optimized_transpose, d_input, d_output, rows, cols, "Optimized (Padded)");
        
        // Verify optimized result
        cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost);
        verify_transpose(h_input, h_output, rows, cols);
        printf("\n");
        
        // Performance summary
        printf("🏆 SPEEDUP ANALYSIS:\n");
        printf("====================\n");
        printf("Shared vs Naive:    %.2fx faster\n", naive_time / shared_time);
        printf("Optimized vs Naive: %.2fx faster\n", naive_time / optimized_time);
        printf("Optimized vs Shared: %.2fx faster\n", shared_time / optimized_time);
        
        // Cleanup
        free(h_input);
        free(h_output);
        cudaFree(d_input);
        cudaFree(d_output);
        
        if (test < num_tests - 1) {
            printf("\n==================================================\n\n");
        }
    }
    
    printf("\n🎓 PROJECT COMPLETED!\n");
    printf("=====================\n");
    printf("You've successfully implemented and optimized matrix transpose!\n");
    printf("Key lessons learned:\n");
    printf("✅ Memory access patterns dramatically affect performance\n");
    printf("✅ Shared memory staging can fix coalescing problems\n");
    printf("✅ Small optimizations (padding) can provide extra speedup\n");
    printf("✅ Always measure and verify your optimizations!\n\n");
    
    printf("🚀 NEXT CHALLENGES:\n");
    printf("==================\n");
    printf("1. Try different tile sizes (16×16, 64×64)\n");
    printf("2. Implement matrix multiplication with good coalescing\n");
    printf("3. Add texture memory version for comparison\n");
    printf("4. Profile with nvprof to see detailed metrics\n");
    printf("5. Implement 3D matrix transpose\n");
    
    return 0;
}



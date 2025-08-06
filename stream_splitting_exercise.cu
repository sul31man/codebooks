#include <stdio.h>
#include <cuda_runtime.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

/*
 * STREAM SPLITTING PRACTICE
 * =========================
 * 
 * COMPILATION:
 * nvcc -O3 -o stream_splitting stream_splitting_exercise.cu
 * ./stream_splitting
 * 
 * YOUR MISSION:
 * ------------
 * Implement a stream splitting pipeline that processes a large dataset by:
 * 1. Splitting data into multiple chunks
 * 2. Using different streams for each chunk
 * 3. Overlapping H2D transfer, computation, and D2H transfer
 * 4. Measuring performance gains vs single stream
 * 
 * REQUIREMENTS:
 * - Use at least 4 streams
 * - Process chunks in parallel
 * - Use pinned memory for async transfers
 * - Time and compare single vs multi-stream approaches
 * - Implement proper cleanup
 */

// =============================================================================
// HELPER FUNCTIONS (PROVIDED FOR YOU)
// =============================================================================

void checkCudaError(cudaError_t error, const char* message) {
    if (error != cudaSuccess) {
        printf("CUDA Error: %s - %s\n", message, cudaGetErrorString(error));
        exit(1);
    }
}

// Simple CPU work simulation for demonstrating overlap
void cpuWork(int milliseconds) {
    clock_t start = clock();
    while ((clock() - start) * 1000.0 / CLOCKS_PER_SEC < milliseconds) {
        volatile float dummy = 0;
        for (int i = 0; i < 50000; i++) {
            dummy += sinf(i * 0.001f);
        }
    }
}

// =============================================================================
// CUDA KERNELS (PROVIDED FOR YOU)
// =============================================================================

// Simple computation kernel - adds value to each element and applies math functions
__global__ void processDataKernel(float* data, int n, float add_value, int iterations) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float value = data[idx] + add_value;
        
        // Simulate some computation work
        for (int i = 0; i < iterations; i++) {
            value = sinf(value * 1.1f) + cosf(value * 0.9f) + sqrtf(fabsf(value));
        }
        
        data[idx] = value;
    }
}

// Matrix multiplication kernel for heavier computation
__global__ void heavyComputeKernel(float* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float result = data[idx];
        
        // Simulate heavy computation
        for (int i = 0; i < 1000; i++) {
            result = fmaf(result, 1.001f, 0.5f);  // Fused multiply-add
            result = tanhf(result * 0.1f);
        }
        
        data[idx] = result;
    }
}

// Reduction kernel to sum all elements (useful for verification)
__global__ void reductionKernel(float* input, float* output, int n) {
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

// =============================================================================
// UTILITY FUNCTIONS (PROVIDED FOR YOU)
// =============================================================================

// Initialize data with some pattern
void initializeData(float* data, int size, float base_value) {
    for (int i = 0; i < size; i++) {
        data[i] = base_value + (float)i * 0.001f;
    }
}

// Verify results are reasonable (not NaN/Inf)
bool verifyResults(float* data, int size) {
    for (int i = 0; i < size; i++) {
        if (isnan(data[i]) || isinf(data[i])) {
            printf("ERROR: Invalid result at index %d: %f\n", i, data[i]);
            return false;
        }
    }
    return true;
}

// Calculate simple checksum for verification
float calculateChecksum(float* data, int size) {
    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        sum += data[i];
    }
    return sum;
}

// Print performance statistics
void printPerformanceStats(const char* method, float time_ms, int data_size, int num_streams) {
    float throughput_mb_s = (data_size * sizeof(float) * 2) / (time_ms / 1000.0f) / (1024.0f * 1024.0f);
    printf("\n=== %s PERFORMANCE ===\n", method);
    printf("Time:       %.2f ms\n", time_ms);
    printf("Throughput: %.1f MB/s\n", throughput_mb_s);
    printf("Streams:    %d\n", num_streams);
    printf("Data size:  %.1f MB\n", (data_size * sizeof(float)) / (1024.0f * 1024.0f));
}

// =============================================================================
// STREAM MANAGEMENT HELPERS (PROVIDED FOR YOU)
// =============================================================================

// Create multiple CUDA streams
cudaStream_t* createStreams(int num_streams) {
    cudaStream_t* streams = (cudaStream_t*)malloc(num_streams * sizeof(cudaStream_t));
    for (int i = 0; i < num_streams; i++) {
        checkCudaError(cudaStreamCreate(&streams[i]), "Failed to create stream");
    }
    return streams;
}

// Destroy multiple CUDA streams
void destroyStreams(cudaStream_t* streams, int num_streams) {
    for (int i = 0; i < num_streams; i++) {
        cudaStreamDestroy(streams[i]);
    }
    free(streams);
}

// Create multiple CUDA events for timing
cudaEvent_t* createEvents(int num_events) {
    cudaEvent_t* events = (cudaEvent_t*)malloc(num_events * sizeof(cudaEvent_t));
    for (int i = 0; i < num_events; i++) {
        checkCudaError(cudaEventCreate(&events[i]), "Failed to create event");
    }
    return events;
}

// Destroy multiple CUDA events
void destroyEvents(cudaEvent_t* events, int num_events) {
    for (int i = 0; i < num_events; i++) {
        cudaEventDestroy(events[i]);
    }
    free(events);
}

// =============================================================================
// CONSTANTS FOR YOUR IMPLEMENTATION
// =============================================================================

// Suggested problem sizes - feel free to adjust
const int TOTAL_ELEMENTS = 4 * 1024 * 1024;  // 4M floats = 16MB
const int NUM_STREAMS = 4;                    // Number of parallel streams
const int THREADS_PER_BLOCK = 256;           // CUDA block size
const int COMPUTE_ITERATIONS = 500;          // Iterations for computation kernel

// =============================================================================
// YOUR IMPLEMENTATION AREA
// =============================================================================

/*
 * TODO 1: SINGLE STREAM BASELINE
 * ==============================
 * Implement a function that processes the entire dataset using a single stream.
 * This will serve as your baseline for comparison.
 * 
 * Steps:
 * 1. Allocate pinned host memory and device memory
 * 2. Initialize input data
 * 3. Create timing events
 * 4. Record start time
 * 5. Transfer H2D (Host to Device)
 * 6. Launch computation kernel
 * 7. Transfer D2H (Device to Host)
 * 8. Record end time and calculate elapsed time
 * 9. Verify results and print performance stats
 * 10. Clean up memory
 */

// TODO: Implement this function
float singleStreamBaseline() {
    printf("\n🔄 SINGLE STREAM BASELINE\n");
    printf("========================\n");
    
    // YOUR CODE HERE
    // Hint: Use processDataKernel with COMPUTE_ITERATIONS
    // Return the elapsed time in milliseconds
    
    return 0.0f; // Replace with actual timing
}

/*
 * TODO 2: MULTI-STREAM IMPLEMENTATION
 * ===================================
 * Implement stream splitting where you divide the data into chunks and
 * process each chunk in a separate stream for maximum parallelism.
 * 
 * Steps:
 * 1. Calculate chunk size (TOTAL_ELEMENTS / NUM_STREAMS)
 * 2. Allocate pinned host memory and device memory for each chunk
 * 3. Create NUM_STREAMS streams and timing events
 * 4. Initialize all input chunks
 * 5. Record start time
 * 6. Launch all H2D transfers simultaneously (different streams)
 * 7. Launch all computation kernels (different streams)
 * 8. Launch all D2H transfers simultaneously (different streams)
 * 9. Synchronize all streams
 * 10. Record end time and calculate elapsed time
 * 11. Verify results and print performance stats
 * 12. Clean up all memory and streams
 */

// TODO: Implement this function
float multiStreamImplementation() {
    printf("\n⚡ MULTI-STREAM IMPLEMENTATION\n");
    printf("=============================\n");
    
    // YOUR CODE HERE
    // Hint: Use arrays to manage multiple buffers and streams
    // Process each chunk: H2D -> Compute -> D2H in parallel
    // Return the elapsed time in milliseconds
    
    return 0.0f; // Replace with actual timing
}

/*
 * TODO 3: ADVANCED PIPELINED STREAMS (BONUS)
 * ===========================================
 * Implement a more advanced pipeline where you overlap different stages:
 * - While chunk N is being computed, chunk N+1 is being transferred H2D
 * - While chunk N is being transferred D2H, chunk N+1 is being computed
 * 
 * This is more complex but can achieve even better performance!
 */

// TODO: Implement this function (BONUS)
float pipelinedStreamsImplementation() {
    printf("\n🚀 PIPELINED STREAMS (BONUS)\n");
    printf("===========================\n");
    
    // YOUR CODE HERE
    // Hint: Use events to coordinate between pipeline stages
    // This is challenging but very rewarding!
    
    return 0.0f; // Replace with actual timing
}

/*
 * TODO 4: PERFORMANCE COMPARISON
 * ==============================
 * Compare all your implementations and calculate speedups
 */

void comparePerformance(float single_time, float multi_time, float pipeline_time) {
    printf("\n📊 PERFORMANCE COMPARISON\n");
    printf("========================\n");
    
    if (single_time > 0) {
        printf("Single Stream:    %.2f ms\n", single_time);
    }
    
    if (multi_time > 0) {
        printf("Multi-Stream:     %.2f ms", multi_time);
        if (single_time > 0) {
            printf(" (%.2fx speedup)", single_time / multi_time);
        }
        printf("\n");
    }
    
    if (pipeline_time > 0) {
        printf("Pipelined:        %.2f ms", pipeline_time);
        if (single_time > 0) {
            printf(" (%.2fx speedup)", single_time / pipeline_time);
        }
        printf("\n");
    }
    
    printf("\n🎯 LESSONS LEARNED:\n");
    printf("- Stream splitting allows overlapped execution\n");
    printf("- Multiple small transfers can be faster than one large transfer\n");
    printf("- Pipelining can achieve even better performance\n");
    printf("- Always use pinned memory for async transfers\n");
}

// =============================================================================
// MAIN FUNCTION - YOUR IMPLEMENTATION AREA
// =============================================================================

int main() {
    printf("🎯 STREAM SPLITTING EXERCISE\n");
    printf("============================\n");
    printf("Total data size: %.1f MB (%d elements)\n", 
           (TOTAL_ELEMENTS * sizeof(float)) / (1024.0f * 1024.0f), 
           TOTAL_ELEMENTS);
    printf("Number of streams: %d\n", NUM_STREAMS);
    printf("Chunk size: %d elements per stream\n", TOTAL_ELEMENTS / NUM_STREAMS);
    
    // Check CUDA device info
    int device;
    cudaGetDevice(&device);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    printf("Using GPU: %s\n", prop.name);
    printf("Memory bandwidth: %.1f GB/s\n", 
           prop.memoryBusWidth / 8.0 * prop.memoryClockRate * 2.0 / 1e6);
    
    // TODO: YOUR IMPLEMENTATION HERE
    // ==============================
    /*
     * YOUR TASK:
     * 1. Call singleStreamBaseline() and store the result
     * 2. Call multiStreamImplementation() and store the result  
     * 3. (Optional) Call pipelinedStreamsImplementation() and store the result
     * 4. Call comparePerformance() with your results
     * 
     * Example structure:
     * 
     * float single_time = singleStreamBaseline();
     * float multi_time = multiStreamImplementation();
     * float pipeline_time = pipelinedStreamsImplementation(); // Optional
     * 
     * comparePerformance(single_time, multi_time, pipeline_time);
     */
    
    printf("\n🎉 Exercise completed! Compare your results above.\n");
    printf("💡 Try experimenting with different NUM_STREAMS values!\n");
    
    return 0;
} 
#include <stdio.h>
#include <cuda_runtime.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

/*
 * CUDA GRAPHS TUTORIAL
 * ====================
 * 
 * COMPILATION:
 * nvcc -O3 -o cuda_graphs cuda_graphs_tutorial.cu
 * ./cuda_graphs
 * 
 * WHAT ARE CUDA GRAPHS?
 * --------------------
 * CUDA Graphs allow you to capture a sequence of CUDA operations and replay them
 * with minimal CPU overhead. Instead of launching kernels one by one, you can
 * launch an entire graph of operations with a single API call.
 * 
 * BENEFITS:
 * - Reduced CPU overhead for repetitive kernel sequences
 * - Better GPU utilization
 * - Lower latency for small kernels
 * - Ideal for RL environments with repetitive simulations!
 * 
 * This tutorial demonstrates:
 * 1. Simple kernel + memory copy sequence
 * 2. Regular execution vs CUDA Graph execution
 * 3. Performance comparison and timing analysis
 * 4. Multiple iterations to show graph benefits
 */

// =============================================================================
// HELPER FUNCTIONS
// =============================================================================

void checkCudaError(cudaError_t error, const char* message) {
    if (error != cudaSuccess) {
        printf("CUDA Error: %s - %s\n", message, cudaGetErrorString(error));
        exit(1);
    }
}

// Simple kernel that adds 1.0f to each element
__global__ void addOneKernel(float* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] = data[idx] + 1.0f;
    }
}

// More complex kernel for comparison
__global__ void complexKernel(float* data, int n, float multiplier) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float value = data[idx];
        
        // Some computation to make it more realistic
        for (int i = 0; i < 10; i++) {
            value = sinf(value * multiplier) + cosf(value * 0.5f);
        }
        
        data[idx] = value;
    }
}

// Initialize array with pattern
void initializeData(float* data, int size) {
    for (int i = 0; i < size; i++) {
        data[i] = (float)i * 0.001f;
    }
}

// Verify results
bool verifyResults(float* expected, float* actual, int size, float tolerance = 1e-5f) {
    for (int i = 0; i < size; i++) {
        if (fabsf(expected[i] - actual[i]) > tolerance) {
            printf("Verification failed at index %d: expected %.6f, got %.6f\n", 
                   i, expected[i], actual[i]);
            return false;
        }
    }
    return true;
}

// =============================================================================
// REGULAR EXECUTION (NO GRAPHS)
// =============================================================================

float regularExecution(float* h_input, float* h_output, float* d_data, 
                      int size, int iterations, cudaStream_t stream) {
    printf("\n🔄 REGULAR EXECUTION (Traditional approach)\n");
    printf("===========================================\n");
    
    int threads_per_block = 256;
    int blocks = (size + threads_per_block - 1) / threads_per_block;
    
    // Create timing events
    cudaEvent_t start, stop;
    checkCudaError(cudaEventCreate(&start), "Failed to create start event");
    checkCudaError(cudaEventCreate(&stop), "Failed to create stop event");
    
    // Record start time
    checkCudaError(cudaEventRecord(start, stream), "Failed to record start event");
    
    // Execute the sequence multiple times
    for (int iter = 0; iter < iterations; iter++) {
        // H2D transfer
        checkCudaError(
            cudaMemcpyAsync(d_data, h_input, size * sizeof(float), 
                           cudaMemcpyHostToDevice, stream),
            "Failed H2D transfer"
        );
        
        // Launch kernel
        addOneKernel<<<blocks, threads_per_block, 0, stream>>>(d_data, size);
        checkCudaError(cudaGetLastError(), "Kernel launch failed");
        
        // D2H transfer
        checkCudaError(
            cudaMemcpyAsync(h_output, d_data, size * sizeof(float), 
                           cudaMemcpyDeviceToHost, stream),
            "Failed D2H transfer"
        );
    }
    
    // Wait for completion and record stop time
    checkCudaError(cudaStreamSynchronize(stream), "Stream synchronization failed");
    checkCudaError(cudaEventRecord(stop, stream), "Failed to record stop event");
    checkCudaError(cudaEventSynchronize(stop), "Failed to synchronize stop event");
    
    // Calculate elapsed time
    float elapsed_ms;
    checkCudaError(cudaEventElapsedTime(&elapsed_ms, start, stop), 
                   "Failed to calculate elapsed time");
    
    printf("Regular execution completed:\n");
    printf("  Iterations: %d\n", iterations);
    printf("  Total time: %.3f ms\n", elapsed_ms);
    printf("  Time per iteration: %.3f ms\n", elapsed_ms / iterations);
    
    // Cleanup events
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    return elapsed_ms;
}

// =============================================================================
// CUDA GRAPH EXECUTION
// =============================================================================

float graphExecution(float* h_input, float* h_output, float* d_data, 
                    int size, int iterations, cudaStream_t stream) {
    printf("\n⚡ CUDA GRAPH EXECUTION (Optimized approach)\n");
    printf("============================================\n");
    
    int threads_per_block = 256;
    int blocks = (size + threads_per_block - 1) / threads_per_block;
    
    // Step 1: Capture the sequence into a graph
    printf("📸 Step 1: Capturing operations into CUDA graph...\n");
    
    cudaGraph_t graph;
    cudaGraphExec_t graph_exec;
    
    // Begin graph capture
    checkCudaError(
        cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal),
        "Failed to begin stream capture"
    );
    
    // Record the sequence we want to capture (single iteration)
    printf("  - Recording H2D transfer...\n");
    checkCudaError(
        cudaMemcpyAsync(d_data, h_input, size * sizeof(float), 
                       cudaMemcpyHostToDevice, stream),
        "Failed to record H2D transfer"
    );
    
    printf("  - Recording kernel launch...\n");
    addOneKernel<<<blocks, threads_per_block, 0, stream>>>(d_data, size);
    checkCudaError(cudaGetLastError(), "Failed to record kernel launch");
    
    printf("  - Recording D2H transfer...\n");
    checkCudaError(
        cudaMemcpyAsync(h_output, d_data, size * sizeof(float), 
                       cudaMemcpyDeviceToHost, stream),
        "Failed to record D2H transfer"
    );
    
    // End capture and create the graph
    checkCudaError(
        cudaStreamEndCapture(stream, &graph),
        "Failed to end stream capture"
    );
    
    printf("  ✅ Graph captured successfully!\n");
    
    // Step 2: Instantiate the graph for execution
    printf("🔧 Step 2: Instantiating graph for execution...\n");
    checkCudaError(
        cudaGraphInstantiate(&graph_exec, graph, NULL, NULL, 0),
        "Failed to instantiate graph"
    );
    printf("  ✅ Graph instantiated successfully!\n");
    
    // Step 3: Execute the graph multiple times
    printf("🚀 Step 3: Executing graph %d times...\n", iterations);
    
    // Create timing events
    cudaEvent_t start, stop;
    checkCudaError(cudaEventCreate(&start), "Failed to create start event");
    checkCudaError(cudaEventCreate(&stop), "Failed to create stop event");
    
    // Record start time
    checkCudaError(cudaEventRecord(start, stream), "Failed to record start event");
    
    // Launch the graph multiple times
    for (int iter = 0; iter < iterations; iter++) {
        checkCudaError(
            cudaGraphLaunch(graph_exec, stream),
            "Failed to launch graph"
        );
    }
    
    // Wait for completion and record stop time
    checkCudaError(cudaStreamSynchronize(stream), "Stream synchronization failed");
    checkCudaError(cudaEventRecord(stop, stream), "Failed to record stop event");
    checkCudaError(cudaEventSynchronize(stop), "Failed to synchronize stop event");
    
    // Calculate elapsed time
    float elapsed_ms;
    checkCudaError(cudaEventElapsedTime(&elapsed_ms, start, stop), 
                   "Failed to calculate elapsed time");
    
    printf("CUDA Graph execution completed:\n");
    printf("  Iterations: %d\n", iterations);
    printf("  Total time: %.3f ms\n", elapsed_ms);
    printf("  Time per iteration: %.3f ms\n", elapsed_ms / iterations);
    
    // Cleanup
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaGraphExecDestroy(graph_exec);
    cudaGraphDestroy(graph);
    
    return elapsed_ms;
}

// =============================================================================
// ADVANCED GRAPH EXAMPLE: MULTI-KERNEL SEQUENCE
// =============================================================================

float advancedGraphExecution(float* h_input, float* h_output, float* d_data, 
                           float* d_temp, int size, int iterations, cudaStream_t stream) {
    printf("\n🎯 ADVANCED GRAPH: Multi-kernel sequence\n");
    printf("=======================================\n");
    
    int threads_per_block = 256;
    int blocks = (size + threads_per_block - 1) / threads_per_block;
    
    cudaGraph_t graph;
    cudaGraphExec_t graph_exec;
    
    // Begin capture
    checkCudaError(
        cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal),
        "Failed to begin advanced capture"
    );
    
    // Complex sequence: H2D -> Kernel1 -> Kernel2 -> D2H
    cudaMemcpyAsync(d_data, h_input, size * sizeof(float), 
                   cudaMemcpyHostToDevice, stream);
    
    addOneKernel<<<blocks, threads_per_block, 0, stream>>>(d_data, size);
    
    // Copy to temp buffer and apply complex kernel
    cudaMemcpyAsync(d_temp, d_data, size * sizeof(float), 
                   cudaMemcpyDeviceToDevice, stream);
    
    complexKernel<<<blocks, threads_per_block, 0, stream>>>(d_temp, size, 1.1f);
    
    // Combine results (add temp back to original)
    addOneKernel<<<blocks, threads_per_block, 0, stream>>>(d_temp, size);
    
    cudaMemcpyAsync(h_output, d_temp, size * sizeof(float), 
                   cudaMemcpyDeviceToHost, stream);
    
    // End capture
    checkCudaError(
        cudaStreamEndCapture(stream, &graph),
        "Failed to end advanced capture"
    );
    
    checkCudaError(
        cudaGraphInstantiate(&graph_exec, graph, NULL, NULL, 0),
        "Failed to instantiate advanced graph"
    );
    
    // Time the execution
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start, stream);
    
    for (int iter = 0; iter < iterations; iter++) {
        cudaGraphLaunch(graph_exec, stream);
    }
    
    cudaStreamSynchronize(stream);
    cudaEventRecord(stop, stream);
    cudaEventSynchronize(stop);
    
    float elapsed_ms;
    cudaEventElapsedTime(&elapsed_ms, start, stop);
    
    printf("Advanced graph execution completed:\n");
    printf("  Multi-kernel sequence per iteration\n");
    printf("  Iterations: %d\n", iterations);
    printf("  Total time: %.3f ms\n", elapsed_ms);
    printf("  Time per iteration: %.3f ms\n", elapsed_ms / iterations);
    
    // Cleanup
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaGraphExecDestroy(graph_exec);
    cudaGraphDestroy(graph);
    
    return elapsed_ms;
}

// =============================================================================
// PERFORMANCE COMPARISON
// =============================================================================

void comparePerformance(float regular_time, float graph_time, float advanced_time, 
                       int iterations) {
    printf("\n📊 PERFORMANCE COMPARISON\n");
    printf("========================\n");
    
    printf("Method                | Total Time | Per Iteration | Relative Performance\n");
    printf("---------------------|------------|---------------|--------------------\n");
    printf("Regular Execution    | %8.3f ms | %9.3f ms | 1.00x (baseline)\n", 
           regular_time, regular_time / iterations);
    
    if (graph_time > 0) {
        float speedup = regular_time / graph_time;
        printf("CUDA Graph          | %8.3f ms | %9.3f ms | %.2fx faster\n", 
               graph_time, graph_time / iterations, speedup);
    }
    
    if (advanced_time > 0) {
        printf("Advanced Graph      | %8.3f ms | %9.3f ms | (complex sequence)\n", 
               advanced_time, advanced_time / iterations);
    }
    
    printf("\n🎯 KEY INSIGHTS:\n");
    printf("• CUDA Graphs reduce CPU overhead by batching GPU operations\n");
    printf("• Benefits increase with more iterations and smaller kernels\n");
    printf("• Ideal for RL environments with repetitive simulation patterns\n");
    printf("• Graph capture has one-time cost, but execution is very fast\n");
    
    if (graph_time > 0 && regular_time > graph_time) {
        float time_saved = regular_time - graph_time;
        printf("• You saved %.3f ms (%.1f%%) using CUDA Graphs!\n", 
               time_saved, (time_saved / regular_time) * 100.0f);
    }
}

// =============================================================================
// MAIN FUNCTION
// =============================================================================

int main() {
    printf("🚀 CUDA GRAPHS TUTORIAL\n");
    printf("=======================\n");
    printf("This tutorial demonstrates the performance benefits of CUDA Graphs\n");
    printf("for repetitive kernel sequences - perfect for RL environments!\n\n");
    
    // Problem setup
    const int SIZE = 1024 * 1024;  // 1M elements
    const int ITERATIONS = 100;    // Number of times to repeat the sequence
    const int BYTES = SIZE * sizeof(float);
    
    printf("Configuration:\n");
    printf("  Array size: %d elements (%.1f MB)\n", SIZE, BYTES / (1024.0f * 1024.0f));
    printf("  Iterations: %d\n", ITERATIONS);
    printf("  Operation: A[i] = A[i] + 1.0f\n\n");
    
    // Check CUDA device
    int device;
    cudaGetDevice(&device);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    printf("GPU: %s\n", prop.name);
    printf("Compute Capability: %d.%d\n", prop.major, prop.minor);
    
    // Check if CUDA Graphs are supported (requires compute capability 3.5+)
    if (prop.major < 3 || (prop.major == 3 && prop.minor < 5)) {
        printf("❌ ERROR: CUDA Graphs require compute capability 3.5 or higher\n");
        printf("Your device has compute capability %d.%d\n", prop.major, prop.minor);
        return 1;
    }
    printf("✅ CUDA Graphs supported!\n\n");
    
    // Allocate memory
    printf("💾 Allocating memory...\n");
    
    // Host memory (pinned for better async performance)
    float *h_input, *h_output_regular, *h_output_graph;
    checkCudaError(
        cudaHostAlloc(&h_input, BYTES, cudaHostAllocDefault),
        "Failed to allocate input host memory"
    );
    checkCudaError(
        cudaHostAlloc(&h_output_regular, BYTES, cudaHostAllocDefault),
        "Failed to allocate regular output host memory"
    );
    checkCudaError(
        cudaHostAlloc(&h_output_graph, BYTES, cudaHostAllocDefault),
        "Failed to allocate graph output host memory"
    );
    
    // Device memory
    float *d_data, *d_temp;
    checkCudaError(
        cudaMalloc(&d_data, BYTES),
        "Failed to allocate device memory"
    );
    checkCudaError(
        cudaMalloc(&d_temp, BYTES),
        "Failed to allocate temp device memory"
    );
    
    // Create stream
    cudaStream_t stream;
    checkCudaError(
        cudaStreamCreate(&stream),
        "Failed to create stream"
    );
    
    // Initialize input data
    printf("🔧 Initializing data...\n");
    initializeData(h_input, SIZE);
    
    printf("✅ Setup complete!\n");
    
    // Run performance comparison
    printf("\n" "=" * 60 "\n");
    printf("PERFORMANCE COMPARISON\n");
    printf("=" * 60 "\n");
    
    // 1. Regular execution
    float regular_time = regularExecution(h_input, h_output_regular, d_data, 
                                        SIZE, ITERATIONS, stream);
    
    // 2. CUDA Graph execution
    float graph_time = graphExecution(h_input, h_output_graph, d_data, 
                                    SIZE, ITERATIONS, stream);
    
    // 3. Advanced graph (bonus)
    float advanced_time = advancedGraphExecution(h_input, h_output_graph, d_data, d_temp,
                                                SIZE, ITERATIONS / 10, stream); // Fewer iterations for complex sequence
    
    // Verify results match
    printf("\n🔍 Verifying results...\n");
    if (verifyResults(h_output_regular, h_output_graph, SIZE)) {
        printf("✅ Results match! CUDA Graph produces identical output.\n");
    } else {
        printf("❌ Results don't match! Check implementation.\n");
    }
    
    // Compare performance
    comparePerformance(regular_time, graph_time, advanced_time, ITERATIONS);
    
    // Application to RL environments
    printf("\n🎮 APPLICATION TO RL ENVIRONMENTS:\n");
    printf("=================================\n");
    printf("Your Monte Carlo RL environment would benefit from CUDA Graphs by:\n");
    printf("1. Capturing the simulation sequence once:\n");
    printf("   • H2D transfer of codebook/parameters\n");
    printf("   • Monte Carlo simulation kernel\n");
    printf("   • Reduction/aggregation kernels\n");
    printf("   • D2H transfer of results\n");
    printf("2. Launching the entire sequence with cudaGraphLaunch()\n");
    printf("3. Reduced overhead = more simulations per second!\n");
    printf("4. Better for batched training where same operations repeat\n");
    
    // Cleanup
    printf("\n🧹 Cleaning up...\n");
    cudaHostFree(h_input);
    cudaHostFree(h_output_regular);
    cudaHostFree(h_output_graph);
    cudaFree(d_data);
    cudaFree(d_temp);
    cudaStreamDestroy(stream);
    
    printf("✅ Tutorial completed successfully!\n");
    printf("\n💡 Next steps:\n");
    printf("   1. Try different iteration counts to see when graphs become beneficial\n");
    printf("   2. Experiment with different kernel complexities\n");
    printf("   3. Apply graphs to your Monte Carlo RL simulation\n");
    printf("   4. Combine with streams for maximum performance\n");
    
    return 0;
} 
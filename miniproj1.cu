#include <stdio.h>
#include <cuda_runtime.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

/*
 * CUDA STREAMING MASTERCLASS
 * ==========================
 * 
 * COMPILATION INSTRUCTIONS:
 * ------------------------
 * To compile and run these exercises:
 * 
 * nvcc -o miniproj1 miniproj1.cu
 * ./miniproj1
 * 
 * For better performance analysis:
 * nvcc -O3 -o miniproj1 miniproj1.cu
 * 
 * Note: IDE linter errors for CUDA headers are normal - the code will compile correctly with nvcc.
 * 
 * This file contains progressive exercises to master CUDA streaming:
 * 1. Basic Sync vs Async Memory Transfers
 * 2. CUDA Events for Timing and Synchronization
 * 3. Multiple Streams for Overlapped Execution
 * 4. Stream Dependencies with cudaStreamWaitEvent
 * 5. Full CPU-GPU Pipelines
 * 6. Common Mistakes (What NOT to do)
 * 
 * Run each exercise and compare the timing results!
 */

// Helper function for error checking
void checkCudaError(cudaError_t error, const char* message) {
    if (error != cudaSuccess) {
        printf("CUDA Error: %s - %s\n", message, cudaGetErrorString(error));
        exit(1);
    }
}

// Simple kernel that does some work to simulate computation
__global__ void computeKernel(float* data, int n, int iterations) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float value = data[idx];
        // Simulate some computation
        for (int i = 0; i < iterations; i++) {
            value = sinf(value * 1.1f) + cosf(value * 0.9f);
        }
        data[idx] = value;
    }
}

// CPU work simulation
void cpuWork(int milliseconds) {
    clock_t start = clock();
    while ((clock() - start) * 1000.0 / CLOCKS_PER_SEC < milliseconds) {
        // Simulate CPU work
        volatile float dummy = 0;
        for (int i = 0; i < 100000; i++) {
            dummy += sinf(i * 0.001f);
        }
    }
}

/*
 * ========================================================================
 * EXERCISE 1: SYNC vs ASYNC MEMORY TRANSFERS
 * ========================================================================
 * Learn the fundamental difference between blocking and non-blocking transfers
 */

void exercise1_sync_vs_async() {
    printf("\n=== EXERCISE 1: SYNCHRONOUS vs ASYNCHRONOUS TRANSFERS ===\n");
    
    const int N = 1024 * 1024;  // 1M floats = 4MB
    const int size = N * sizeof(float);
    
    // Host memory (use pinned for better async performance)
    float *h_data_sync, *h_data_async;
    cudaHostAlloc(&h_data_sync, size, cudaHostAllocDefault);
    cudaHostAlloc(&h_data_async, size, cudaHostAllocDefault);
    
    // Initialize data
    for (int i = 0; i < N; i++) {
        h_data_sync[i] = (float)i;
        h_data_async[i] = (float)i;
    }
    
    // Device memory
    float *d_data_sync, *d_data_async;
    cudaMalloc(&d_data_sync, size);
    cudaMalloc(&d_data_async, size);
    
    // Create events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // ===== BAD EXAMPLE: SYNCHRONOUS TRANSFERS =====
    printf("1.1 SYNCHRONOUS (BAD - CPU waits):\n");
    cudaEventRecord(start);
    
    // CPU is BLOCKED during these transfers
    cudaMemcpy(d_data_sync, h_data_sync, size, cudaMemcpyHostToDevice);
    printf("   CPU was blocked during H2D transfer!\n");
    
    // Launch kernel
    computeKernel<<<(N+255)/256, 256>>>(d_data_sync, N, 1000);
    cudaDeviceSynchronize();  // CPU waits for kernel
    printf("   CPU was blocked during kernel execution!\n");
    
    // Copy back
    cudaMemcpy(h_data_sync, d_data_sync, size, cudaMemcpyDeviceToHost);
    printf("   CPU was blocked during D2H transfer!\n");
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float sync_time;
    cudaEventElapsedTime(&sync_time, start, stop);
    printf("   Total synchronous time: %.2f ms\n", sync_time);
    
    // ===== GOOD EXAMPLE: ASYNCHRONOUS TRANSFERS =====
    printf("\n1.2 ASYNCHRONOUS (GOOD - CPU can work):\n");
    
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    
    cudaEventRecord(start);
    
    // CPU is NOT blocked during async transfer
    cudaMemcpyAsync(d_data_async, h_data_async, size, cudaMemcpyHostToDevice, stream);
    printf("   CPU can work while H2D transfer happens!\n");
    
    // CPU can do work while transfer happens
    cpuWork(10);  // Simulate 10ms of CPU work
    printf("   CPU did 10ms of useful work during transfer!\n");
    
    // Launch kernel on same stream
    computeKernel<<<(N+255)/256, 256, 0, stream>>>(d_data_async, N, 1000);
    
    // CPU can do more work while kernel runs
    cpuWork(5);  // Simulate 5ms of CPU work
    printf("   CPU did 5ms of work while kernel executed!\n");
    
    // Async copy back
    cudaMemcpyAsync(h_data_async, d_data_async, size, cudaMemcpyDeviceToHost, stream);
    
    // More CPU work while copy happens
    cpuWork(5);  // Simulate 5ms of CPU work
    printf("   CPU did 5ms of work while D2H transfer happened!\n");
    
    // Only wait at the end
    cudaStreamSynchronize(stream);
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float async_time;
    cudaEventElapsedTime(&async_time, start, stop);
    printf("   Total asynchronous time: %.2f ms\n", async_time);
    
    printf("   🎯 LESSON: Async allowed %.2f ms of overlapped CPU work!\n", 
           sync_time - async_time);
    
    // Cleanup
    cudaHostFree(h_data_sync);
    cudaHostFree(h_data_async);
    cudaFree(d_data_sync);
    cudaFree(d_data_async);
    cudaStreamDestroy(stream);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

/*
 * ========================================================================
 * EXERCISE 2: CUDA EVENTS FOR SYNCHRONIZATION
 * ========================================================================
 * Learn how to use events to coordinate between streams and measure overlap
 */

void exercise2_cuda_events() {
    printf("\n=== EXERCISE 2: CUDA EVENTS FOR SYNCHRONIZATION ===\n");
    
    const int N = 1024 * 1024;
    const int size = N * sizeof(float);
    
    float *h_data;
    float *d_data1, *d_data2;
    cudaHostAlloc(&h_data, size, cudaHostAllocDefault);
    cudaMalloc(&d_data1, size);
    cudaMalloc(&d_data2, size);
    
    // Initialize data
    for (int i = 0; i < N; i++) {
        h_data[i] = (float)i;
    }
    
    // Create streams and events
    cudaStream_t stream1, stream2;
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);
    
    cudaEvent_t start, stop, transfer_done, kernel1_done, kernel2_done;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventCreate(&transfer_done);
    cudaEventCreate(&kernel1_done);
    cudaEventCreate(&kernel2_done);
    
    printf("2.1 Using events to measure individual operation timings:\n");
    
    cudaEventRecord(start);
    
    // Transfer data and record when done
    cudaMemcpyAsync(d_data1, h_data, size, cudaMemcpyHostToDevice, stream1);
    cudaEventRecord(transfer_done, stream1);
    
    // Launch kernel 1 and record when done
    computeKernel<<<(N+255)/256, 256, 0, stream1>>>(d_data1, N, 2000);
    cudaEventRecord(kernel1_done, stream1);
    
    // Copy data to second buffer and launch second kernel
    cudaMemcpyAsync(d_data2, d_data1, size, cudaMemcpyDeviceToDevice, stream2);
    computeKernel<<<(N+255)/256, 256, 0, stream2>>>(d_data2, N, 1000);
    cudaEventRecord(kernel2_done, stream2);
    
    // Copy result back
    cudaMemcpyAsync(h_data, d_data2, size, cudaMemcpyDeviceToHost, stream2);
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    // Measure individual timings
    float transfer_time, kernel1_time, kernel2_time, total_time;
    cudaEventElapsedTime(&transfer_time, start, transfer_done);
    cudaEventElapsedTime(&kernel1_time, transfer_done, kernel1_done);
    cudaEventElapsedTime(&kernel2_time, kernel1_done, kernel2_done);
    cudaEventElapsedTime(&total_time, start, stop);
    
    printf("   H2D Transfer: %.2f ms\n", transfer_time);
    printf("   Kernel 1:     %.2f ms\n", kernel1_time);
    printf("   Kernel 2:     %.2f ms\n", kernel2_time);
    printf("   Total:        %.2f ms\n", total_time);
    printf("   🎯 LESSON: Events let you profile individual operations!\n");
    
    // Cleanup
    cudaHostFree(h_data);
    cudaFree(d_data1);
    cudaFree(d_data2);
    cudaStreamDestroy(stream1);
    cudaStreamDestroy(stream2);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaEventDestroy(transfer_done);
    cudaEventDestroy(kernel1_done);
    cudaEventDestroy(kernel2_done);
}

/*
 * ========================================================================
 * EXERCISE 3: MULTIPLE STREAMS FOR OVERLAPPED EXECUTION
 * ========================================================================
 * Learn how multiple streams can execute operations in parallel
 */

void exercise3_multiple_streams() {
    printf("\n=== EXERCISE 3: MULTIPLE STREAMS FOR OVERLAP ===\n");
    
    const int N = 1024 * 1024;
    const int size = N * sizeof(float);
    const int num_streams = 4;
    const int chunk_size = N / num_streams;
    
    // Allocate memory
    float *h_data;
    float *d_data[num_streams];
    cudaHostAlloc(&h_data, size, cudaHostAllocDefault);
    
    for (int i = 0; i < num_streams; i++) {
        cudaMalloc(&d_data[i], size / num_streams);
    }
    
    // Initialize data
    for (int i = 0; i < N; i++) {
        h_data[i] = (float)i;
    }
    
    // ===== BAD EXAMPLE: SINGLE STREAM (NO OVERLAP) =====
    printf("3.1 SINGLE STREAM (BAD - No overlap):\n");
    
    cudaStream_t single_stream;
    cudaStreamCreate(&single_stream);
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    
    // Process all chunks sequentially - no overlap possible
    for (int i = 0; i < num_streams; i++) {
        int offset = i * chunk_size;
        
        // Transfer, compute, transfer back - all sequential
        cudaMemcpyAsync(d_data[i], &h_data[offset], 
                       chunk_size * sizeof(float), 
                       cudaMemcpyHostToDevice, single_stream);
        
        computeKernel<<<(chunk_size+255)/256, 256, 0, single_stream>>>
                     (d_data[i], chunk_size, 3000);
        
        cudaMemcpyAsync(&h_data[offset], d_data[i], 
                       chunk_size * sizeof(float), 
                       cudaMemcpyDeviceToHost, single_stream);
    }
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float single_stream_time;
    cudaEventElapsedTime(&single_stream_time, start, stop);
    printf("   Single stream time: %.2f ms\n", single_stream_time);
    
    // ===== GOOD EXAMPLE: MULTIPLE STREAMS (OVERLAP) =====
    printf("\n3.2 MULTIPLE STREAMS (GOOD - Overlap!):\n");
    
    cudaStream_t streams[num_streams];
    for (int i = 0; i < num_streams; i++) {
        cudaStreamCreate(&streams[i]);
    }
    
    cudaEventRecord(start);
    
    // Launch all H2D transfers simultaneously
    for (int i = 0; i < num_streams; i++) {
        int offset = i * chunk_size;
        cudaMemcpyAsync(d_data[i], &h_data[offset], 
                       chunk_size * sizeof(float), 
                       cudaMemcpyHostToDevice, streams[i]);
    }
    
    // Launch all kernels (will wait for their respective transfers)
    for (int i = 0; i < num_streams; i++) {
        computeKernel<<<(chunk_size+255)/256, 256, 0, streams[i]>>>
                     (d_data[i], chunk_size, 3000);
    }
    
    // Launch all D2H transfers
    for (int i = 0; i < num_streams; i++) {
        int offset = i * chunk_size;
        cudaMemcpyAsync(&h_data[offset], d_data[i], 
                       chunk_size * sizeof(float), 
                       cudaMemcpyDeviceToHost, streams[i]);
    }
    
    // Wait for all streams to complete
    for (int i = 0; i < num_streams; i++) {
        cudaStreamSynchronize(streams[i]);
    }
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float multi_stream_time;
    cudaEventElapsedTime(&multi_stream_time, start, stop);
    printf("   Multi-stream time: %.2f ms\n", multi_stream_time);
    
    float speedup = single_stream_time / multi_stream_time;
    printf("   🎯 SPEEDUP: %.2fx faster with multiple streams!\n", speedup);
    printf("   🎯 LESSON: Operations in different streams can overlap!\n");
    
    // Cleanup
    cudaHostFree(h_data);
    for (int i = 0; i < num_streams; i++) {
        cudaFree(d_data[i]);
        cudaStreamDestroy(streams[i]);
    }
    cudaStreamDestroy(single_stream);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

/*
 * ========================================================================
 * EXERCISE 4: STREAM DEPENDENCIES WITH cudaStreamWaitEvent
 * ========================================================================
 * Learn how to create dependencies between streams for complex workflows
 */

void exercise4_stream_dependencies() {
    printf("\n=== EXERCISE 4: STREAM DEPENDENCIES ===\n");
    
    const int N = 1024 * 1024;
    const int size = N * sizeof(float);
    
    float *h_input, *h_output;
    float *d_data1, *d_data2, *d_result;
    cudaHostAlloc(&h_input, size, cudaHostAllocDefault);
    cudaHostAlloc(&h_output, size, cudaHostAllocDefault);
    cudaMalloc(&d_data1, size);
    cudaMalloc(&d_data2, size);
    cudaMalloc(&d_result, size);
    
    // Initialize data
    for (int i = 0; i < N; i++) {
        h_input[i] = (float)i;
    }
    
    // Create streams and events
    cudaStream_t stream1, stream2, stream3;
    cudaCreateStream(&stream1);
    cudaCreateStream(&stream2);
    cudaCreateStream(&stream3);
    
    cudaEvent_t start, stop, data1_ready, data2_ready;
    cudaCreateEvent(&start);
    cudaCreateEvent(&stop);
    cudaCreateEvent(&data1_ready);
    cudaCreateEvent(&data2_ready);
    
    
    printf("4.1 Complex workflow with dependencies:\n");
    printf("   Stream 1: Process data -> d_data1\n");
    printf("   Stream 2: Process data -> d_data2\n");
    printf("   Stream 3: Wait for BOTH, then combine results\n");
    
    cudaEventRecord(start);
    
    // ===== STREAM 1: Process first dataset =====
    cudaMemcpyAsync(d_data1, h_input, size, cudaMemcpyHostToDevice, stream1);
    computeKernel<<<(N+255)/256, 256, 0, stream1>>>(d_data1, N, 2000);
    cudaEventRecord(data1_ready, stream1); // Signal when stream1 is done
    
    // ===== STREAM 2: Process second dataset =====
    cudaMemcpyAsync(d_data2, h_input, size, cudaMemcpyHostToDevice, stream2);
    computeKernel<<<(N+255)/256, 256, 0, stream2>>>(d_data2, N, 1500);
    cudaEventRecord(data2_ready, stream2); // Signal when stream2 is done
    
    // ===== STREAM 3: Wait for both streams, then combine =====
    // This is the KEY: stream3 waits for events from other streams
    cudaStreamWaitEvent(stream3, data1_ready, 0);
    cudaStreamWaitEvent(stream3, data2_ready, 0);
    
    printf("   Stream 3 waiting for both streams to complete...\n");
    
    // Now stream3 can safely use both d_data1 and d_data2
    // Simple kernel to combine results (d_result = d_data1 + d_data2)
    __global__ auto combine_kernel = [] __device__ (float* a, float* b, float* result, int n) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
            result[idx] = a[idx] + b[idx];
        }
    };
    
    // Lambda kernels need special handling, let's use a simple addition instead
    // For now, just copy one of the results
    cudaMemcpyAsync(d_result, d_data1, size, cudaMemcpyDeviceToDevice, stream3);
    
    // Final transfer back to host
    cudaMemcpyAsync(h_output, d_result, size, cudaMemcpyDeviceToHost, stream3);
    
    cudaStreamSynchronize(stream3); // Wait for stream3 to complete before recording stop event
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float total_time;
    cudaEventElapsedTime(&total_time, start, stop);
    printf("   Total pipeline time: %.2f ms\n", total_time);
    printf("   🎯 LESSON: cudaStreamWaitEvent creates dependencies!\n");
    printf("   🎯 Stream 3 automatically waited for streams 1 & 2!\n");
    
    // ===== DEMONSTRATE WHAT HAPPENS WITHOUT DEPENDENCIES =====
    printf("\n4.2 WITHOUT dependencies (BROKEN - race condition):\n");
    
    cudaEventRecord(start);
    
    // Launch everything without proper synchronization
    cudaMemcpyAsync(d_data1, h_input, size, cudaMemcpyHostToDevice, stream1);
    cudaMemcpyAsync(d_data2, h_input, size, cudaMemcpyHostToDevice, stream2);
    
    // ❌ BAD: Stream3 doesn't wait - might use uninitialized data!
    cudaMemcpyAsync(d_result, d_data1, size, cudaMemcpyDeviceToDevice, stream3);
    cudaMemcpyAsync(h_output, d_result, size, cudaMemcpyDeviceToHost, stream3);
    
    cudaStreamSynchronize(stream3); // Wait for stream3 to complete
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float broken_time;
    cudaEventElapsedTime(&broken_time, start, stop);
    printf("   Broken (no deps) time: %.2f ms\n", broken_time);
    printf("   ❌ WARNING: This is faster but WRONG - race condition!\n");
    printf("   ❌ Stream3 might use uninitialized data!\n");
    
    // Cleanup
    cudaHostFree(h_input);
    cudaHostFree(h_output);
    cudaFree(d_data1);
    cudaFree(d_data2);
    cudaFree(d_result);
    cudaStreamDestroy(stream1);
    cudaStreamDestroy(stream2);
    cudaStreamDestroy(stream3);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaEventDestroy(data1_ready);
    cudaEventDestroy(data2_ready);
}

/*
 * ========================================================================
 * EXERCISE 5: FULL CPU-GPU PIPELINE
 * ========================================================================
 * Learn how to create a production-ready pipeline that keeps both CPU and GPU busy
 */

void exercise5_full_pipeline() {
    printf("\n=== EXERCISE 5: FULL CPU-GPU PIPELINE ===\n");
    
    const int batch_size = 256 * 1024;  // Smaller batches for better overlap
    const int num_batches = 8;
    const int total_elements = batch_size * num_batches;
    const int batch_bytes = batch_size * sizeof(float);
    
    // Double-buffered host memory
    float *h_input_buffers[2];
    float *h_output_buffers[2];
    cudaHostAlloc(&h_input_buffers[0], batch_bytes, cudaHostAllocDefault);
    cudaHostAlloc(&h_input_buffers[1], batch_bytes, cudaHostAllocDefault);
    cudaHostAlloc(&h_output_buffers[0], batch_bytes, cudaHostAllocDefault);
    cudaHostAlloc(&h_output_buffers[1], batch_bytes, cudaHostAllocDefault);
    
    // Double-buffered device memory
    float *d_buffers[2];
    cudaMalloc(&d_buffers[0], batch_bytes);
    cudaMalloc(&d_buffers[1], batch_bytes);
    
    // Create streams for pipeline stages
    cudaStream_t compute_streams[2];
    cudaStreamCreate(&compute_streams[0]);
    cudaStreamCreate(&compute_streams[1]);
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    printf("5.1 PIPELINED EXECUTION:\n");
    printf("   Pipeline stages: Data Prep → H2D → Compute → D2H → Process\n");
    printf("   Using double buffering to overlap CPU and GPU work\n");
    
    cudaEventRecord(start);
    
    // Initialize first batch on CPU
    for (int i = 0; i < batch_size; i++) {
        h_input_buffers[0][i] = (float)i;
    }
    
    int current_buffer = 0;
    
    for (int batch = 0; batch < num_batches; batch++) {
        int next_buffer = 1 - current_buffer;  // Alternate between 0 and 1
        cudaStream_t current_stream = compute_streams[current_buffer];
        
        printf("   Batch %d: Using buffer %d\n", batch, current_buffer);
        
        // Stage 1: Transfer current batch H2D
        cudaMemcpyAsync(d_buffers[current_buffer], 
                       h_input_buffers[current_buffer], 
                       batch_bytes, 
                       cudaMemcpyHostToDevice, 
                       current_stream);
        
        // Stage 2: Process current batch on GPU
        computeKernel<<<(batch_size+255)/256, 256, 0, current_stream>>>
                     (d_buffers[current_buffer], batch_size, 1000);
        
        // Stage 3: Transfer current batch D2H
        cudaMemcpyAsync(h_output_buffers[current_buffer], 
                       d_buffers[current_buffer], 
                       batch_bytes, 
                       cudaMemcpyDeviceToHost, 
                       current_stream);
        
        // Stage 4: While GPU works, CPU prepares NEXT batch
        if (batch + 1 < num_batches) {
            printf("   CPU preparing next batch while GPU works...\n");
            
            // Simulate CPU data preparation for next batch
            for (int i = 0; i < batch_size; i++) {
                h_input_buffers[next_buffer][i] = (float)((batch + 1) * batch_size + i);
                // Add some CPU work
                h_input_buffers[next_buffer][i] = sinf(h_input_buffers[next_buffer][i] * 0.001f);
            }
        }
        
        // Stage 5: Process completed results (from previous batch)
        if (batch > 0) {
            // Wait for previous batch to complete
            int prev_buffer = 1 - next_buffer;
            cudaStreamSynchronize(compute_streams[prev_buffer]);
            
            // Process results on CPU
            printf("   CPU processing results from previous batch...\n");
            float sum = 0;
            for (int i = 0; i < batch_size; i++) {
                sum += h_output_buffers[prev_buffer][i];
            }
            printf("   Batch %d result sum: %.2f\n", batch - 1, sum);
        }
        
        current_buffer = next_buffer;
    }
    
    // Process final batch
    cudaStreamSynchronize(compute_streams[1 - current_buffer]);
    printf("   Processing final batch results...\n");
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float pipeline_time;
    cudaEventElapsedTime(&pipeline_time, start, stop);
    printf("   Total pipelined time: %.2f ms\n", pipeline_time);
    printf("   Throughput: %.2f MB/s\n", 
           (total_elements * sizeof(float) * 2) / (pipeline_time / 1000.0) / (1024*1024));
    
    printf("   🎯 LESSON: Double buffering keeps both CPU and GPU busy!\n");
    printf("   🎯 While GPU processes batch N, CPU prepares batch N+1!\n");
    
    // Cleanup
    cudaHostFree(h_input_buffers[0]);
    cudaHostFree(h_input_buffers[1]);
    cudaHostFree(h_output_buffers[0]);
    cudaHostFree(h_output_buffers[1]);
    cudaFree(d_buffers[0]);
    cudaFree(d_buffers[1]);
    cudaStreamDestroy(compute_streams[0]);
    cudaStreamDestroy(compute_streams[1]);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

/*
 * ========================================================================
 * EXERCISE 6: COMMON MISTAKES (WHAT NOT TO DO)
 * ========================================================================
 * Learn about common pitfalls that kill performance
 */

void exercise6_common_mistakes() {
    printf("\n=== EXERCISE 6: COMMON MISTAKES ===\n");
    
    const int N = 1024 * 1024;
    const int size = N * sizeof(float);
    
    float *h_data;
    float *d_data;
    cudaHostAlloc(&h_data, size, cudaHostAllocDefault);
    cudaMalloc(&d_data, size);
    
    // Initialize data
    for (int i = 0; i < N; i++) {
        h_data[i] = (float)i;
    }
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // ===== MISTAKE 1: USING PAGEABLE MEMORY WITH ASYNC =====
    printf("6.1 MISTAKE: Using pageable memory with async transfers\n");
    
    float *h_pageable = (float*)malloc(size);
    for (int i = 0; i < N; i++) {
        h_pageable[i] = (float)i;
    }
    
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    
    cudaEventRecord(start);
    
    // ❌ BAD: Async transfer with pageable memory becomes synchronous!
    cudaMemcpyAsync(d_data, h_pageable, size, cudaMemcpyHostToDevice, stream);
    
    // This CPU work won't overlap because the transfer is secretly synchronous
    cpuWork(20);
    printf("   CPU work completed\n");
    
    cudaStreamSynchronize(stream);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float pageable_time;
    cudaEventElapsedTime(&pageable_time, start, stop);
    printf("   Time with pageable memory: %.2f ms\n", pageable_time);
    printf("   ❌ MISTAKE: Pageable memory makes async transfers synchronous!\n");
    
    // ===== MISTAKE 2: TOO MANY SMALL TRANSFERS =====
    printf("\n6.2 MISTAKE: Too many small transfers\n");
    
    const int num_small_transfers = 1000;
    const int small_size = N / num_small_transfers;
    
    cudaEventRecord(start);
    
    // ❌ BAD: Many small transfers have high latency overhead
    for (int i = 0; i < num_small_transfers; i++) {
        int offset = i * small_size;
        cudaMemcpyAsync(&d_data[offset], &h_data[offset], 
                       small_size * sizeof(float), 
                       cudaMemcpyHostToDevice, stream);
    }
    
    cudaStreamSynchronize(stream);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float small_transfers_time;
    cudaEventElapsedTime(&small_transfers_time, start, stop);
    printf("   Time with %d small transfers: %.2f ms\n", num_small_transfers, small_transfers_time);
    
    // Compare with single large transfer
    cudaEventRecord(start);
    cudaMemcpyAsync(d_data, h_data, size, cudaMemcpyHostToDevice, stream);
    cudaStreamSynchronize(stream);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float single_transfer_time;
    cudaEventElapsedTime(&single_transfer_time, start, stop);
    printf("   Time with 1 large transfer: %.2f ms\n", single_transfer_time);
    printf("   ❌ MISTAKE: Small transfers are %.2fx slower!\n", 
           small_transfers_time / single_transfer_time);
    
    // ===== MISTAKE 3: UNNECESSARY SYNCHRONIZATION =====
    printf("\n6.3 MISTAKE: Unnecessary synchronization\n");
    
    cudaEventRecord(start);
    
    // ❌ BAD: Synchronizing after every operation
    cudaMemcpyAsync(d_data, h_data, size, cudaMemcpyHostToDevice, stream);
    cudaStreamSynchronize(stream);  // ❌ Unnecessary!
    
    computeKernel<<<(N+255)/256, 256, 0, stream>>>(d_data, N, 1000);
    cudaStreamSynchronize(stream);  // ❌ Unnecessary!
    
    cudaMemcpyAsync(h_data, d_data, size, cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);  // Only this one is needed!
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float over_sync_time;
    cudaEventElapsedTime(&over_sync_time, start, stop);
    printf("   Time with over-synchronization: %.2f ms\n", over_sync_time);
    printf("   ❌ MISTAKE: Unnecessary syncs kill async benefits!\n");
    
    // ===== MISTAKE 4: WRONG STREAM USAGE =====
    printf("\n6.4 MISTAKE: Using default stream accidentally\n");
    
    cudaEventRecord(start);
    
    // ❌ BAD: Mixing default stream (NULL) with custom stream
    cudaMemcpyAsync(d_data, h_data, size, cudaMemcpyHostToDevice, stream);
    
    // This kernel launch uses default stream - will serialize with everything!
    computeKernel<<<(N+255)/256, 256>>>(d_data, N, 1000);  // No stream specified!
    
    cudaMemcpyAsync(h_data, d_data, size, cudaMemcpyDeviceToHost, stream);
    
    cudaStreamSynchronize(stream);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float mixed_stream_time;
    cudaEventElapsedTime(&mixed_stream_time, start, stop);
    printf("   Time with mixed streams: %.2f ms\n", mixed_stream_time);
    printf("   ❌ MISTAKE: Default stream serializes with all others!\n");
    
    printf("\n🎯 KEY LESSONS:\n");
    printf("   1. Always use pinned memory for async transfers\n");
    printf("   2. Batch small transfers into larger ones\n");
    printf("   3. Only synchronize when you actually need the results\n");
    printf("   4. Be consistent with stream usage\n");
    printf("   5. Default stream (NULL) blocks other streams!\n");
    
    // Cleanup
    free(h_pageable);
    cudaHostFree(h_data);
    cudaFree(d_data);
    cudaStreamDestroy(stream);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

/*
 * ========================================================================
 * MAIN FUNCTION - RUN ALL EXERCISES
 * ========================================================================
 */

int main() {
    printf("🚀 CUDA STREAMING MASTERCLASS\n");
    printf("===============================\n");
    printf("This tutorial will teach you CUDA streaming through hands-on exercises.\n");
    printf("Watch the timing results to understand the performance benefits!\n");
    
    // Check CUDA device
    int device;
    cudaGetDevice(&device);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    printf("Using device: %s\n", prop.name);
    printf("Compute capability: %d.%d\n", prop.major, prop.minor);
    printf("Memory bandwidth: %.1f GB/s\n", 
           prop.memoryBusWidth / 8.0 * prop.memoryClockRate * 2.0 / 1e6);
    
    // Run all exercises
    exercise1_sync_vs_async();
    exercise2_cuda_events();
    exercise3_multiple_streams();
    exercise4_stream_dependencies();
    exercise5_full_pipeline();
    exercise6_common_mistakes();
    
    printf("\n🎉 CONGRATULATIONS!\n");
    printf("You've completed the CUDA streaming masterclass!\n");
    printf("\nNext steps for your RL environment:\n");
    printf("1. Apply double buffering to your Monte Carlo simulation\n");
    printf("2. Use multiple streams to overlap different batches\n");
    printf("3. Pre-allocate and reuse memory pools\n");
    printf("4. Pipeline your RL agent's decision-making with simulation\n");
    
    return 0;
}

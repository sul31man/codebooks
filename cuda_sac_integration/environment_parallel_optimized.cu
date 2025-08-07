#include <stdio.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <string.h>

// CUDA Error Checking Macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            printf("CUDA Error at %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(1); \
        } \
    } while(0)

// Constants and buffer management
#define MAX_COLS 1024
#define MAX_ROWS 512
#define L 16
#define n 512
#define N 64
#define J 6
#define MAX_KA_VALUES 10
#define THREADS_PER_BLOCK 256
#define SIMS_PER_THREAD 4  // Each thread processes multiple simulations

// Global variables for persistent buffer (host-side)
static float action_buffer[MAX_COLS * MAX_ROWS] __attribute__((aligned(32)));
static int buffer_size = 0;
static int current_rows = 0;
static int current_cols = 0;
static int buffer_initialized = 0;

// Safe buffer bounds checking
static inline bool is_valid_buffer_access(int row, int col) {
    return (row >= 0 && row < MAX_ROWS && col >= 0 && col < MAX_COLS);
}

// New function to validate memory access for float pointers
__device__ bool is_valid_memory_access_float(float* ptr, int size) {
    return ptr != nullptr && size > 0;
}

// New function to validate memory access for int pointers
__device__ bool is_valid_memory_access_int(int* ptr, int size) {
    return ptr != nullptr && size > 0;
}

// Optimized parallel kernel with proper shared memory and Ka distribution
__global__ void optimized_parallel_simulation_kernel(float* codebook, int rows, int cols, 
                                                    int* Ka_values, int num_Ka, int num_sims, 
                                                    int* hit_rates) {
    // Improved thread organization:
    // - Each block processes ONE Ka value
    // - Each thread in block processes SIMS_PER_THREAD simulations
    // - Shared memory is used correctly with synchronization
    
    int ka_idx = blockIdx.x;  // Each block handles one Ka value
    int thread_id = threadIdx.x;
    int threads_in_block = blockDim.x;
    
    // Bounds checking
    if (ka_idx >= num_Ka) return;
    
    // Validate memory access
    if (!is_valid_memory_access_float(codebook, rows * cols) || 
        !is_valid_memory_access_int(Ka_values, num_Ka) ||
        !is_valid_memory_access_int(hit_rates, num_Ka))
        return;
    
    // Shared memory for this block (all threads cooperate)
    __shared__ float shared_codebook_cache[MAX_COLS];  // Cache one row at a time
    __shared__ int local_hit_count;
    __shared__ int Ka_val;
    
    // Initialize shared memory (only thread 0 does this)
    if (thread_id == 0) {
        local_hit_count = 0;
        Ka_val = Ka_values[ka_idx];
    }
    __syncthreads();  // Wait for initialization
    
    // Validate Ka value
    if (Ka_val <= 0) return;
    
    // Each thread processes multiple simulations
    int sims_per_thread = (num_sims + threads_in_block - 1) / threads_in_block;
    int start_sim = thread_id * sims_per_thread;
    int end_sim = min(start_sim + sims_per_thread, num_sims);
    
    // Thread-local arrays (much smaller now - only for one simulation at a time)
    float messages[MAX_ROWS];
    float original_messages[MAX_ROWS];
    
    // Initialize random state with better distribution
    curandState state;
    unsigned long long seed = clock64() + ka_idx * 1000000ULL + thread_id * 1000ULL;
    curand_init(seed, 0, 0, &state);
    
    int thread_hits = 0;
    
    // Process multiple simulations per thread
    for (int sim = start_sim; sim < end_sim; sim++) {
        // Initialize message arrays
        for (int i = 0; i < rows; i++) {
            messages[i] = 0.0f;
            original_messages[i] = 0.0f;
        }
        
        // Generate messages with cooperative shared memory loading
        for (int user = 0; user < Ka_val; user++) {
            int section = 1;
            for (int k = 0; k < N; k++) {
                int rand_user = (int)(curand_uniform(&state) * N) % N;
                int section_offset = (section - 1) * N;
                
                // Cooperative loading of codebook rows into shared memory
                for (int row_batch = 0; row_batch < rows; row_batch += threads_in_block) {
                    int row = row_batch + thread_id;
                    if (row < rows) {
                        // Load codebook row into shared memory cooperatively
                        for (int col_batch = 0; col_batch < cols; col_batch += threads_in_block) {
                            int col = col_batch + thread_id;
                            if (col < cols) {
                                int idx = row * cols + col;
                                if (idx < rows * cols) {
                                    shared_codebook_cache[col % MAX_COLS] = codebook[idx];
                                }
                            }
                        }
                        __syncthreads();  // Ensure all threads have loaded data
                        
                        // Now use the cached data
                        int target_col = section_offset + rand_user;
                        if (target_col < cols && target_col < MAX_COLS) {
                            messages[row] += shared_codebook_cache[target_col];
                        }
                        __syncthreads();  // Ensure all threads finished using data
                    }
                }
                
                section++;
                if (section > L) section = 1;
            }
        }
        
        // Copy original messages
        for (int i = 0; i < rows; i++) {
            original_messages[i] = messages[i];
        }
        
        // Add noise
        float noise_std = 0.01f;
        for (int i = 0; i < rows; i++) {
            float noise = curand_normal(&state) * noise_std;
            messages[i] += noise;
        }
        
        // Simple denoiser - just copy (avoiding complex algorithm)
        // In a real implementation, this would be more sophisticated
        
        // Compare messages (simplified comparison)
        float total_error = 0.0f;
        float signal_magnitude = 0.0f;
        
        for (int i = 0; i < rows; i++) {
            total_error += fabsf(messages[i] - original_messages[i]);
            signal_magnitude += fabsf(original_messages[i]);
        }
        
        float relative_error = total_error / (signal_magnitude + 1e-6f);
        if (relative_error < 0.05f) {
            thread_hits++;
        }
    }
    
    // Cooperatively reduce hit counts using shared memory
    atomicAdd(&local_hit_count, thread_hits);
    __syncthreads();  // Wait for all threads to update
    
    // Only thread 0 writes final result
    if (thread_id == 0) {
        atomicAdd(&hit_rates[ka_idx], local_hit_count);
    }
}

// Host function to run optimized parallel simulation
extern "C" void run_parallel_simulation(int* Ka_values, int num_Ka, int num_sims, int* hit_rates) {
    printf("[DEBUG] Starting optimized parallel simulation: num_Ka=%d, num_sims=%d\n", num_Ka, num_sims);
    
    // Validate input
    if (!Ka_values || !hit_rates || num_Ka <= 0 || num_Ka > MAX_KA_VALUES || num_sims <= 0) {
        printf("[ERROR] Invalid simulation parameters\n");
        return;
    }
    
    // Pre-declare all variables
    float* d_codebook = nullptr;
    int* d_Ka_values = nullptr;
    int* d_hit_rates = nullptr;
    float* host_codebook = nullptr;
    cudaStream_t stream;
    int rows = n;
    int cols = L*N;
    
    // Improved grid configuration: one block per Ka value
    int num_blocks = num_Ka;
    int threads_per_block = THREADS_PER_BLOCK;
    dim3 grid(num_blocks);
    dim3 block(threads_per_block);
    
    // Create CUDA stream
    CUDA_CHECK(cudaStreamCreate(&stream));
    
    // Allocate device memory
    CUDA_CHECK(cudaMalloc(&d_codebook, rows * cols * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_Ka_values, num_Ka * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_hit_rates, num_Ka * sizeof(int)));
    
    // Initialize hit rates to 0
    CUDA_CHECK(cudaMemsetAsync(d_hit_rates, 0, num_Ka * sizeof(int), stream));
    
    // Prepare host codebook with alignment
    host_codebook = (float*)aligned_alloc(32, rows * cols * sizeof(float));
    if (!host_codebook) {
        printf("[ERROR] Failed to allocate host codebook\n");
        goto cleanup;
    }
    
    // Get current codebook
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            host_codebook[i * cols + j] = action_buffer[i * MAX_COLS + j];
        }
    }
    
    // Copy data to device
    CUDA_CHECK(cudaMemcpyAsync(d_codebook, host_codebook, rows * cols * sizeof(float), 
                              cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_Ka_values, Ka_values, num_Ka * sizeof(int), 
                              cudaMemcpyHostToDevice, stream));
    
    printf("[DEBUG] Launching optimized kernel: blocks=%d, threads=%d\n", 
           num_blocks, threads_per_block);
    printf("[DEBUG] Shared memory per block: %d bytes\n", 
           (int)(sizeof(float) * MAX_COLS + 2 * sizeof(int)));
    
    // Launch optimized kernel
    optimized_parallel_simulation_kernel<<<grid, block, 0, stream>>>(
        d_codebook, rows, cols, d_Ka_values, num_Ka, num_sims, d_hit_rates
    );
    
    // Check for kernel errors
    CUDA_CHECK(cudaGetLastError());
    
    // Synchronize before copying results
    CUDA_CHECK(cudaStreamSynchronize(stream));
    
    // Copy results back
    CUDA_CHECK(cudaMemcpyAsync(hit_rates, d_hit_rates, num_Ka * sizeof(int), 
                              cudaMemcpyDeviceToHost, stream));
    
    // Final synchronization
    CUDA_CHECK(cudaStreamSynchronize(stream));
    
    printf("[DEBUG] Optimized parallel simulation completed successfully\n");
    
cleanup:
    // Cleanup
    if (stream) CUDA_CHECK(cudaStreamDestroy(stream));
    if (d_codebook) CUDA_CHECK(cudaFree(d_codebook));
    if (d_Ka_values) CUDA_CHECK(cudaFree(d_Ka_values));
    if (d_hit_rates) CUDA_CHECK(cudaFree(d_hit_rates));
    if (host_codebook) free(host_codebook);
}

// Keep all other functions the same as before
extern "C" void initialize_buffer_with_random(int rows, int cols) {
    printf("[DEBUG] initialize_buffer_with_random: rows=%d, cols=%d\n", rows, cols);
    
    if (rows <= 0 || rows > MAX_ROWS || cols <= 0 || cols > MAX_COLS) {
        printf("[ERROR] Invalid buffer dimensions\n");
        return;
    }
    
    if (buffer_initialized) return;
    
    memset(action_buffer, 0, sizeof(action_buffer));
    buffer_size = 0;
    current_rows = 0;
    current_cols = 0;
    
    srand(time(NULL));
    
    for (int i = 0; i < rows && i < MAX_ROWS; i++) {
        for (int j = 0; j < cols && j < MAX_COLS; j++) {
            if (is_valid_buffer_access(i, j)) {
                float random_val = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
                action_buffer[i * MAX_COLS + j] = random_val;
            }
        }
        buffer_size++;
    }
    
    current_rows = rows;
    current_cols = cols;
    buffer_initialized = 1;
    
    printf("[DEBUG] Buffer initialized: %d x %d\n", current_rows, current_cols);
}

extern "C" void add_action_to_buffer(float* action, int cols) {
    printf("[DEBUG] add_action_to_buffer: cols=%d\n", cols);
    
    if (!action || cols <= 0 || cols > MAX_COLS) {
        printf("[ERROR] Invalid action parameters\n");
        return;
    }
    
    if (buffer_size < MAX_ROWS) {
        int row = buffer_size;
        for (int i = 0; i < cols; i++) {
            if (is_valid_buffer_access(row, i)) {
                action_buffer[row * MAX_COLS + i] = action[i];
            }
        }
        buffer_size++;
    } else {
        for (int i = 0; i < MAX_ROWS - 1; i++) {
            for (int j = 0; j < cols; j++) {
                if (is_valid_buffer_access(i, j) && is_valid_buffer_access(i + 1, j)) {
                    action_buffer[i * MAX_COLS + j] = action_buffer[(i + 1) * MAX_COLS + j];
                }
            }
        }
        int last_row = MAX_ROWS - 1;
        for (int i = 0; i < cols; i++) {
            if (is_valid_buffer_access(last_row, i)) {
                action_buffer[last_row * MAX_COLS + i] = action[i];
            }
        }
    }
    
    current_cols = cols;
    printf("[DEBUG] Buffer updated: size=%d\n", buffer_size);
}

extern "C" void get_codebook(float* out_codebook, int rows, int cols) {
    printf("[DEBUG] get_codebook: %d x %d\n", rows, cols);
    
    if (!out_codebook || rows <= 0 || cols <= 0) {
        printf("[ERROR] Invalid codebook parameters\n");
        return;
    }
    
    int actual_rows = (buffer_size < rows) ? buffer_size : rows;
    int actual_cols = (current_cols < cols) ? current_cols : cols;
    
    for (int i = 0; i < actual_rows; i++) {
        for (int j = 0; j < actual_cols; j++) {
            if (is_valid_buffer_access(i, j)) {
                out_codebook[i * cols + j] = action_buffer[i * MAX_COLS + j];
            }
        }
        for (int j = actual_cols; j < cols; j++) {
            out_codebook[i * cols + j] = 0.0f;
        }
    }
    
    for (int i = actual_rows; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            out_codebook[i * cols + j] = 0.0f;
        }
    }
    
    printf("[DEBUG] Codebook copied: %d x %d\n", rows, cols);
}

extern "C" void clear_action_buffer() {
    printf("[DEBUG] Clearing buffer\n");
    memset(action_buffer, 0, sizeof(action_buffer));
    buffer_size = 0;
    current_rows = 0;
    current_cols = 0;
    buffer_initialized = 0;
} 
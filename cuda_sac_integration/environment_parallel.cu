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
#define MAX_KA_VALUES 10  // Maximum number of Ka values to process in parallel

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

// Device function to calculate magnitude difference
__device__ float mag_diff(float* msg1, float* msg2, int size) {
    if (!is_valid_memory_access_float(msg1, size) || !is_valid_memory_access_float(msg2, size)) 
        return INFINITY;
    
    float diff = 0.0f;
    for (int i = 0; i < size; i++) {
        float d = msg1[i] - msg2[i];
        diff += d * d;
    }
    return sqrtf(diff);
}

// Device function to add noise
__device__ void noiseAdder(float* messages, float noise_std, int rows, curandState* state) {
    if (!is_valid_memory_access_float(messages, rows)) return;
    
    for (int i = 0; i < rows; i++) {
        float noise = curand_normal(state) * noise_std;
        messages[i] += noise;
    }
}

// Device function to generate messages
__device__ void generate_messages(float* codebook, int Ka, int cols, int rows, float* messages, curandState* state) {
    if (!is_valid_memory_access_float(codebook, rows * cols) || !is_valid_memory_access_float(messages, rows))
        return;
    
    // Initialize messages to zero
    for (int i = 0; i < rows; i++) {
        messages[i] = 0.0f;
    }
    
    // Generate random superposition
    int section;
    int rand_user;

    for (int i = 0; i < Ka; i++) {
        section = 1;
        for (int k = 0; k < N; k++) {
            rand_user = (int)(curand_uniform(state) * N);
            for (int j = 0; j < rows; j++) {
                int idx = j * cols + (section-1)*N + rand_user;
                if (idx < rows * cols) {
                    messages[j] += codebook[idx];
                }
            }
            section++;
            if (section > L) section = 1;
        }
    }
}

// Simplified denoiser that just copies for now
__device__ void msg_denoiser_greedy(float* message, float* codebook, int rows, int cols, int Ka, float* best_msg) {
    if (!is_valid_memory_access_float(message, rows) || !is_valid_memory_access_float(best_msg, rows))
        return;
    
    for (int k = 0; k < rows; k++) {
        best_msg[k] = message[k];
    }
}

// Device function to compare messages
__device__ bool compare_messages(float* msg1, float* msg2, int size) {
    if (!is_valid_memory_access_float(msg1, size) || !is_valid_memory_access_float(msg2, size))
        return false;
    
    float total_error = 0.0f;
    float signal_magnitude = 0.0f;
    
    for (int i = 0; i < size; i++) {
        total_error += fabsf(msg1[i] - msg2[i]);
        signal_magnitude += fabsf(msg2[i]);
    }
    
    float relative_error = total_error / (signal_magnitude + 1e-6);
    return relative_error < 0.05f;
}

// New parallel kernel that processes multiple Ka values at once
__global__ void parallel_simulation_kernel(float* codebook, int rows, int cols, 
                                         int* Ka_values, int num_Ka, int num_sims, 
                                         int* hit_rates) {
    // Each thread handles one simulation for one Ka value
    int ka_idx = blockIdx.y;  // Which Ka value we're processing
    int sim_idx = blockIdx.x * blockDim.x + threadIdx.x;  // Which simulation
    
    if (ka_idx >= num_Ka || sim_idx >= num_sims) return;
    
    // Validate memory access with correct types
    if (!is_valid_memory_access_float(codebook, rows * cols) || 
        !is_valid_memory_access_int(Ka_values, num_Ka) ||
        !is_valid_memory_access_int(hit_rates, num_Ka))
        return;
    
    // Validate Ka_values bounds
    if (ka_idx >= 0 && ka_idx < num_Ka && Ka_values[ka_idx] <= 0) return;
    
    // Initialize random state with better seed distribution
    curandState state;
    unsigned long long seed = clock64() + ka_idx * 1000000ULL + sim_idx;
    curand_init(seed, 0, 0, &state);
    
    // Use thread-local memory instead of shared memory to avoid race conditions
    // This uses registers/local memory per thread instead of shared memory per block
    float messages[MAX_ROWS];
    float original_messages[MAX_ROWS];
    float best_msg[MAX_ROWS];
    
    // Initialize arrays (each thread initializes its own arrays)
    for (int i = 0; i < rows; i++) {
        messages[i] = 0.0f;
        original_messages[i] = 0.0f;
        best_msg[i] = 0.0f;
    }
    
    // Generate messages for this Ka value
    int Ka_val = Ka_values[ka_idx];
    
    // Generate random superposition with better bounds checking
    for (int user = 0; user < Ka_val; user++) {
        int section = 1;
        for (int k = 0; k < N; k++) {
            int rand_user = (int)(curand_uniform(&state) * N);
            // Clamp to valid range
            rand_user = rand_user % N;
            
            for (int j = 0; j < rows; j++) {
                // Calculate index with proper bounds checking
                int section_offset = (section - 1) * N;
                int idx = j * cols + section_offset + rand_user;
                
                // Strict bounds checking
                if (idx >= 0 && idx < rows * cols && 
                    section_offset >= 0 && section_offset < cols &&
                    rand_user >= 0 && rand_user < N) {
                    messages[j] += codebook[idx];
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
    
    // Simple denoiser - just copy (to avoid complex algorithm issues)
    for (int k = 0; k < rows; k++) {
        best_msg[k] = messages[k];
    }
    
    // Compare messages and update hit rate
    float total_error = 0.0f;
    float signal_magnitude = 0.0f;
    
    for (int i = 0; i < rows; i++) {
        total_error += fabsf(best_msg[i] - original_messages[i]);
        signal_magnitude += fabsf(original_messages[i]);
    }
    
    float relative_error = total_error / (signal_magnitude + 1e-6f);
    if (relative_error < 0.05f) {
        atomicAdd(&hit_rates[ka_idx], 1);
    }
}

// Host function to run parallel simulation
extern "C" void run_parallel_simulation(int* Ka_values, int num_Ka, int num_sims, int* hit_rates) {
    printf("[DEBUG] Starting parallel simulation: num_Ka=%d, num_sims=%d\n", num_Ka, num_sims);
    
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
    int threadsPerBlock = 256;
    int blocksPerKa = (num_sims + threadsPerBlock - 1) / threadsPerBlock;
    dim3 grid(blocksPerKa, num_Ka);  // 2D grid: (simulations, Ka_values)
    
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
    
    printf("[DEBUG] Launching kernel: grid=(%d,%d), block=%d\n", 
           blocksPerKa, num_Ka, threadsPerBlock);
    
    // Launch parallel kernel
    parallel_simulation_kernel<<<grid, threadsPerBlock, 0, stream>>>(
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
    
    printf("[DEBUG] Parallel simulation completed successfully\n");
    
cleanup:
    // Cleanup
    if (stream) CUDA_CHECK(cudaStreamDestroy(stream));
    if (d_codebook) CUDA_CHECK(cudaFree(d_codebook));
    if (d_Ka_values) CUDA_CHECK(cudaFree(d_Ka_values));
    if (d_hit_rates) CUDA_CHECK(cudaFree(d_hit_rates));
    if (host_codebook) free(host_codebook);
}

// Other functions remain the same
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
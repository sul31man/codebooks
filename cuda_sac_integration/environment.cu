#include <stdio.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <string.h>

// Constants and buffer management
#define MAX_COLS 1024
#define MAX_ROWS 512
#define L 16
#define n 512 // number of rows for the inner encoding matrix
#define N 64 // LN is the number of columns for the inner encoding matrix = 2^J, due to the nature of this encoding scheme
#define j 6 // for the L J bit encoding scheme

// Global variables for persistent buffer (host-side)
float action_buffer[MAX_COLS * MAX_ROWS];  // 2D buffer: rows x cols
int buffer_size = 0;
int buffer_start = 0;
int current_rows = 0;
int current_cols = 0;
int buffer_initialized = 0;  // Flag to track if buffer has been initialized

// Host function to initialize buffer with random matrix (called from Python)
extern "C" void initialize_buffer_with_random(int rows, int cols) {
    // Only initialize if not already initialized
    if (buffer_initialized) {
        return;  // Buffer already initialized, don't reinitialize
    }
    
    // Clear existing buffer
    buffer_size = 0;
    current_rows = 0;
    current_cols = 0;
    
    // Initialize random seed
    srand(time(NULL));
    
    // Fill buffer with random values
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            // Generate random values between -1 and 1
            float random_val = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
            action_buffer[i * MAX_COLS + j] = random_val;
        }
        buffer_size++;
    }
    current_cols = cols;
    current_rows = buffer_size;
    buffer_initialized = 1;  // Mark as initialized
}

//we need to setup the outer encoding scheme so that when we generate the random messages it's used
//clearly we're are making actions such that it adds rows instead of columns to the matrix we want to produce, which is fine, 
//we just need to change a few things later !/

// Host function to add action to buffer (called from Python)
extern "C" void add_action_to_buffer(float* action, int cols) {
    // Add the action as a new row to the buffer
    if (buffer_size < MAX_ROWS) {
        for (int i = 0; i < cols; i++) {
            action_buffer[buffer_size * MAX_COLS + i] = action[i];
        }
        buffer_size++;
    } else {
        // Shift buffer and add new action
        for (int i = 0; i < MAX_ROWS - 1; i++) {
            for (int j = 0; j < cols; j++) {
                action_buffer[i * MAX_COLS + j] = action_buffer[(i + 1) * MAX_COLS + j];
            }
        }
        // Add new action at the end
        for (int i = 0; i < cols; i++) {
            action_buffer[(MAX_ROWS - 1) * MAX_COLS + i] = action[i];
        }
    }
    current_cols = cols;
    current_rows = buffer_size;
}

// Host function to get current codebook (called from Python)
extern "C" void get_codebook(float* out_codebook, int rows, int cols) {
    int actual_rows = (buffer_size < rows) ? buffer_size : rows;
    for (int i = 0; i < actual_rows; i++) {
        for (int j = 0; j < cols; j++) {
            out_codebook[i * cols + j] = action_buffer[i * MAX_COLS + j];
        }
    }
    // Pad with zeros if needed
    for (int i = actual_rows; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            out_codebook[i * cols + j] = 0.0f;
        }
    }
}

// Host function to run simulation (called from Python)
//reminder to add streams here and to look into async functions and data splitting to increase speed
extern "C" void run_simulation(int Ka, int num_sims, int* hit_rate) {
    // Allocate memory on device
    float* d_codebook;
    int* d_hit_rate;
    int rows = n;  // Always use hardcoded n=512
    int cols = L*N;  // Always use hardcoded L*N=16*64=1024
    
    cudaStream_t stream1; 
    cudaStreamCreate(&stream1);

    cudaMalloc(&d_codebook, rows * cols * sizeof(float));
    cudaMalloc(&d_hit_rate, sizeof(int));
    
    // Initialize hit rate to 0
    cudaMemsetAsync(d_hit_rate, 0, sizeof(int), stream1);
    
    // Copy current codebook to device
    float* host_codebook = (float*)malloc(rows * cols * sizeof(float));
    get_codebook(host_codebook, rows, cols);
    cudaMemcpyAsync(d_codebook, host_codebook, rows * cols * sizeof(float), cudaMemcpyHostToDevice, stream1);
    
    // Launch simulation kernel
    int threadsPerBlock = 256;
    int blocks = (num_sims + threadsPerBlock - 1) / threadsPerBlock;
    mainKernel<<<blocks, threadsPerBlock, 0, stream1>>>(d_codebook, rows, cols, Ka, num_sims, d_hit_rate);
    
    // Copy result back to host
    cudaMemcpyAsync(hit_rate, d_hit_rate, sizeof(int), cudaMemcpyDeviceToHost, stream1);
    
    // Synchronize stream before cleanup
    cudaStreamSynchronize(stream1);
    
    // Clean up
    cudaStreamDestroy(stream1);
    cudaFree(d_codebook);
    cudaFree(d_hit_rate);
    free(host_codebook);
}

// Device function to calculate magnitude difference
__device__ float mag_diff(float* msg1, float* msg2, int size) {
    float diff = 0.0f;
    for (int i = 0; i < size; i++) {
        float d = msg1[i] - msg2[i];
        diff += d * d;
    }
    return sqrtf(diff);
}

__device__ void generate_messages(float* codebook, int Ka, int cols, int rows, float* messages) {
    // Initialize messages to zero
    for (int i = 0; i < rows; i++) {
        messages[i] = 0.0f;
    }
    
    // Generate random superposition
    curandState state;
    curand_init(clock64(), blockIdx.x * blockDim.x + threadIdx.x, 0, &state);
    int section;
    int rand_user;

    for (int i = 0; i < Ka; i++) {
        section = 1; // for each user we start back in the first section out of the L sections. 

        for (int k = 0; k < N; k++){

        rand_user = (int)(curand_uniform(&state) * N); // pick a random column out of the N columns in each of the L sections

        for (int j = 0; j < rows; j++) {
            //we need to select a random vector from each of the L sections
            messages[j] += codebook[j*cols + rand_user*section]; //this selects a given column that we want

        
        }
        section++; //so that we start to add the vector from the next section
    }

    }
}

__device__ void noiseAdder(float* messages, float noise_std, int rows) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    curandState state;
    curand_init(clock64(), idx, 0, &state);
    
    if (idx < rows) {
        float noise = curand_normal(&state) * noise_std;
        messages[idx] += noise;
    }
}

__device__ void msg_denoiser_greedy(float* message, float* codebook, int rows, int cols, int Ka, float* best_msg) {
    curandState state;
    curand_init(blockIdx.x * blockDim.x + threadIdx.x + 54321, 0, 0, &state);
    
    float current_msg[10];
    float trial_msg[10];
    float global_best[10];
    float global_best_error = 1e9;
    
    int num_restarts = 5;
    
    for (int restart = 0; restart < num_restarts; restart++) {
        // Initialize current message to zero
        for (int k = 0; k < cols; k++) {
            current_msg[k] = 0.0f;
        }
        
        // Greedy construction phase
        for (int iter = 0; iter < Ka; iter++) {
            float best_improvement = -1e9;
            int best_codeword = 0;
            
            float current_error = mag_diff(message, current_msg, cols);
            
            // Try adding each possible codeword
            for (int i = 0; i < rows; i++) {
                // Build trial message = current + codeword i
                for (int k = 0; k < cols; k++) {
                    trial_msg[k] = current_msg[k] + codebook[i * cols + k];
                }
                
                // Calculate improvement
                float trial_error = mag_diff(message, trial_msg, cols);
                float improvement = current_error - trial_error;
                
                // Add some randomness to break ties
                if (curand_uniform(&state) < 0.1f) {
                    improvement += curand_uniform(&state) * 0.1f;
                }
                
                if (improvement > best_improvement) {
                    best_improvement = improvement;
                    best_codeword = i;
                }
            }
            
            // Add the best codeword
            for (int k = 0; k < cols; k++) {
                current_msg[k] += codebook[best_codeword * cols + k];
            }
        }
        
        // Check if this restart found a better solution
        float final_error = mag_diff(message, current_msg, cols);
        if (final_error < global_best_error) {
            global_best_error = final_error;
            for (int k = 0; k < cols; k++) {
                global_best[k] = current_msg[k];
            }
        }
    }
    
    // Copy best result
    for (int k = 0; k < cols; k++) {
        best_msg[k] = global_best[k];
    }
}

__device__ bool compare_messages(float* msg1, float* msg2, int size) {
    float total_error = 0.0f;
    float signal_magnitude = 0.0f;
    
    for (int i = 0; i < size; i++) {
        total_error += fabsf(msg1[i] - msg2[i]);
        signal_magnitude += fabsf(msg2[i]);
    }
    
    float relative_error = total_error / (signal_magnitude + 1e-6);
    return relative_error < 0.05f;
}

__global__ void mainKernel(float* codebook, int rows, int cols, int Ka, int num_sims, int* hit_rate) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < num_sims) {
        float messages[rows];
        float original_messages[rows];
        
        // Initialize arrays
        for (int i = 0; i < rows; i++) {
            messages[i] = 0.0f;
            original_messages[i] = 0.0f;
        }
        
        // Generate messages
        generate_messages(codebook, Ka, cols, rows, messages);
        
        // Copy original messages
        for (int i = 0; i < rows; i++) {
            original_messages[i] = messages[i];
        }
        
        // Add noise
        float noise_std = 0.01f;
        noiseAdder(messages, noise_std, rows);
        
        // Decode messages
        float best_msg[rows];
        msg_denoiser_greedy(messages, codebook, rows, cols, Ka, best_msg);
        
        // Compare and update hit rate
        if (compare_messages(best_msg, original_messages, cols)) {
            atomicAdd(hit_rate, 1);
        }
    }
} 
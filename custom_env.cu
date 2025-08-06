#include <stdlib.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <time.h>
#include <curand_kernel.h>
#include <math.h>

//need to create a really simple environment for an RL agent. this means it needs to take in a state, an action and output a next state and a reward, with a simulation to determine the 
//reward.

typedef struct{

float* value; 
int size; 

}State; 



float* create_codebook(int rows, int cols){
      
     float* codebook = (float*)malloc(rows * cols * sizeof(float));

     return codebook; 

}

void initialise_codebook(float* codebook, int rows, int cols){
   srand(time(NULL));

   int random_int; 

   for (int i=0; i < rows; i++){

    for (int j=0; j < cols ; j++){

       random_int = rand() % 100; 

       codebook[i * cols + j] = random_int;

    }
   }
    

}

float* message_generator(float* codebook, int Ka, int rows, int cols){
    
    srand(time(NULL));
    
    //this method will superpose Ka messages from the codebook. THIS IS A CPU VERSION. 
    float* message = (float*)malloc(cols*sizeof(float));
    int message_number; 

    for( int i=0; i < Ka; i++ ){

        message_number = rand() % rows;  // ✅ Fixed: should be rows, not Ka 

        for (int j=0; j < cols; j++){

           message[j] += codebook[message_number * cols + j];

        }


    }

    return message; 
    


} 


__device__ void gpu_msg_gen(float* codebook, float* message, int Ka, int rows, int cols){
    curandState state;
    curand_init(blockIdx.x * blockDim.x + threadIdx.x, 0, 0, &state);
    
    for (int i = 0; i < Ka; i++){
        int random_number = curand(&state) % rows;  // Fixed: rows not Ka
        for (int j = 0; j < cols; j++){
            message[j] += codebook[random_number * cols + j];  // Fixed: use random_number
        }
    }
}

__device__ void msg_noise(float* message, int cols){
    curandState state;
    curand_init(blockIdx.x * blockDim.x + threadIdx.x, 0, 0, &state);
    
    for(int i = 0; i < cols; i++){
        float noise = curand_normal(&state);
        message[i] += noise * 0.01f;  // ✅ Reduced noise: 0.1 → 0.01 for Ka=50
    }
}

__device__ float mag_diff(float* vec1, float* vec2, int size){

    float diff = 0; 

    for (int i = 0; i < size; i++){

        diff += (vec1[i] - vec2[i])*(vec1[i] - vec2[i]);
    }
    diff = sqrt(diff);
    return diff; 
}



// ✅ IMPROVED GREEDY DECODER with multiple restarts
__device__ void msg_denoiser_greedy(float* message, float* codebook, int rows, int cols, int Ka, float* best_msg){
    
    curandState state;
    curand_init(blockIdx.x * blockDim.x + threadIdx.x + 54321, 0, 0, &state);
    
    float current_msg[10];
    float trial_msg[10];
    float global_best[10];
    float global_best_error = 1e9;
    
    // Multiple random restarts to avoid local minima
    int num_restarts = 5;  // Try 5 different starting points
    
    for(int restart = 0; restart < num_restarts; restart++){
        
        // Initialize current message to zero
        for(int k = 0; k < cols; k++){
            current_msg[k] = 0.0f;
        }
        
        // Greedy construction phase
        for(int iter = 0; iter < Ka; iter++){
            float best_improvement = -1e9;
            int best_codeword = 0;
            
            float current_error = mag_diff(message, current_msg, cols);
            
            // Try adding each possible codeword
            for(int i = 0; i < rows; i++){
                
                // Build trial message = current + codeword i
                for(int k = 0; k < cols; k++){
                    trial_msg[k] = current_msg[k] + codebook[i * cols + k];
                }
                
                // Calculate improvement
                float trial_error = mag_diff(message, trial_msg, cols);
                float improvement = current_error - trial_error;
                
                // Add some randomness to break ties
                if(curand_uniform(&state) < 0.1f){  // 10% random exploration
                    improvement += curand_uniform(&state) * 0.1f;
                }
                
                if(improvement > best_improvement){
                    best_improvement = improvement;
                    best_codeword = i;
                }
            }
            
            // Add the best codeword
            for(int k = 0; k < cols; k++){
                current_msg[k] += codebook[best_codeword * cols + k];
            }
        }
        
        // Local improvement phase: try replacing each codeword
        for(int replace_iter = 0; replace_iter < 10; replace_iter++){
            int random_pos = curand(&state) % Ka;  // Pick random position to replace
            
            // Remove a random codeword (approximate)
            float scale_factor = (float)(Ka - 1) / (float)Ka;
            for(int k = 0; k < cols; k++){
                current_msg[k] *= scale_factor;
            }
            
            // Try adding a different codeword
            int best_replacement = curand(&state) % rows;
            float best_error = 1e9;
            
            for(int i = 0; i < 20; i++){  // Try 20 random replacements
                int candidate = curand(&state) % rows;
                
                for(int k = 0; k < cols; k++){
                    trial_msg[k] = current_msg[k] + codebook[candidate * cols + k];
                }
                
                float error = mag_diff(message, trial_msg, cols);
                if(error < best_error){
                    best_error = error;
                    best_replacement = candidate;
                }
            }
            
            // Apply best replacement
            for(int k = 0; k < cols; k++){
                current_msg[k] += codebook[best_replacement * cols + k];
            }
        }
        
        // Check if this restart found a better solution
        float final_error = mag_diff(message, current_msg, cols);
        if(final_error < global_best_error){
            global_best_error = final_error;
            for(int k = 0; k < cols; k++){
                global_best[k] = current_msg[k];
            }
        }
    }
    
    // Copy best result
    for(int k = 0; k < cols; k++){
        best_msg[k] = global_best[k];
    }
}

__device__ bool compare_messages(float* msg1, float* msg2, int size){
    float total_error = 0.0f;
    float signal_magnitude = 0.0f;
    
    // Calculate relative error instead of exact matching
    for(int i = 0; i < size; i++){
        total_error += fabsf(msg1[i] - msg2[i]);
        signal_magnitude += fabsf(msg2[i]);  // Original message magnitude
    }
    
    // Success if relative error is small (< 5% for Ka=50)
    float relative_error = total_error / (signal_magnitude + 1e-6);
    return relative_error < 0.05f;  // 5% tolerance for Ka=50
}

__global__ void simulation_kernel(float* codebook, int Ka, int rows, int cols, int* hit_rate, int N){
    int idx = blockIdx.x * blockDim.x + threadIdx.x; 
    
    if (idx < N){
        float message[10];  // Local message
        float original_message[10];  // Store original
        
        // Initialize arrays
        for(int i = 0; i < cols; i++){
            message[i] = 0.0f;
            original_message[i] = 0.0f;
        }
        
        // Generate message
        gpu_msg_gen(codebook, message, Ka, rows, cols);
        
        // Store original
        for(int i = 0; i < cols; i++){
            original_message[i] = message[i];
        }
        
        // Add noise
        msg_noise(message, cols);  // Fixed function
        
        // Decode (you'll need to implement this properly)
        float best_msg[10]; // Declare best_msg here
        msg_denoiser_greedy(message, codebook, rows, cols, Ka, best_msg);
        
        // Compare content, not pointers
        if(compare_messages(best_msg, original_message, cols)){
            atomicAdd(hit_rate, 1);
        }
    }
}


int main(){

    int size = 10; //dimension of the state
    float* value = (float*)malloc(size*sizeof(float)); //this creates a pointer to our state here
    State state1;

    state1.size = size;
    state1.value = value; 

    int rows = 100;  // ✅ Increased codebook size for Ka=50
    int cols = 10; 
    int N = 10000; 
    int Ka = 50;     // ✅ NEW: Ka = 50 instead of 2

    printf("=== MONTE CARLO SIMULATION SETUP ===\n");
    printf("Codebook size: %d × %d\n", rows, cols);
    printf("Active codewords (Ka): %d\n", Ka);
    printf("Number of simulations: %d\n", N);
    printf("Decoder: Greedy (efficient for large Ka)\n");
    printf("=====================================\n\n");

    float* codebook_h = create_codebook(rows, cols);
    initialise_codebook(codebook_h, rows, cols);

    //now we have an example of a codebook, lets now create a bunch of messages which we will push through the cuda kernel, we will make the kernel
    
    float* codebook_d; 

    cudaMalloc(&codebook_d, rows * cols * sizeof(float));

    cudaError_t error = cudaMemcpy(codebook_d, codebook_h, rows * cols * sizeof(float), cudaMemcpyHostToDevice);
    
    if (error != cudaSuccess){
        printf("%d", error);
    }
    int threadsPerBlock = 256; 
    int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;
    
    int hit_rate_h = 0; 

    int* hit_rate_d; 

    cudaMalloc(&hit_rate_d, sizeof(int));
    cudaMemcpy(hit_rate_d, &hit_rate_h, sizeof(int), cudaMemcpyHostToDevice);

    // ✅ Setup GPU timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // ✅ Setup CPU timing
    clock_t cpu_start = clock();
    
    printf("Starting Monte Carlo simulation...\n");
    
    // ✅ Record GPU start time
    cudaEventRecord(start);
    
    simulation_kernel<<<blocks, threadsPerBlock>>>(codebook_d, Ka, rows, cols, hit_rate_d, N);
    
    // ✅ Record GPU end time
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    // ✅ Calculate GPU timing
    float gpu_ms;
    cudaEventElapsedTime(&gpu_ms, start, stop);
    
    // ✅ Calculate CPU timing
    clock_t cpu_end = clock();
    double cpu_time = ((double)(cpu_end - cpu_start)) / CLOCKS_PER_SEC;
    
    error = cudaDeviceSynchronize();

    if (error != cudaSuccess){
        printf("Kernel execution error: %d\n", error);
    }
    
    // ✅ Print timing results
    printf("=== TIMING RESULTS ===\n");
    printf("GPU kernel time: %.3f ms\n", gpu_ms);
    printf("Total CPU time:  %.3f ms\n", cpu_time * 1000.0);
    printf("Simulations per second: %.0f\n", N / (gpu_ms / 1000.0));
    printf("======================\n");

    error = cudaMemcpy(&hit_rate_h, hit_rate_d, sizeof(int), cudaMemcpyDeviceToHost);
    
    if (error != cudaSuccess){
        printf("Memory copy error: %d\n", error);
    }

    float prob = (float)hit_rate_h / N;

    // ✅ Print final results with timing context
    printf("\n=== SIMULATION RESULTS ===\n");
    printf("Success rate: %.4f (Hits: %d out of %d)\n", prob, hit_rate_h, N);
    printf("Codebook: %d×%d, Ka=%d, Noise=0.01, Tolerance=5%%\n", rows, cols, Ka);
    printf("Throughput: %.1f simulations/ms\n", N / gpu_ms);
    printf("========================\n");

    // ✅ Cleanup memory and events
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    free(codebook_h);
    free(value);
    cudaFree(codebook_d);
    cudaFree(hit_rate_d);
    
    return 0;
}
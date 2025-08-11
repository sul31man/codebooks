#include <torch/extension.h>
#include <curand_kernel.h> 
#include <cuda_runtime.h>
#include <stdio.h> 
#include <time.h>
#include <vector>
#include <cstring>  // For memset 

//can we make the codebook a torch tensor ? 
int rows = 512; 
int cols = 1024;
int state_size = 1;
int N = 64;
int L = 16;
int J = 6;
int Ka = 20;
int mc_simulations = 10000;  // Number of Monte Carlo simulations per environment



// Global codebook tensor in device memory
torch::Tensor global_codebook;
bool codebook_initialised = false;

// Function to initialise the global codebook tensor
torch::Tensor initialise_global_codebook(int batch_size) {
    if (!codebook_initialised) {
        // Create tensor on CUDA device with random values
        auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
        global_codebook = torch::randn({batch_size, 512, 1024}, options);
        codebook_initialised = true;
        
        printf("Global codebook initialised with shape [%d, 512, 1024]\n", batch_size);
    }
    return global_codebook;
}


void update_global_codebook(torch::Tensor actions, int batch_size){
   
    for (int i=0; i < batch_size; i++){

        torch::Tensor our_codebook = global_codebook[i];
        


        //now we shift columns left and add new action to last column
        if (cols > 1) {
            auto shifted = our_codebook.index({torch::indexing::Slice(), torch::indexing::Slice(1, cols)});
            our_codebook.index({torch::indexing::Slice(), torch::indexing::Slice(0, cols-1)}) = shifted;
        }
        // Add new action to last column 
        our_codebook.index({torch::indexing::Slice(), -1}) = actions[i];

    }

}



torch::Tensor reset_batch(int batch_size){
    
    //I don't think we need a reset function because it should just start us off with a random state again 
    //should give us a batch of random states
    //no need to even clear the codebooks

    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    torch::Tensor states = torch::randn({batch_size, state_size}, options);

    return states; 


}



/
__device__ void select_messages(const float* env_codebook,
                                int rows,
                                int cols,
                                int L,
                                int N,
                                float* messages,
                                curandState* state){
    // Accumulate a superposition of one randomly selected column from each section
    for (int i = 0; i < L; i++){
        int random_col = (int)(curand_uniform(state) * N);
        if (random_col >= N) random_col = N - 1; // guard against edge case
        int colIndex = i * N + random_col;      // absolute column within [0, cols)
        for (int j = 0; j < rows; j++){
            messages[j] += env_codebook[j * cols + colIndex];
        }
    }
}


//simulator kernel should take in the batched actions, and each thread should simulate the effect of the action. But we do require a monte carlo simulation... therefore we shold make it such that 
//if we require n MC simulations, the first n threads work on the first action, the second n MC simulations work on the second action and so on. 
__global__ void simulatorKernel(float* actions, int batch_size, int mc_simulations, float* global_codebook, int* hit_rates, int rows, int cols, int Ka, int L, int N) {
   
    int global_idx = blockIdx.x * blockDim.x + threadIdx.x; 
    int env_idx = global_idx / mc_simulations;  // Which environment
    int mc_idx = global_idx % mc_simulations;   // Which MC simulation
    int thread_id = threadIdx.x; 

    // Important: do not early-return before __syncthreads().
    // Guard computation with a flag instead.
    bool is_valid_env = (env_idx < batch_size);

    // Declare shared memory properly
    __shared__ int shared_hits[256];  // Assuming max 256 threads per block
    shared_hits[thread_id] = 0;
    __syncthreads();  // Wait for initialization

        curandState state;
        if (is_valid_env) {
            curand_init(clock64(), global_idx, 0, &state);
        }
        
        // Point to the correct environment's codebook
        const float* env_codebook = is_valid_env ? &global_codebook[env_idx * rows * cols] : nullptr;

        float messages[512];
        float original_messages[512]; // for comparisons later

        int thread_hits = 0;
        if (is_valid_env) {
            // Initialize local buffers
            for (int i = 0; i < rows; i++) messages[i] = 0.0f;

            // Implement the channel message for the users we have by accumulating selections
            for (int user = 0; user < Ka; user++){
                select_messages(env_codebook, rows, cols, L, N, messages, &state);
            }

            for (int i = 0; i < rows; i++){
                original_messages[i] = messages[i];
            }

            // Add noise
            float noise_std = 0.01f;
            for (int i = 0; i < rows; i++) {
                float noise = curand_normal(&state) * noise_std;
                messages[i] += noise;
            }

            // Simple decoder: relative error threshold
            float total_error = 0.0f;
            float signal_magnitude = 0.0f;
            for (int i = 0; i < rows; i++) {
                total_error += fabsf(messages[i] - original_messages[i]);
                signal_magnitude += fabsf(original_messages[i]);
            }
            
            float relative_error = total_error / (signal_magnitude + 1e-6f);
            if (relative_error < 0.05f) {
                thread_hits = 1;
            }
        }

        // Store thread hits in shared memory
        shared_hits[thread_id] = thread_hits;
        __syncthreads();  // Wait for all threads to store their results
        
        // Reduction in shared memory (safe pattern)
        // Only add to hit_rates if this thread belongs to the current environment
        if (thread_id == 0) {
            int total_hits = 0;
            for (int i = 0; i < blockDim.x; i++) {
                // Only count hits from threads in the same environment
                int thread_env = (blockIdx.x * blockDim.x + i) / mc_simulations;
                if (thread_env == env_idx && thread_env < batch_size) {
                    total_hits += shared_hits[i];
                }
            }
            if (is_valid_env) {
                atomicAdd(&hit_rates[env_idx], total_hits);
            }
        }

}



std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> step(torch::Tensor actions){

    //first we need to update the global codebook
    int batch_size = actions.size(0);
    printf("Environment step called with actions shape: [%d, %d]\n", (int)actions.size(0), (int)actions.size(1));
    
    try {
        update_global_codebook(actions, batch_size);
        printf("Codebook update completed successfully\n");
    } catch (const std::exception& e) {
        printf("Error in codebook update: %s\n", e.what());
        return std::make_tuple(torch::tensor({}), torch::tensor({}), torch::tensor({}));
    }

    // Calculate kernel launch parameters
    int threads_per_block = 256;
    int blocks_per_env = (mc_simulations + threads_per_block - 1) / threads_per_block;
    int total_blocks = batch_size * blocks_per_env;
    
    // Allocate memory for hit rates
    int* hit_rates_d;
    int* hit_rates_h = (int*)malloc(batch_size * sizeof(int));
    
    // Initialize host memory
    memset(hit_rates_h, 0, batch_size * sizeof(int));
    
    cudaError_t error = cudaMalloc(&hit_rates_d, batch_size * sizeof(int));
    if (error != cudaSuccess) {
        printf("CUDA malloc failed: %s\n", cudaGetErrorString(error));
        free(hit_rates_h);
        return std::make_tuple(torch::tensor({}), torch::tensor({}), torch::tensor({}));
    }
    
    // Initialize device memory
    cudaMemset(hit_rates_d, 0, batch_size * sizeof(int));

    //now lets run the cuda kernels so that we can get the hit rates 
    dim3 gridDim(total_blocks);
    dim3 blockDim(threads_per_block);
    
    simulatorKernel<<<gridDim, blockDim>>>(
        actions.data_ptr<float>(),
        batch_size, 
        mc_simulations, 
        global_codebook.data_ptr<float>(), 
        hit_rates_d,
        rows,    // Pass host variables as parameters
        cols,
        Ka,
        L,
        N
    );

    error = cudaDeviceSynchronize();
    if (error != cudaSuccess) {
        printf("Kernel execution failed: %s\n", cudaGetErrorString(error));
        cudaFree(hit_rates_d);
        free(hit_rates_h);
        return std::make_tuple(torch::tensor({}), torch::tensor({}), torch::tensor({}));
    }

    // Copy results back to host
    error = cudaMemcpy(hit_rates_h, hit_rates_d, batch_size * sizeof(int), cudaMemcpyDeviceToHost);
    if (error != cudaSuccess) {
        printf("CUDA memcpy failed: %s\n", cudaGetErrorString(error));
        cudaFree(hit_rates_d);
        free(hit_rates_h);
        return std::make_tuple(torch::tensor({}), torch::tensor({}), torch::tensor({}));
    }

    // Convert results to PyTorch tensors
    auto options = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCPU);
    torch::Tensor hit_rates_tensor = torch::from_blob(hit_rates_h, {batch_size}, options).clone();
    
    // Calculate hit rates as percentage
    torch::Tensor hit_rate_percentages = hit_rates_tensor.to(torch::kFloat32) / mc_simulations;
    
    // Create dummy rewards and dones for now (you can implement proper logic later)
    torch::Tensor rewards = hit_rate_percentages;  // Use hit rate as reward
    torch::Tensor dones = torch::zeros({batch_size}, torch::TensorOptions().dtype(torch::kBool));
    
    // Cleanup
    cudaFree(hit_rates_d);
    free(hit_rates_h);

    return std::make_tuple(hit_rate_percentages, rewards, dones);
}

// Function to get the current global codebook
torch::Tensor get_codebook() {
    if (!codebook_initialised) {
        printf("Warning: Codebook not initialized yet!\n");
        return torch::empty({0, 0, 0});
    }
    return global_codebook;
}

// Python bindings
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("initialise_global_codebook", &initialise_global_codebook, "Initialize global codebook");
    m.def("step", &step, "Environment step function");
    m.def("reset_batch", &reset_batch, "Reset environment batch");
    m.def("get_codebook", &get_codebook, "Get current global codebook");
}



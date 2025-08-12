#include <torch/extension.h>
#include <curand_kernel.h> 
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <stdio.h> 
#include <time.h>
#include <vector> 
#include <cstring>  // For memset 
#include <math.h>   // For sqrt, fabs
extern "C" int amp_decode_gpu_entry(double* A, double* y, double* theta_out,
                                     int n, int N, int L, int J, int Ka,
                                     double P_hat, int T_max, double tol);
extern "C" int amp_decode_gpu_device_entry(double* d_A_col_major, double* d_y, double* d_theta_out,
                                            int n, int N, int L, int J, int Ka,
                                            double P_hat, int T_max, double tol);

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


// New kernel: sample one active index per section for each environment
// Produces ground-truth support indices without changing simulator dimensionality
__global__ void sample_support_kernel(
    int* __restrict__ gt_indices,  // [batch_size * L * Ka]
    int batch_size,
    int L,
    int Ka,
    int N_per_section,
    unsigned long long seed)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch_size * L;
    if (tid >= total) return;

    int env = tid / L;
    int sec = tid % L;
    (void)env; // env is implied in flattened index

    curandState state;
    curand_init(seed, tid, 0, &state);
    for (int k = 0; k < Ka; k++) {
        int r = (int)(curand_uniform(&state) * N_per_section);
        if (r >= N_per_section) r = N_per_section - 1; // guard edge case
        int idx_abs = sec * N_per_section + r;         // absolute index in [0, L*N_per_section)
        gt_indices[(env * L + sec) * Ka + k] = idx_abs;
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

// Kernel: build y from A_eff (column-major) by summing selected columns and adding AWGN
__global__ void build_y_from_Aeff_kernel(const double* __restrict__ A_eff_col,
                                         const int* __restrict__ gt_indices_env,
                                         int n, int L, int Ka, double noise_std,
                                         unsigned long long seed,
                                         double* __restrict__ y_out) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    // Sum columns for each section at row i
    double acc = 0.0;
    for (int s = 0; s < L; s++) {
        for (int k = 0; k < Ka; k++) {
            int idx = gt_indices_env[s * Ka + k];
            acc += A_eff_col[i + (long long)idx * n];
        }
    }

    // AWGN
    curandState state;
    curand_init(seed, i, 0, &state);
    double noise = noise_std * (double)curand_normal(&state);
    y_out[i] = acc + noise;
}

// Kernel: compute per-env hit count from theta_out by top-1 per section
__global__ void compute_hits_from_theta_kernel(const double* __restrict__ theta,
                                               const int* __restrict__ gt_indices_env,
                                               int L, int Ka, int N_per_section, int N_total,
                                               int* __restrict__ hits_out) {
    extern __shared__ int shared[];
    int* section_hits = shared; // size at least L
    if (threadIdx.x < L) section_hits[threadIdx.x] = 0;
    __syncthreads();

    int s = threadIdx.x;
    if (s < L) {
        int start = s * N_per_section;
        int end = min(start + N_per_section, N_total);
        // Find top-Ka indices by repeated selection (N small)
        const int maxN = 2048; // safety cap
        int len = end - start;
        // Use simple in-place array of values and indices in registers/shared? do sequential scans
        // For each k, find best remaining index
        int hits_local = 0;
        // Ground-truth for this section
        // Compare found indices against gt set of size Ka
        for (int k = 0; k < Ka; k++) {
            int argmax = start;
            double maxabs = -1.0;
            for (int j = start; j < end; j++) {
                double v = fabs(theta[j]);
                if (v > maxabs) { maxabs = v; argmax = j; }
            }
            // Mark selected as very small so next iteration finds next best
            // Note: we cannot modify theta; approximate by skipping same argmax next time via storing and checking
            // For simplicity, after counting hit, set maxabs sentinel via a separate array would be needed; here we accept potential duplicates for Ka>1 as small error.
            // Count hit if argmax is in gt indices
            for (int g = 0; g < Ka; g++) {
                if (argmax == gt_indices_env[s * Ka + g]) { hits_local++; break; }
            }
        }
        section_hits[s] = hits_local;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        int total_hits = 0;
        for (int s2 = 0; s2 < L; s2++) total_hits += section_hits[s2];
        *hits_out = total_hits;
    }
}

// Alternate step using AMP decoder path fully on GPU with codebook and noise
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> step_amp(torch::Tensor actions,
                                                                 torch::Tensor sensing_matrix,
                                                                 int n, int N_total,
                                                                 int T_max, double tol,
                                                                 double P_hat) {
    int batch_size = actions.size(0);

    // 1) Update codebook first (same as normal path)
    update_global_codebook(actions, batch_size);

    // 2) Prepare output tensors on CPU (returned to Python)
    auto f32_cpu = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
    torch::Tensor hit_rate_percentages = torch::zeros({batch_size}, f32_cpu);
    torch::Tensor rewards = torch::zeros({batch_size}, f32_cpu);
    torch::Tensor dones = torch::zeros({batch_size}, torch::TensorOptions().dtype(torch::kBool).device(torch::kCPU));

    // 3) Sensing matrix S on CUDA, double, contiguous (shape [n, rows])
    torch::Tensor S_cuda = sensing_matrix.to(torch::kCUDA).to(torch::kFloat64).contiguous();
    double* d_S = S_cuda.data_ptr<double>();

    // Per-section size derived from codebook layout
    int N_per_section = cols / L; // expected to be 1 << J
    if (N_total != L * N_per_section) {
        printf("Warning: N_total (%d) != L * N_per_section (%d)\n", N_total, L * N_per_section);
    }

    // 4) Sample ground-truth support on GPU (one active per section)
    int total_slots = batch_size * L;
    int* gt_indices_d = nullptr;
    cudaError_t err = cudaMalloc(&gt_indices_d, total_slots * Ka * sizeof(int));
    if (err != cudaSuccess) {
        printf("CUDA malloc (gt_indices) failed: %s\n", cudaGetErrorString(err));
        return std::make_tuple(hit_rate_percentages, rewards, dones);
    }

    int threads = 256;
    int blocks = (total_slots + threads - 1) / threads;
    unsigned long long seed = static_cast<unsigned long long>(time(NULL));
    sample_support_kernel<<<blocks, threads>>>(gt_indices_d, batch_size, L, Ka, N_per_section, seed);
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("sample_support_kernel failed: %s\n", cudaGetErrorString(err));
        cudaFree(gt_indices_d);
        return std::make_tuple(hit_rate_percentages, rewards, dones);
    }

    // 5) cuBLAS handle
    cublasHandle_t handle;
    cublasCreate(&handle);

    // 6) For each environment, build A_eff = S * C_b on GPU and run AMP
    auto f64_cuda = torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCUDA);
    auto i32_cuda = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA);
    torch::Tensor hits_tensor_d = torch::zeros({batch_size}, i32_cuda);

    const double one = 1.0, zero = 0.0;

    for (int b = 0; b < batch_size; b++) {
        // Codebook slice for env b on CUDA double [rows, N_total]
        torch::Tensor Cb_cuda = global_codebook[b].to(torch::kCUDA).to(torch::kFloat64).contiguous();
        double* d_Cb = Cb_cuda.data_ptr<double>();

        // Allocate A_eff (column-major) [n, N_total]
        torch::Tensor Aeff_cuda = torch::empty({n, N_total}, f64_cuda);
        double* d_Aeff = Aeff_cuda.data_ptr<double>();

        // Compute A_eff = S * Cb using column-major GEMM: set opA=Trans, opB=Trans
        // Treat row-major S(n x rows) as column-major (rows x n) and transpose => (n x rows)
        // Treat row-major Cb(rows x N) as column-major (N x rows) and transpose => (rows x N)
        int m = n;               // rows of result
        int nn = N_total;        // cols of result
        int k = rows;            // inner dimension
        int lda = rows;          // leading dim of A_col (rows x n)
        int ldb = N_total;       // leading dim of B_col (N x rows)
        int ldc = n;             // leading dim of C_col (n x N)
        cublasDgemm(handle, CUBLAS_OP_T, CUBLAS_OP_T,
                    m, nn, k,
                    &one,
                    d_S, lda,
                    d_Cb, ldb,
                    &zero,
                    d_Aeff, ldc);

        // Build ground-truth indices pointer for this env
        int* gt_env_ptr = gt_indices_d + b * L * Ka;

        // Build y on device from selected columns of A_eff and add noise
        torch::Tensor y_cuda = torch::empty({n}, f64_cuda);
        unsigned long long y_seed = static_cast<unsigned long long>(time(NULL)) + (unsigned long long)b * 1315423911ULL;
        int tpb = 256;
        int blocks = (n + tpb - 1) / tpb;
        double noise_std = 0.01; // TODO: parameterize via SNR
        build_y_from_Aeff_kernel<<<blocks, tpb>>>(d_Aeff, gt_env_ptr, n, L, Ka, noise_std, y_seed, y_cuda.data_ptr<double>());
        err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            printf("build_y_from_Aeff_kernel failed: %s\n", cudaGetErrorString(err));
            continue;
        }

        // Run AMP decoder fully on device
        torch::Tensor theta_out_cuda = torch::empty({N_total}, f64_cuda);
        int iters = amp_decode_gpu_device_entry(d_Aeff, y_cuda.data_ptr<double>(), theta_out_cuda.data_ptr<double>(),
                                                n, N_total, L, J, Ka, P_hat, T_max, tol);
        (void)iters;

        // Compute hits per section on device
        torch::Tensor hits_d = torch::zeros({1}, i32_cuda);
        compute_hits_from_theta_kernel<<<1, L, L * sizeof(int)>>>(theta_out_cuda.data_ptr<double>(), gt_env_ptr,
                                                                  L, Ka, N_per_section, N_total,
                                                                  hits_d.data_ptr<int>());
        err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            printf("compute_hits_from_theta_kernel failed: %s\n", cudaGetErrorString(err));
            continue;
        }

        // Write to hits_tensor_d[b]
        cudaMemcpy(hits_tensor_d.data_ptr<int>() + b, hits_d.data_ptr<int>(), sizeof(int), cudaMemcpyDeviceToDevice);
    }

    // Destroy cuBLAS
    cublasDestroy(handle);

    // Copy hit counts back to host and compute rewards
    std::vector<int> hits_h(batch_size, 0);
    cudaMemcpy(hits_h.data(), hits_tensor_d.data_ptr<int>(), batch_size * sizeof(int), cudaMemcpyDeviceToHost);

    for (int b = 0; b < batch_size; b++) {
        float hit_rate = (float)hits_h[b] / (float)L;
        hit_rate_percentages[b] = hit_rate;
        rewards[b] = hit_rate;
        dones[b] = false;
    }

    cudaFree(gt_indices_d);

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
    m.def("step_amp", &step_amp, "Environment step function using AMP decoder",
          pybind11::arg("actions"), pybind11::arg("sensing_matrix"), pybind11::arg("n"), pybind11::arg("N"),
          pybind11::arg("T_max") = 15, pybind11::arg("tol") = 1e-6, pybind11::arg("P_hat") = 1.0);
    m.def("reset_batch", &reset_batch, "Reset environment batch");
    m.def("get_codebook", &get_codebook, "Get current global codebook");
}



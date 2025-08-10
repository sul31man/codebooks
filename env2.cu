//lets create a more complicated example now 
#include <stdio.h>
#include <stdlib.h> 
#include <torch/extension.h> 
#include <cuda_runtime.h> 
#include <curand_kernel.h> 
#include <random>
#include <vector> 
#include <iostream> 
//this will the exact same thing as CartPole-v1 but with a CUDA kernel environment instead

// CUDA kernel for resetting environments on GPU
__global__ void reset_environments_kernel(
    float* states,      // [batch_size, 4] output
    int batch_size,
    unsigned int seed
) {
    int env_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (env_idx < batch_size) {
        // Initialize random state for this environment
        curandState rng_state;
        curand_init(seed + env_idx, 0, 0, &rng_state);
        
        // Get pointer to this environment's state [4 values]
        float* env_state = &states[env_idx * 4];
        
        // Generate CartPole reset state: 4 values in [-0.05, 0.05]
        for (int i = 0; i < 4; i++) {
            env_state[i] = (curand_uniform(&rng_state) - 0.5f) * 0.1f;
        }
    }
}

// Pure GPU single environment reset

__global__ void env_step_kernel(
    float* states,      // [batch_size, 4] - current states
    float* actions,     // [batch_size] - actions (0 or 1)
    float* next_states, // [batch_size, 4] - output next states
    float* rewards,     // [batch_size] - output rewards
    bool* dones,        // [batch_size] - output termination flags
    int batch_size
){
    int env_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (env_idx < batch_size){
        // CartPole physics constants
        const float gravity = 9.8f;
        const float masscart = 1.0f;
        const float masspole = 0.1f;
        const float total_mass = masspole + masscart;
        const float length = 0.5f;  // half-pole length
        const float polemass_length = masspole * length;
        const float force_mag = 10.0f;
        const float tau = 0.02f;  // time step
        
        // Thresholds for termination
        const float x_threshold = 2.4f;
        const float theta_threshold_radians = 12.0f * 2.0f * M_PI / 360.0f;  // 12 degrees
        
        // Get current state
        float x = states[env_idx * 4 + 0];      // cart position
        float x_dot = states[env_idx * 4 + 1];  // cart velocity
        float theta = states[env_idx * 4 + 2];  // pole angle
        float theta_dot = states[env_idx * 4 + 3]; // pole angular velocity
        
        // Get action (0 = left, 1 = right)
        float force = (actions[env_idx] > 0.5f) ? force_mag : -force_mag;
        
        // Physics calculations
        float costheta = cosf(theta);
        float sintheta = sinf(theta);
        
        // Temporary variable for dynamics
        float temp = (force + polemass_length * theta_dot * theta_dot * sintheta) / total_mass;
        
        // Angular acceleration
        float thetaacc = (gravity * sintheta - costheta * temp) / 
                        (length * (4.0f/3.0f - masspole * costheta * costheta / total_mass));
        
        // Linear acceleration
        float xacc = temp - polemass_length * thetaacc * costheta / total_mass;
        
        // Update state using Euler integration
        float new_x = x + tau * x_dot;
        float new_x_dot = x_dot + tau * xacc;
        float new_theta = theta + tau * theta_dot;
        float new_theta_dot = theta_dot + tau * thetaacc;
        
        // Store next state
        next_states[env_idx * 4 + 0] = new_x;
        next_states[env_idx * 4 + 1] = new_x_dot;
        next_states[env_idx * 4 + 2] = new_theta;
        next_states[env_idx * 4 + 3] = new_theta_dot;
        
        // Check termination conditions
        bool done = (new_x < -x_threshold) || (new_x > x_threshold) ||
                   (new_theta < -theta_threshold_radians) || (new_theta > theta_threshold_radians);
        
        // Reward: +1 for each step the pole stays up
        rewards[env_idx] = done ? 0.0f : 1.0f;
        dones[env_idx] = done;
    }
}

// Pure GPU batched environment reset
torch::Tensor env_reset(int batch_size){
    // Input validation
    TORCH_CHECK(batch_size > 0, "batch_size must be positive");
    TORCH_CHECK(batch_size <= 1000000, "batch_size too large (max 1M environments)");
    
    // Create tensor on GPU (already contiguous)
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    torch::Tensor states = torch::zeros({batch_size, 4}, options);
    
    // Launch kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (batch_size + threadsPerBlock - 1) / threadsPerBlock;
    unsigned int seed = static_cast<unsigned int>(time(nullptr));
    
    reset_environments_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        states.data_ptr<float>(),
        batch_size,
        seed
    );
    
    // Check for CUDA errors
    cudaError_t err = cudaDeviceSynchronize();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed: ", cudaGetErrorString(err));
    
    return states;
}

// C++ wrapper function for CartPole environment step
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> env_step(
    torch::Tensor states,   // [batch_size, 4] current states
    torch::Tensor actions   // [batch_size] actions (0 or 1)
) {
    // Input validation
    TORCH_CHECK(states.device().is_cuda(), "states must be on CUDA device");
    TORCH_CHECK(actions.device().is_cuda(), "actions must be on CUDA device");
    TORCH_CHECK(states.dtype() == torch::kFloat32, "states must be float32");
    TORCH_CHECK(actions.dtype() == torch::kFloat32, "actions must be float32");
    TORCH_CHECK(states.dim() == 2 && states.size(1) == 4, "states must be [batch_size, 4]");
    TORCH_CHECK(actions.dim() == 1, "actions must be [batch_size]");
    TORCH_CHECK(states.size(0) == actions.size(0), "batch size mismatch between states and actions");
    
    // Ensure contiguous memory layout
    states = states.contiguous();
    actions = actions.contiguous();
    
    int batch_size = states.size(0);
    auto device = states.device();
    auto float_options = torch::TensorOptions().dtype(torch::kFloat32).device(device);
    auto bool_options = torch::TensorOptions().dtype(torch::kBool).device(device);
    
    // Create output tensors (already contiguous)
    torch::Tensor next_states = torch::zeros({batch_size, 4}, float_options);
    torch::Tensor rewards = torch::zeros({batch_size}, float_options);
    torch::Tensor dones = torch::zeros({batch_size}, bool_options);
    
    // Launch kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (batch_size + threadsPerBlock - 1) / threadsPerBlock;
    
    env_step_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        states.data_ptr<float>(),
        actions.data_ptr<float>(),
        next_states.data_ptr<float>(),
        rewards.data_ptr<float>(),
        dones.data_ptr<bool>(),
        batch_size
    );
    
    // Check for CUDA errors
    cudaError_t err = cudaDeviceSynchronize();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed: ", cudaGetErrorString(err));
    
    return std::make_tuple(next_states, rewards, dones);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m){
    m.def("env_reset", &env_reset, "Reset batch of environments to CartPole initial state");
    m.def("env_step", &env_step, "Step CartPole environments with given actions");
}
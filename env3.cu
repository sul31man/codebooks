//lets create a more complicated example now 
#include <stdio.h>
#include <stdlib.h> 
#include <torch/extension.h> 
#include <cuda_runtime.h> 
#include <curand_kernel.h> 
#include <random>
#include <vector> 
#include <iostream> 



__global__ void reset_envs_kernel(float* states, int batch_size, unsigned int seed){

    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx < batch_size){

        curandState rng_state; 
        curand_init(seed + env_idx, 0, 0, &rng_state);

        float* env_state = &states[env_idx * 4];

        for (int i =0; i < 4; i++){

            env_state[4*env_idx + i] = (curand_uniform(&rng_state) -0.5f)*0.1f;

        }
    }
}

torch::Tensor env_reset_batch(int batch_size){


    TORCH_CHECK(batch_size > 0, "batch size must be positive");
    TORCH_CHECK(batch_size < 10000000, "batch size is to large");

    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    torch::Tensor states = torch::zeros((batch_size, 4), options);

    int threadsPerBlock = 256; 
    int blocksPerGrid = (threadsPerBlock + batch_size -1)/ (threadsPerBlock); 
    unsigned int seed = static_cast<unsigned int>(time(nullptr));

    reset_envs_kernel<<<blocksPerGrid, threadsPerBlock>>>(states.data_ptr<float>(),batch_size, seed );

    cudaError_t err = cudaDeviceSyncrhonize();

    TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed: ", cudaGetErrorString(err));

    return states; 

}


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

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> env_step(
    torch::Tensor states, torch::Tensor actions //taking in the state and the action 
){

   

}
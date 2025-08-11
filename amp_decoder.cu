#include <stdio.h> 
#include <stdlib.h>
#include <math.h> 
#include <string.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <curand_kernel.h>

// Function to compute binomial coefficient C(n,k)
double binomial_coeff(int n, int k) {
    if (k > n || k < 0) return 0.0;
    if (k == 0 || k == n) return 1.0;
    
    double result = 1.0;
    for (int i = 0; i < k; i++) {
        result = result * (n - i) / (i + 1);
    }
    return result;
}

// Vector norm computation
double vector_norm(double* vec, int size) {
    double sum = 0.0;
    for (int i = 0; i < size; i++) {
        sum += vec[i] * vec[i];
    }
    return sqrt(sum);
}

// Matrix-vector multiplication: result = A * x
void matvec_multiply(double* A, double* x, double* result, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        result[i] = 0.0;
        for (int j = 0; j < cols; j++) {
            result[i] += A[i * cols + j] * x[j];
        }
    }
}

// Matrix-vector multiplication (transpose): result = A^T * x
void matvec_multiply_transpose(double* A, double* x, double* result, int rows, int cols) {
    for (int j = 0; j < cols; j++) {
        result[j] = 0.0;
        for (int i = 0; i < rows; i++) {
            result[j] += A[i * cols + j] * x[i];
        }
    }
}

// CUDA kernel for element-wise operations
__global__ void elementwise_add_kernel(double* result, double* a, double* b, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        result[idx] = a[idx] + b[idx];
    }
}

__global__ void elementwise_subtract_kernel(double* result, double* a, double* b, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        result[idx] = a[idx] - b[idx];
    }
}

__global__ void vector_scale_kernel(double* vec, double scale, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        vec[idx] *= scale;
    }
}

// CUDA kernel for AMP denoiser (vectorized over all N positions)
__global__ void amp_denoiser_kernel(double* v, double* theta_next, double* f_prime_out,
                                    double* p_k, int Ka, double sqrt_P, double tau, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        double numerator = 0.0;
        double denominator = 0.0;
        double f_prime_sum = 0.0;
        
        // Compute weighted sum over k = 0, 1, ..., Ka
        double g_prime_num = 0.0;
        double Z_prime = 0.0;
        
        for (int k = 0; k <= Ka; k++) {
            double mu_k = k * sqrt_P;
            double diff = v[idx] - mu_k;
            double exp_arg = -diff * diff / (2.0 * tau * tau);
            
            // Clip exponential argument for numerical stability
            if (exp_arg < -50.0) exp_arg = -50.0;
            
            double exp_val = exp(exp_arg);
            double weight = p_k[k] * exp_val;
            
            numerator += k * weight;
            denominator += weight;
            
            // Compute derivative terms for Onsager correction (quotient rule)
            double diff_over_tau2 = diff / (tau * tau);
            g_prime_num += (-k * diff_over_tau2) * weight;
            Z_prime += (-diff_over_tau2) * weight;
        }
        
        // Set θ_{t+1}[idx]
        if (denominator > 1e-15) {
            theta_next[idx] = sqrt_P * numerator / denominator;
            // Apply quotient rule: f'(v) = sqrt_P * (g'*Z - g*Z') / Z^2
            f_prime_out[idx] = sqrt_P * (g_prime_num * denominator - numerator * Z_prime) / (denominator * denominator);
        } else {
            theta_next[idx] = 0.0;
            f_prime_out[idx] = 0.0;
        }
    }
}

// GPU-accelerated AMP decoder using cuBLAS
int amp_decode_gpu(double* A, double* y, double* theta_out, 
                   int n, int N, int L, int J, int Ka, 
                   double P_hat, int T_max, double tol) {
    
    // Initialize cuBLAS
    cublasHandle_t handle;
    cublasCreate(&handle);
    
    // GPU memory allocation
    double *d_A, *d_y, *d_theta, *d_z, *d_theta_next, *d_z_next, *d_v;
    double *d_temp_vec, *d_p_k, *d_f_prime;
    
    cudaMalloc(&d_A, n * N * sizeof(double));
    cudaMalloc(&d_y, n * sizeof(double));
    cudaMalloc(&d_theta, N * sizeof(double));
    cudaMalloc(&d_z, n * sizeof(double));
    cudaMalloc(&d_theta_next, N * sizeof(double));
    cudaMalloc(&d_z_next, n * sizeof(double));
    cudaMalloc(&d_v, N * sizeof(double));
    cudaMalloc(&d_temp_vec, n * sizeof(double));
    cudaMalloc(&d_p_k, (Ka + 1) * sizeof(double));
    cudaMalloc(&d_f_prime, N * sizeof(double));
    
    // Copy input data to GPU
    // cuBLAS expects column-major; our host A is row-major. Convert.
    double* A_col_major = (double*)malloc(n * N * sizeof(double));
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < N; j++) {
            A_col_major[j * n + i] = A[i * N + j];
        }
    }
    cudaMemcpy(d_A, A_col_major, n * N * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, y, n * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemset(d_theta, 0, N * sizeof(double)); // Initialize theta to zeros
    cudaMemcpy(d_z, y, n * sizeof(double), cudaMemcpyHostToDevice); // Initialize z = y
    
    // AMP parameters
    double sqrt_P = sqrt(P_hat / L);
    double alpha = (2.0 * J * L) / n;
    
    // Compute and copy prior probabilities to GPU
    double* h_p_k = (double*)malloc((Ka + 1) * sizeof(double));
    for (int k = 0; k <= Ka; k++) {
        double prob_active = pow(2.0, -J);
        double prob_inactive = 1.0 - prob_active;
        h_p_k[k] = binomial_coeff(Ka, k) * 
                   pow(prob_active, k) * 
                   pow(prob_inactive, Ka - k);
    }
    cudaMemcpy(d_p_k, h_p_k, (Ka + 1) * sizeof(double), cudaMemcpyHostToDevice);
    
    // Host arrays for convergence checking
    double* h_theta = (double*)malloc(N * sizeof(double));
    double* h_theta_next = (double*)malloc(N * sizeof(double));
    
    // Grid and block dimensions
    int blockSize = 256;
    int gridSize_N = (N + blockSize - 1) / blockSize;
    int gridSize_n = (n + blockSize - 1) / blockSize;
    
    // cuBLAS constants
    const double one = 1.0, zero = 0.0, neg_one = -1.0;
    
    // Main AMP iterations
    for (int t = 0; t < T_max; t++) {
        // Compute τ_t = ||z||₂ / √n using cuBLAS
        double tau_squared;
        cublasDdot(handle, n, d_z, 1, d_z, 1, &tau_squared);
        double tau = sqrt(tau_squared / n);
        
        // Compute pseudo-data: v = A^T * z + θ
        // v = A^T * z (using DGEMV: y = alpha*A*x + beta*y)
        cublasDgemv(handle, CUBLAS_OP_T, n, N, &one, d_A, n, d_z, 1, &zero, d_v, 1);
        // v += θ
        cublasDaxpy(handle, N, &one, d_theta, 1, d_v, 1);
        
        // Apply denoiser kernel
        amp_denoiser_kernel<<<gridSize_N, blockSize>>>(d_v, d_theta_next, d_f_prime, 
                                                       d_p_k, Ka, sqrt_P, tau, N);
        cudaDeviceSynchronize();
        
        // Compute mean of f_prime for Onsager term (sum, not absolute sum!)
        double mean_f_prime;
        double sum_f_prime = 0.0;
        double* h_f_prime = (double*)malloc(N * sizeof(double));
        cudaMemcpy(h_f_prime, d_f_prime, N * sizeof(double), cudaMemcpyDeviceToHost);
        for (int i = 0; i < N; i++) {
            sum_f_prime += h_f_prime[i];
        }
        mean_f_prime = sum_f_prime / N;
        free(h_f_prime);
        
        // Compute residual: z_{t+1} = y - A * θ_{t+1} + α * z_t * mean(f'(v))
        // temp_vec = A * θ_{t+1}
        cublasDgemv(handle, CUBLAS_OP_N, n, N, &one, d_A, n, d_theta_next, 1, &zero, d_temp_vec, 1);
        // z_next = y - temp_vec
        cudaMemcpy(d_z_next, d_y, n * sizeof(double), cudaMemcpyDeviceToDevice);
        cublasDaxpy(handle, n, &neg_one, d_temp_vec, 1, d_z_next, 1);
        // z_next += α * z * mean(f'(v))
        double onsager_coeff = alpha * mean_f_prime;
        cublasDaxpy(handle, n, &onsager_coeff, d_z, 1, d_z_next, 1);
        
        // Check convergence: ||θ_{t+1} - θ_t||₂ / √N
        cudaMemcpy(h_theta, d_theta, N * sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_theta_next, d_theta_next, N * sizeof(double), cudaMemcpyDeviceToHost);
        
        double diff_norm = 0.0;
        for (int i = 0; i < N; i++) {
            double diff = h_theta_next[i] - h_theta[i];
            diff_norm += diff * diff;
        }
        diff_norm = sqrt(diff_norm) / sqrt(N);
        
        if (diff_norm < tol) {
            // Converged - copy final result
            cudaMemcpy(theta_out, d_theta_next, N * sizeof(double), cudaMemcpyDeviceToHost);
            
            // Cleanup
            cudaFree(d_A); cudaFree(d_y); cudaFree(d_theta); cudaFree(d_z);
            cudaFree(d_theta_next); cudaFree(d_z_next); cudaFree(d_v);
            cudaFree(d_temp_vec); cudaFree(d_p_k); cudaFree(d_f_prime);
            free(A_col_major);
            free(h_p_k); free(h_theta); free(h_theta_next);
            cublasDestroy(handle);
            return t + 1;
        }
        
        // Update for next iteration: θ = θ_{t+1}, z = z_{t+1}
        cudaMemcpy(d_theta, d_theta_next, N * sizeof(double), cudaMemcpyDeviceToDevice);
        cudaMemcpy(d_z, d_z_next, n * sizeof(double), cudaMemcpyDeviceToDevice);
    }
    
    // Did not converge - return final estimate
    cudaMemcpy(theta_out, d_theta_next, N * sizeof(double), cudaMemcpyDeviceToHost);
    
    // Cleanup
    cudaFree(d_A); cudaFree(d_y); cudaFree(d_theta); cudaFree(d_z);
    cudaFree(d_theta_next); cudaFree(d_z_next); cudaFree(d_v);
    cudaFree(d_temp_vec); cudaFree(d_p_k); cudaFree(d_f_prime);
    free(A_col_major);
    free(h_p_k); free(h_theta); free(h_theta_next);
    cublasDestroy(handle);
    return T_max;
}

// Original CPU version (renamed for clarity)
int amp_decode(double* A, double* y, double* theta_out, 
               int n, int N, int L, int J, int Ka, 
               double P_hat, int T_max, double tol) {
    
    // Allocate working arrays
    double* theta = (double*)calloc(N, sizeof(double));      // θ (initialized to zeros)
    double* z = (double*)malloc(n * sizeof(double));         // residual
    double* theta_next = (double*)malloc(N * sizeof(double));
    double* z_next = (double*)malloc(n * sizeof(double));
    double* v = (double*)malloc(N * sizeof(double));         // pseudo-data
    double* temp_vec = (double*)malloc(n * sizeof(double));  // temporary vector for matvec
    
    // Initialize z = y
    memcpy(z, y, n * sizeof(double));
    
    // AMP parameters
    double sqrt_P = sqrt(P_hat / L);
    double alpha = (2.0 * J * L) / n;
    
    // Pre-compute prior probabilities p_k for k = 0, 1, ..., Ka
    double* p_k = (double*)malloc((Ka + 1) * sizeof(double));
    for (int k = 0; k <= Ka; k++) {
        double prob_active = pow(2.0, -J);                     // 2^(-J)
        double prob_inactive = 1.0 - prob_active;              // 1 - 2^(-J)
        p_k[k] = binomial_coeff(Ka, k) * 
                 pow(prob_active, k) * 
                 pow(prob_inactive, Ka - k);
    }
    
    // Main AMP iterations
    for (int t = 0; t < T_max; t++) {
        // Compute τ_t = ||z||₂ / √n
        double tau = vector_norm(z, n) / sqrt(n);
        
        // Compute pseudo-data: v = A^T * z + θ
        matvec_multiply_transpose(A, z, v, n, N);
        for (int i = 0; i < N; i++) {
            v[i] += theta[i];
        }
        
        // Denoiser: θ_{t+1} = f_t(v)
        double mean_f_prime = 0.0;
        for (int i = 0; i < N; i++) {
            double numerator = 0.0;
            double denominator = 0.0;
            double f_prime_sum = 0.0;
            
            // Compute weighted sum over k = 0, 1, ..., Ka
            for (int k = 0; k <= Ka; k++) {
                double mu_k = k * sqrt_P;
                double diff = v[i] - mu_k;
                double exp_arg = -diff * diff / (2.0 * tau * tau);
                
                // Clip exponential argument for numerical stability
                if (exp_arg < -50.0) exp_arg = -50.0;
                
                double exp_val = exp(exp_arg);
                double weight = p_k[k] * exp_val;
                
                numerator += k * weight;
                denominator += weight;
                
                // Compute derivative term for Onsager correction
                double deriv_term = -(diff / (tau * tau)) * weight;
                f_prime_sum += deriv_term;
            }
            
            // Set θ_{t+1}[i]
            if (denominator > 1e-15) {
                theta_next[i] = sqrt_P * numerator / denominator;
                
                // Compute f'(v[i]) for Onsager term
                double f_prime_i = sqrt_P * f_prime_sum / denominator;
                mean_f_prime += f_prime_i;
            } else {
                theta_next[i] = 0.0;
            }
        }
        mean_f_prime /= N;
        
        // Compute residual: z_{t+1} = y - A * θ_{t+1} + α * z_t * mean(f'(v))
        matvec_multiply(A, theta_next, temp_vec, n, N);
        for (int i = 0; i < n; i++) {
            z_next[i] = y[i] - temp_vec[i] + alpha * z[i] * mean_f_prime;
        }
        
        // Check convergence
        double diff_norm = 0.0;
        for (int i = 0; i < N; i++) {
            double diff = theta_next[i] - theta[i];
            diff_norm += diff * diff;
        }
        diff_norm = sqrt(diff_norm) / sqrt(N);
        
        if (diff_norm < tol) {
            // Converged - copy final result
            memcpy(theta_out, theta_next, N * sizeof(double));
            
            // Cleanup
            free(theta); free(z); free(theta_next); free(z_next);
            free(v); free(temp_vec); free(p_k);
            return t + 1;  // Return number of iterations
        }
        
        // Update for next iteration
        memcpy(theta, theta_next, N * sizeof(double));
        memcpy(z, z_next, n * sizeof(double));
    }
    
    // Did not converge - return final estimate anyway
    memcpy(theta_out, theta_next, N * sizeof(double));
    
    // Cleanup
    free(theta); free(z); free(theta_next); free(z_next);
    free(v); free(temp_vec); free(p_k);
    return T_max;  // Return max iterations (no convergence)
}

// Test function to verify the implementation
void test_amp_decoder() {
    printf("=== Testing AMP Decoder Implementation ===\n");
    
    // Test parameters (matching your Python example)
    int Ka = 3, L = 4, J = 6, B = 24, n = 128;
    int N = L * (1 << J);  // L * 2^J = 4 * 64 = 256
    double P_hat = 1.0;
    double snr_db = 10.0;
    int T_max = 15;
    double tol = 1e-6;
    
    printf("Parameters: Ka=%d, L=%d, J=%d, n=%d, N=%d, P_hat=%.1f, SNR=%.1f dB\n", 
           Ka, L, J, n, N, P_hat, snr_db);
    
    // Allocate arrays
    double* A = (double*)malloc(n * N * sizeof(double));
    double* theta_true = (double*)calloc(N, sizeof(double));
    double* signal = (double*)malloc(n * sizeof(double));
    double* y = (double*)malloc(n * sizeof(double));
    double* theta_decoded_cpu = (double*)malloc(N * sizeof(double));
    double* theta_decoded_gpu = (double*)malloc(N * sizeof(double));
    
    // Create random sensing matrix A ~ N(0, 1/n)
    srand(42);  // Fixed seed for reproducible results
    for (int i = 0; i < n * N; i++) {
        // Box-Muller transform for Gaussian random numbers
        static int has_spare = 0;
        static double spare;
        
        if (has_spare) {
            has_spare = 0;
            A[i] = spare / sqrt(n);
        } else {
            has_spare = 1;
            double u = ((double)rand() / RAND_MAX) * 2.0 - 1.0;
            double v = ((double)rand() / RAND_MAX) * 2.0 - 1.0;
            double s = u * u + v * v;
            while (s >= 1.0 || s == 0.0) {
                u = ((double)rand() / RAND_MAX) * 2.0 - 1.0;
                v = ((double)rand() / RAND_MAX) * 2.0 - 1.0;
                s = u * u + v * v;
            }
            s = sqrt(-2.0 * log(s) / s);
            A[i] = u * s / sqrt(n);
            spare = v * s;
        }
    }
    
    // Create sparse test vector: place Ka=3 active users in different sections
    double sqrt_P_per_section = sqrt(P_hat / L);
    int active_positions[] = {10, 70, 150, 200};  // One per section roughly
    
    printf("Creating sparse vector with active positions: ");
    for (int k = 0; k < Ka; k++) {
        if (active_positions[k] < N) {
            theta_true[active_positions[k]] = sqrt_P_per_section;
            printf("%d ", active_positions[k]);
        }
    }
    printf("\n");
    
    // Generate noiseless signal: signal = A * theta_true
    matvec_multiply(A, theta_true, signal, n, N);
    
    // Add noise to achieve target SNR
    double signal_power = 0.0;
    for (int i = 0; i < n; i++) {
        signal_power += signal[i] * signal[i];
    }
    signal_power /= n;
    
    double snr_linear = pow(10.0, snr_db / 10.0);
    double noise_var = signal_power / snr_linear;
    double noise_std = sqrt(noise_var);
    
    printf("Signal power: %.6f, Noise variance: %.6f, Noise std: %.6f\n", 
           signal_power, noise_var, noise_std);
    
    // Generate noisy signal
    for (int i = 0; i < n; i++) {
        // Simple noise generation (not perfect Gaussian, but good enough for testing)
        double noise = noise_std * (((double)rand() / RAND_MAX) - 0.5) * 2.0 * 1.732;  // roughly Gaussian
        y[i] = signal[i] + noise;
    }
    
    printf("\nRunning CPU decoder...\n");
    int iter_cpu = amp_decode(A, y, theta_decoded_cpu, n, N, L, J, Ka, P_hat, T_max, tol);
    printf("CPU decoder converged in %d iterations\n", iter_cpu);
    
    printf("\nRunning GPU decoder...\n");
    int iter_gpu = amp_decode_gpu(A, y, theta_decoded_gpu, n, N, L, J, Ka, P_hat, T_max, tol);
    printf("GPU decoder converged in %d iterations\n", iter_gpu);
    
    // Compare results
    printf("\n=== Results Comparison ===\n");
    printf("Position | True Value | CPU Decoded | GPU Decoded | CPU Error | GPU Error\n");
    printf("---------|------------|-------------|-------------|-----------|----------\n");
    
    double cpu_mse = 0.0, gpu_mse = 0.0;
    double cpu_max_error = 0.0, gpu_max_error = 0.0;
    
    // Show active positions and a few others
    int positions_to_check[] = {10, 70, 150, 200, 5, 65, 145, 195, 0, 1, 2, 3};
    int num_check = sizeof(positions_to_check) / sizeof(positions_to_check[0]);
    
    for (int i = 0; i < num_check && positions_to_check[i] < N; i++) {
        int pos = positions_to_check[i];
        double true_val = theta_true[pos];
        double cpu_val = theta_decoded_cpu[pos];
        double gpu_val = theta_decoded_gpu[pos];
        double cpu_err = fabs(cpu_val - true_val);
        double gpu_err = fabs(gpu_val - true_val);
        
        printf("%8d | %10.6f | %11.6f | %11.6f | %9.6f | %9.6f\n",
               pos, true_val, cpu_val, gpu_val, cpu_err, gpu_err);
        
        if (cpu_err > cpu_max_error) cpu_max_error = cpu_err;
        if (gpu_err > gpu_max_error) gpu_max_error = gpu_err;
    }
    
    // Compute MSE over all positions
    for (int i = 0; i < N; i++) {
        double cpu_err = theta_decoded_cpu[i] - theta_true[i];
        double gpu_err = theta_decoded_gpu[i] - theta_true[i];
        cpu_mse += cpu_err * cpu_err;
        gpu_mse += gpu_err * gpu_err;
    }
    cpu_mse /= N;
    gpu_mse /= N;
    
    printf("\n=== Summary ===\n");
    printf("CPU MSE: %.8f, Max Error: %.6f\n", cpu_mse, cpu_max_error);
    printf("GPU MSE: %.8f, Max Error: %.6f\n", gpu_mse, gpu_max_error);
    printf("Relative GPU vs CPU MSE: %.6f\n", gpu_mse / (cpu_mse + 1e-15));
    
    // Check if implementations agree
    double implementation_diff = 0.0;
    for (int i = 0; i < N; i++) {
        double diff = theta_decoded_cpu[i] - theta_decoded_gpu[i];
        implementation_diff += diff * diff;
    }
    implementation_diff = sqrt(implementation_diff);
    printf("CPU vs GPU difference (L2 norm): %.8f\n", implementation_diff);
    
    if (implementation_diff < 1e-10) {
        printf("✓ CPU and GPU implementations agree!\n");
    } else if (implementation_diff < 1e-6) {
        printf("~ CPU and GPU implementations mostly agree (small numerical differences)\n");
    } else {
        printf("✗ CPU and GPU implementations differ significantly!\n");
    }
    
    // Cleanup
    free(A); free(theta_true); free(signal); free(y);
    free(theta_decoded_cpu); free(theta_decoded_gpu);
    
    printf("\n=== Test Complete ===\n");
}

// Main function
int main() {
    printf("AMP Decoder Test Program\n");
    printf("========================\n\n");
    
    // Check CUDA availability
    int deviceCount;
    cudaError_t error = cudaGetDeviceCount(&deviceCount);
    
    if (error != cudaSuccess || deviceCount == 0) {
        printf("No CUDA devices found. Running CPU-only test.\n\n");
        // Could add CPU-only test here
        return 1;
    } else {
        printf("Found %d CUDA device(s). Running full test.\n\n", deviceCount);
    }
    
    test_amp_decoder();
    
    return 0;
}
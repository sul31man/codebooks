##lets quickly protype the AMP algorithm here
import numpy as np
from scipy import sparse
import matplotlib.pyplot as plt

class AMPAlgorithm:
    """
    Approximate Message Passing (AMP) algorithm for sparse signal recovery.
    
    This implementation solves the compressed sensing problem:
    y = Ax + w
    where A is the measurement matrix, x is the sparse signal to recover,
    and w is additive noise.
    """
    
    def __init__(self, A, y, sigma_w=0.1, max_iter=100, tol=1e-6):
        """
        Initialize AMP algorithm.
        
        Args:
            A: Measurement matrix (m x n)
            y: Measurements (m x 1)
            sigma_w: Noise standard deviation
            max_iter: Maximum number of iterations
            tol: Convergence tolerance
        """
        self.A = A
        self.y = y
        self.m, self.n = A.shape
        self.sigma_w = sigma_w
        self.max_iter = max_iter
        self.tol = tol
        
        # Initialize variables
        self.x_hat = np.zeros(self.n)
        self.z = np.copy(y)
        self.tau_x = 1.0
        
        # Store history for analysis
        self.x_history = []
        self.mse_history = []
        
    def soft_threshold(self, x, threshold):
        """Soft thresholding function (denoiser for sparse signals)."""
        return np.sign(x) * np.maximum(np.abs(x) - threshold, 0)
    
    def update_tau_x(self, x_hat_prev, iteration):
        """Update the variance parameter tau_x."""
        # Simple variance estimation
        self.tau_x = np.var(x_hat_prev) + 1e-10  # Add small epsilon for stability
        
    def denoiser(self, r, tau_r):
        """
        Denoising function - here we use soft thresholding for sparse recovery.
        Can be replaced with other denoisers based on signal prior.
        """
        # Threshold scales with noise level
        threshold = self.sigma_w**2 / np.sqrt(tau_r + 1e-10)
        return self.soft_threshold(r, threshold)
    
    def derivative_denoiser(self, r, tau_r):
        """Derivative of the denoiser for AMP bias correction."""
        threshold = self.sigma_w**2 / np.sqrt(tau_r + 1e-10)
        return (np.abs(r) > threshold).astype(float)
    
    def run(self, x_true=None, verbose=False):
        """
        Run the AMP algorithm.
        
        Args:
            x_true: True signal (for MSE calculation)
            verbose: Print progress
            
        Returns:
            x_hat: Recovered signal
            convergence_info: Dictionary with convergence information
        """
        
        for iteration in range(self.max_iter):
            x_hat_prev = np.copy(self.x_hat)
            
            # AMP iteration
            # 1. Compute effective observation
            r = self.x_hat + self.A.T @ self.z / self.m
            
            # 2. Update variance estimate
            self.update_tau_x(x_hat_prev, iteration)
            tau_r = self.tau_x + self.sigma_w**2 / self.m
            
            # 3. Denoising step
            self.x_hat = self.denoiser(r, tau_r)
            
            # 4. Compute bias correction term
            div_term = np.mean(self.derivative_denoiser(r, tau_r))
            
            # 5. Update residual with bias correction
            self.z = self.y - self.A @ self.x_hat + self.z * div_term
            
            # Store history
            self.x_history.append(np.copy(self.x_hat))
            
            if x_true is not None:
                mse = np.mean((self.x_hat - x_true)**2)
                self.mse_history.append(mse)
                
                if verbose and iteration % 10 == 0:
                    print(f"Iteration {iteration}: MSE = {mse:.6f}")
            
            # Check convergence
            if np.linalg.norm(self.x_hat - x_hat_prev) < self.tol:
                if verbose:
                    print(f"Converged at iteration {iteration}")
                break
        
        convergence_info = {
            'iterations': len(self.x_history),
            'converged': iteration < self.max_iter - 1,
            'final_residual': np.linalg.norm(self.y - self.A @ self.x_hat),
            'x_history': self.x_history,
            'mse_history': self.mse_history
        }
        
        return self.x_hat, convergence_info

def generate_test_problem(m=100, n=200, sparsity=0.1, snr_db=20):
    """Generate a test compressed sensing problem."""
    # Generate random measurement matrix
    A = np.random.randn(m, n) / np.sqrt(m)
    
    # Generate sparse signal
    x_true = np.zeros(n)
    support = np.random.choice(n, int(sparsity * n), replace=False)
    x_true[support] = np.random.randn(len(support))
    
    # Generate measurements with noise
    y_clean = A @ x_true
    noise_power = np.var(y_clean) / (10**(snr_db/10))
    noise = np.sqrt(noise_power) * np.random.randn(m)
    y = y_clean + noise
    
    return A, y, x_true, np.sqrt(noise_power)

def demo_amp():
    """Demonstrate AMP algorithm on a test problem."""
    print("=== AMP Algorithm Demo ===")
    
    # Generate test problem
    m, n = 100, 200
    A, y, x_true, sigma_w = generate_test_problem(m, n, sparsity=0.1, snr_db=20)
    
    print(f"Problem size: {m} measurements, {n} unknowns")
    print(f"True sparsity: {np.sum(x_true != 0)} non-zeros")
    print(f"Noise level: σ = {sigma_w:.4f}")
    
    # Run AMP
    amp = AMPAlgorithm(A, y, sigma_w=sigma_w, max_iter=100)
    x_recovered, info = amp.run(x_true=x_true, verbose=True)
    
    # Results
    final_mse = np.mean((x_recovered - x_true)**2)
    recovered_sparsity = np.sum(np.abs(x_recovered) > 0.01)
    
    print(f"\n=== Results ===")
    print(f"Final MSE: {final_mse:.6f}")
    print(f"Recovered sparsity: {recovered_sparsity} non-zeros")
    print(f"Iterations: {info['iterations']}")
    print(f"Converged: {info['converged']}")
    
    # Plot results
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    plt.plot(x_true, 'b-', label='True signal', alpha=0.7)
    plt.plot(x_recovered, 'r--', label='Recovered', alpha=0.7)
    plt.legend()
    plt.title('Signal Recovery')
    plt.xlabel('Index')
    plt.ylabel('Amplitude')
    
    plt.subplot(2, 2, 2)
    plt.scatter(x_true, x_recovered, alpha=0.6)
    plt.plot([-3, 3], [-3, 3], 'r--', alpha=0.5)
    plt.xlabel('True values')
    plt.ylabel('Recovered values')
    plt.title('Recovery Scatter Plot')
    
    plt.subplot(2, 2, 3)
    if info['mse_history']:
        plt.semilogy(info['mse_history'])
        plt.xlabel('Iteration')
        plt.ylabel('MSE')
        plt.title('Convergence')
        plt.grid(True)
    
    plt.subplot(2, 2, 4)
    residual = y - A @ x_recovered
    plt.plot(residual)
    plt.xlabel('Measurement index')
    plt.ylabel('Residual')
    plt.title('Final Residual')
    
    plt.tight_layout()
    plt.show()
    
    return x_recovered, info

if __name__ == "__main__":
    # Run demonstration
    demo_amp()

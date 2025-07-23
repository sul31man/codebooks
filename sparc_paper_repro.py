##this will be used for anlaysing the AMP algorithm
##first we need to generate the sparse vector signal
import numpy as np
from itertools import product
from scipy.special import comb
import matplotlib.pyplot as plt

class SPARC:
    def __init__(self, E_Ka, L, J, B, n, Eb_N0_dB):
        self.E_Ka = E_Ka  # Expected number of users
        self.L = L
        self.J = J
        self.B = B
        self.n = n
        self.Eb_N0_dB = Eb_N0_dB
        self.N = L * (2 ** J)
        self.R = B / n
        self.sigma2 = 1.0  # Fix noise variance to 1
        Eb_N0_linear = 10 ** (self.Eb_N0_dB / 10)
        self.P_total = 2 * self.R * Eb_N0_linear  # Paper formula: P = 2 * R * Eb/N0
        self.sqrt_P_col = np.sqrt(self.P_total / self.L)

    def generate_A(self):
        n, N = self.n, self.N
        A = np.random.randn(n, N)
        A = A - np.mean(A, axis=0)  # mean zero per column
        A = A / np.linalg.norm(A, axis=0)  # unit norm per column
        return A


    def generate_theta(self):
        # Sample Ka ~ Poisson(E[Ka])
        Ka = np.random.poisson(self.E_Ka)
        Ka = max(1, Ka)  # Ensure at least one user
        # For each user, pick one active column per section
        theta = np.zeros(self.N)
        user_support = []
        for _ in range(Ka):
            indices = []
            for l in range(self.L):
                idx = l * (2 ** self.J) + np.random.randint(0, 2 ** self.J)
                indices.append(idx)
                theta[idx] = self.sqrt_P_col
            user_support.append(indices)
        return theta, user_support, Ka

    def generate_signal(self):
        self.A = self.generate_A()
        self.theta_true, self.user_support, self.Ka = self.generate_theta()
        self.signal = self.A @ self.theta_true
        return self.signal

    def transmit_signal(self):
        noise = np.random.randn(self.n) * np.sqrt(self.sigma2)
        self.y = self.signal + noise
        return self.y

    def amp_decode(self, T_max=15, tol=1e-6):
        A, y = self.A, self.y
        n, N = A.shape
        L, J = self.L, self.J
        Ka = self.Ka
        sqrt_P = self.sqrt_P_col
        # Prior for each column: probability of being active = Ka / N
        p0 = Ka / N
        theta = np.zeros(N)
        z = y.copy()
        for t in range(T_max):
            tau = np.linalg.norm(z) / np.sqrt(n)
            v = A.T @ z + theta
            # Denoiser: section-wise softmax (per section)
            theta_next = np.zeros_like(theta)
            for l in range(L):
                start = l * (2 ** J)
                end = start + (2 ** J)
                v_sec = v[start:end]
                # Section-wise softmax
                expv = np.exp((v_sec * sqrt_P) / (tau ** 2))
                probs = expv / np.sum(expv)
                theta_next[start:end] = sqrt_P * probs
            # Onsager term (approximate)
            z_next = y - A @ theta_next + z * (L * (2 ** J) / n) * np.mean(theta_next != 0)
            # Convergence check on both theta and z
            if np.linalg.norm(theta_next - theta) / np.sqrt(N) < tol and np.linalg.norm(z_next - z) / np.sqrt(n) < tol:
                theta = theta_next
                break
            theta, z = theta_next, z_next
        self.theta_hat = theta
        return theta

    def section_wise_argmax(self):
        # For each section, pick the index with the largest value
        L, J = self.L, self.J
        N = self.N
        theta_hat = self.theta_hat
        detected = []
        for l in range(L):
            start = l * (2 ** J)
            end = start + (2 ** J)
            idx = np.argmax(theta_hat[start:end]) + start
            detected.append(idx)
        return detected

    def error_metrics(self):
        # Compute MD/FA as in the paper
        true_support = set(np.where(self.theta_true != 0)[0])
        detected_support = set(self.section_wise_argmax())
        MD = len(true_support - detected_support) / (self.Ka * self.L)
        FA = len(detected_support - true_support) / (self.Ka * self.L)
        return MD, FA

    def simulate(self, num_sims=10):
        MDs, FAs = [], []
        for _ in range(num_sims):
            self.generate_signal()
            self.transmit_signal()
            self.amp_decode()
            MD, FA = self.error_metrics()
            MDs.append(MD)
            FAs.append(FA)
        return np.mean(MDs), np.mean(FAs)

def sweep_ebn0_for_ka50():
    E_Ka = 50
    L = 32
    J = 8
    B = 128
    n = 19600
    ebn0_range = [1.0 + 0.25 * i for i in range(21)]  # 1 to 6 dB
    MDs, FAs = [], []
    for Eb_N0_dB in ebn0_range:
        print(f"[Sweep] Ka={E_Ka}, Eb/N0={Eb_N0_dB:.2f}dB")
        system = SPARC(E_Ka, L, J, B, n, Eb_N0_dB)
        MD, FA = system.simulate(num_sims=10)
        print(f"[Sweep] MD={MD:.3f}, FA={FA:.3f}, MD+FA={MD+FA:.3f}")
        MDs.append(MD)
        FAs.append(FA)
    plt.figure()
    plt.plot(ebn0_range, [m+f for m, f in zip(MDs, FAs)], label='MD+FA')
    plt.plot(ebn0_range, MDs, label='MD')
    plt.plot(ebn0_range, FAs, label='FA')
    plt.xlabel('$E_b/N_0$ (dB)')
    plt.ylabel('Error Probability')
    plt.title('SPARC: MD/FA vs $E_b/N_0$ (Ka=50)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('sparc_md_fa_vs_ebn0_ka50.png')
    plt.show()
    print("[Sweep] Plot saved as sparc_md_fa_vs_ebn0_ka50.png")

if __name__ == "__main__":
    sweep_ebn0_for_ka50() 
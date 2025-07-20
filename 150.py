import math, numpy as np, matplotlib.pyplot as plt
from math import lgamma, log
from itertools import combinations
from scipy.stats import chi2
import scipy.special as sp

# ------------------------------------------------------------
#  helpers: log-factorial / log-binomial (stable)
# ------------------------------------------------------------
def ln_fact(t):          # ln(t!)
    return lgamma(t + 1)

def ln_comb(n, k):       # ln C(n,k)
    return ln_fact(n) - ln_fact(k) - ln_fact(n - k)

# ------------------------------------------------------------
#  p0  – eq. (4)
# ------------------------------------------------------------
def p0(Ka, M, n, P_prime, P):
    p0 = math.comb(Ka, 2) / M + Ka * sp.gammaincc(n, n * P / P_prime)
    return float(p0)

# ------------------------------------------------------------
#  rates R1 , R2 – eqs. (9) & (10)
# ------------------------------------------------------------
def R1(t, M, n):
    R1_value = 1 / n * np.log(float(M)) - 1 / (n * t) * sp.gammaln(t + 1)
    return R1_value

def R2(t, Ka, n):
    R2_value = 1 / n * (sp.gammaln(Ka + 1) - sp.gammaln(t + 1) - sp.gammaln(Ka - t + 1))
    return R2_value

# ------------------------------------------------------------
#  E₀(ρ₁) – eq. (11)
# ------------------------------------------------------------
def E0(rho1, a, b):

    E0 = rho1 * a + np.log(1 - b * rho1)
    return E0

# ------------------------------------------------------------
#  E(t)  – maximise (6)–(8)  on ρ,ρ₁ ∈ [0,1]
# ------------------------------------------------------------
def E_t(P_prime, t, Ka, n, M):
    # For a specific t value, we only need 2D grids for rho and rho1
    rho_vec  = np.linspace(0, 1, 100)
    rho1_vec = np.linspace(0, 1, 100)
    rho, rho1 = np.meshgrid(rho_vec, rho1_vec)
    
    D = (P_prime * t - 1) ** 2 + 4 * P_prime * t * (1 + rho * rho1) / (1 + rho)
    lambda_ = (P_prime * t - 1 + np.sqrt(D)) / (2 * (1 + rho1 * rho) * P_prime * t)
    # compute mu
    mu = rho * lambda_ / (1 + P_prime * t * lambda_)
    a = rho * np.log(1 + P_prime * t * lambda_) + np.log(1 + P_prime * t * mu)
    b = rho * lambda_ - mu / (1 + P_prime * t * mu)
    R1_v = R1(t, M, n)
    R2_v = R2(t, Ka, n)
    E0_v = rho1 * a + np.log(1 - b * rho1)
    Et = np.max(-rho * rho1 * t * R1_v - rho1 * R2_v + E0_v)
    
    return float(Et)

def p_t(P_prime, t, Ka, n, M):
    return float(np.exp(-n * E_t(P_prime, t, Ka, n, M)))

# ------------------------------------------------------------
#  codebook & signal utilities
# ------------------------------------------------------------
def generate_codebook(Ka, n, P_prime):
    """
    Generates a random Gaussian codebook as specified in the paper.
    
    Args:
        M: Number of codewords (2^k)
        n: Blocklength (real degrees of freedom)
        P_prime: Variance for generating codewords (P' < P)
    
    Returns:
        C: Codebook matrix of shape (M, n) with i.i.d. N(0, P') entries
    """
    # Generate i.i.d. Gaussian codewords with variance P'
    C = np.random.normal(0, np.sqrt(P_prime), size=(Ka, n))
    
    return C

def generate_signal(codebook, Ka):
    idx = np.random.choice(codebook.shape[0], Ka, replace=False)  # no duplicates
    return codebook[idx].sum(axis=0)

def add_noise(signal):                  # unit-variance AWGN
    return signal + np.random.randn(signal.shape[0])

# ------------------------------------------------------------
#  I_t
# ------------------------------------------------------------
def I_t(Ka, n, P_prime):

        Zi       = np.sqrt(0.5) * (np.random.randn(n) + 1j * np.random.randn(n))
        codebook = np.sqrt(0.5 * P_prime) * (np.random.randn(Ka, n) + 1j * np.random.randn(Ka, n))
        it       = n * np.log(1 + P_prime) + \
                  (np.sum(np.abs(np.tile(Zi, (Ka, 1)) + codebook) ** 2, axis=1) / (1 + P_prime) - \
                   np.sum(np.abs(np.tile(Zi, (Ka, 1))) ** 2, axis=1))
        min_it   = np.min(it)

        return min_it

# ------------------------------------------------------------
#  q_t  – eq. (5)
# ------------------------------------------------------------
def q_t(t, Ka, n, P_prime, M, N_mc=5000):
    I_samples = [I_t(Ka, n, P_prime) for _ in range(N_mc)]
    R1v, R2v  = R1(t, M, n), R2(t, Ka, n)
    
    
    It = np.array(I_samples)
    gamma = np.sort(It)
    prob = np.arange(1, len(gamma) + 1) / len(gamma)
    
    qt = np.min(prob + np.exp(n * (R1v + R2v) - gamma))

    return float(qt)

# ------------------------------------------------------------
#  ε  upper-bound
# ------------------------------------------------------------
def epsilon_bound(Ka, M, n, P_prime, P):
    term0  = p0(Ka, M, n, P_prime, P)
    qt = q_t(1, Ka, n, P_prime, M)

    if Ka > 150:

    
        term1 = min(qt, p_t(P_prime, 1, Ka, n, M))

        series = sum((t / Ka) * p_t(P_prime, t, Ka, n, M)
                 for t in range(2, Ka + 1))
        
        return min(1.0, series + term0 + term1)
    else :
      
        series = sum((t / Ka) * min(p_t(P_prime, t, Ka, n, M), qt) 
                 for t in range(1, Ka + 1))
    
        return min(1.0, series + term0 )

# ------------------------------------------------------------
#  convert P′ → Eb/N0   (eq. (1))
# ------------------------------------------------------------
def EbN0_dB(P_prime, M, n):
    R       = math.log2(M) / n
    EbN0    = P_prime / (R)
    return 10 * np.log10(EbN0)

# ------------------------------------------------------------
#  search P′ giving ε ≤ ε_target
# ------------------------------------------------------------
def find_Pprime(Ka, M, n, P,starting_point, eps_target=0.1050, num_points=20):
    
    
    for P_prime in np.linspace(starting_point, P, num_points):  # start from 0.1 to avoid very small values
        
        eps = epsilon_bound(Ka, M, n, P_prime, P)
        decibels = EbN0_dB(P_prime, M, n)
        print(f"Testing P'={P_prime:.4f}, ε={eps:.4f}, P_db = {decibels}")  # Print progress
        
        
        if eps <= eps_target:
            print(f"\nFound solution: P'={P_prime:.4f} gives ε={eps:.4f}")
            return P_prime
            
    print(f"\nWarning: No solution found below P'={P}. Last ε={eps:.4f}")

    
    return P

# ------------------------------------------------------------
#  main sweep
# ------------------------------------------------------------
if __name__ == "__main__":
    n        = 30000           # codeword length  (was 30 000)
    k = 100
    M        = 2 ** k       # messages per user (was 2**100)
    Ka_grid = range(154, 375, 25 )  # run Ka = 2…10    (was up to 100)
    P = 10**(2/10)*k/n
    
    MC_qt    = 1000          # samples in q_t   (was 5 000)
    grid_Et  = 21           # ρ,ρ1 grid steps  (was 61)

    Ka_list, Eb_list = [], []
    Pp = 0.0038
    for Ka in Ka_grid:
      
      print(Ka)
      Pp    = find_Pprime(Ka, M, n, P,eps_target=0.1050, starting_point = Pp)   # removed extra arguments
      Eb_dB = EbN0_dB(Pp, M, n)
      Ka_list.append(Ka);  Eb_list.append(Eb_dB)
      print(f"Ka={Ka:2d}   P′={Pp:.4f}   Eb/N0={Eb_dB:+.2f} dB")

    plt.plot(Ka_list, Eb_list, "bo-")
    plt.xlabel("$K_a$"), plt.ylabel("$E_b/N_0$  [dB]"), plt.grid(True)
    plt.tight_layout(); plt.show()

    

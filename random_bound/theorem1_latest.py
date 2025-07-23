import math, numpy as np, matplotlib.pyplot as plt
from math import lgamma, log
from itertools import combinations
from scipy.stats import chi2

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
def p0(Ka, M, n, P_prime):
    coll = math.comb(Ka, 2) / M                     # collision term
    P_tot = (Ka - 1) * P_prime                      # interference power
    tail  = chi2.sf(n * P_tot / P_prime, df=n)      # χ² tail
    return coll + Ka * tail

# ------------------------------------------------------------
#  rates R1 , R2 – eqs. (9) & (10)
# ------------------------------------------------------------
def R1(t, M, n):
    return (log(M) - ln_fact(t)) / n                # *** no  “/t” ***

def R2(t, Ka, n):
    return ln_comb(Ka, t) / n

# ------------------------------------------------------------
#  E₀(ρ₁) – eq. (11)
# ------------------------------------------------------------
def E0(rho1, a, b):
    inner = 1 - 2 * b * rho1
    return -np.inf if inner <= 0 else rho1 * a + 0.5 * log(inner)

# ------------------------------------------------------------
#  E(t)  – maximise (6)–(8)  on ρ,ρ₁ ∈ [0,1]
# ------------------------------------------------------------
def E_t(P_prime, t, Ka, n, M, grid=61):
    best = -np.inf
    rhos, rho1s = np.linspace(0, 1, grid), np.linspace(0, 1, grid)

    for rho in rhos:
        for rho1 in rho1s:
            D = (P_prime * t - 1) ** 2 + 4 * P_prime * t * (1 + rho * rho1) / (1 + rho)
            if D < 0: continue
            lam = (P_prime * t - 1 + math.sqrt(D)) / (2 * (1 + rho * rho1) * P_prime * t)
            if lam <= 0: continue
            mu  = rho * lam / (1 + 2 * P_prime * t * lam)
            if mu <= 0: continue

            a  = (rho ) * log(1 + P_prime * t * lam) + 0.5 * log(1 + P_prime * t * mu)
            b  = rho * lam - mu / (1 + P_prime * t * mu)
            e0 = E0(rho1, a, b)
            if not math.isfinite(e0): continue

            val = -rho * rho1 * t * R1(t, M, n) - rho1 * R2(t, Ka, n) + e0
            best = max(best, val)

    return max(0.0, best)

def p_t(P_prime, t, Ka, n, M):
    return math.exp(-n * E_t(P_prime, t, Ka, n, M))

# ------------------------------------------------------------
#  codebook & signal utilities
# ------------------------------------------------------------
def generate_codebook(Ka, n, P_prime):
    C = np.random.randn(Ka, n) ##we should have M codewords, both with different variances. one has variance P and the other with variance (M-Ka)*P_prime. 
    C /= np.linalg.norm(C, axis=1, keepdims=True)
    C *= math.sqrt(n * P_prime)
    return C

def generate_signal(codebook, Ka):
    idx = np.random.choice(codebook.shape[0], Ka, replace=False)  # no duplicates
    return codebook[idx].sum(axis=0)

def add_noise(signal):                  # unit-variance AWGN
    return signal + np.random.randn(signal.shape[0])

# ------------------------------------------------------------
#  I_t
# ------------------------------------------------------------
def I_t(t, Ka, codebook, n, P_prime):
    Y      = add_noise(generate_signal(codebook, Ka)) ## put the codebook function in here !
    C_t    = 0.5 * log(1 + P_prime * t)
    min_it = float('inf')                 ##new codebook must be generated everytime i look into I_t

    for S0 in combinations(range(Ka), t):
        S0      = list(S0)
        S0_c    = [i for i in range(Ka) if i not in S0] ## make sure to add on the rest of the codewords 
        a       = codebook[S0].sum(axis=0)
        b       = codebook[S0_c].sum(axis=0)
        term1   = np.linalg.norm(Y - b)**2 / (1 + P_prime * t)
        term2   = np.linalg.norm(Y - a - b)**2 / (1 + P_prime * t)
        info    = n * C_t + (log(np.e) / 2) * (term1 - term2)
        min_it  = min(min_it, info)

    return min_it

# ------------------------------------------------------------
#  q_t  – eq. (5)
# ------------------------------------------------------------ ##only call this once in the sum, for t =1 . 
def q_t(t, Ka, codebook, n, P_prime, M, N_mc=5000):
    I_samples = [I_t(t, Ka, codebook, n, P_prime) for _ in range(N_mc)]
    R1v, R2v  = R1(t, M, n), R2(t, Ka, n)
    a_star    = n * (t * R1v + R2v)                # pivot point

    g_min = min(I_samples) - 5
    g_max = max(a_star + 20, max(I_samples) + 5)
    gammas = np.linspace(g_min, g_max, 400)

    best = float('inf')
    I_arr = np.array(I_samples)
    for g in gammas:
        prob   = np.mean(I_arr <= g)
        union  = math.exp(a_star - g)
        best   = min(best, prob + union)
    return best

# ------------------------------------------------------------
#  ε  upper-bound
# ------------------------------------------------------------
def epsilon_bound(Ka, M, n, P_prime, codebook):
    term0  = p0(Ka, M, n, P_prime)
    series = sum((t / Ka) * min(p_t(P_prime, t, Ka, n, M),
                                q_t(t, Ka, codebook, n, P_prime, M))
                 for t in range(1, Ka + 1))
    return min(1.0, term0 + series)

# ------------------------------------------------------------
#  convert P′ → Eb/N0   (eq. (1))
# ------------------------------------------------------------
def EbN0_dB(P_prime, M, n):
    R       = log(M) / n
    EbN0    = P_prime / (2 * R)
    return 10 * math.log10(EbN0)

# ------------------------------------------------------------
#  search P′ giving ε ≤ ε_target
# ------------------------------------------------------------
def find_Pprime(Ka, M, n, eps_target=0.1):
    codebook = generate_codebook(Ka, n, 1.0)       # power will be rescaled later
    lo, hi = 1e-4, 50.0
    for _ in range(50):
        mid = 0.5 * (lo + hi)
        eps = epsilon_bound(Ka, M, n, mid, codebook * math.sqrt(mid))
        if eps <= eps_target:
            hi = mid
        else:
            lo = mid
    return hi

# ------------------------------------------------------------
#  main sweep
# ------------------------------------------------------------
if __name__ == "__main__":
    n        = 300           # codeword length  (was 30 000)
    M        = 2 ** 12       # messages per user (was 2**100)
    Ka_grid  = range(2, 11)  # run Ka = 2…10    (was up to 100)

    MC_qt    = 400           # samples in q_t   (was 5 000)
    grid_Et  = 21            # ρ,ρ1 grid steps  (was 61)

    Ka_list, Eb_list = [], []
    for Ka in Ka_grid:
        Pp    = find_Pprime(Ka, M, n, eps_target=0.15)   # removed extra arguments
        Eb_dB = EbN0_dB(Pp, M, n)
        Ka_list.append(Ka);  Eb_list.append(Eb_dB)
        print(f"Ka={Ka:2d}   P′={Pp:.4f}   Eb/N0={Eb_dB:+.2f} dB")

    plt.plot(Ka_list, Eb_list, "bo-")
    plt.xlabel("$K_a$"), plt.ylabel("$E_b/N_0$  [dB]"), plt.grid(True)
    plt.tight_layout(); plt.show()

    
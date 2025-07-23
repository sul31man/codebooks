import math, numpy as np, matplotlib.pyplot as plt
from math import lgamma, log
from scipy.stats import chi2

# -------------------------------------------------------------------------
# log-factorial & log-binomial
# -------------------------------------------------------------------------
ln_fact = lambda t: lgamma(t + 1)
ln_comb = lambda n, k: ln_fact(n) - ln_fact(k) - ln_fact(n - k)

# -------------------------------------------------------------------------
# p₀  – collisions  +  χ² tail  (eq. 4)
# -------------------------------------------------------------------------
def p0(Ka, M, n, Pprime, Ppeak):
    return math.comb(Ka, 2) / M + Ka * chi2.sf(n * Ppeak / Pprime, df=n)

# -------------------------------------------------------------------------
# R₁ , R₂  (eqs. 9–10)   – natural logs!
# -------------------------------------------------------------------------
def R1(t, M, n):
    return (log(M) - ln_fact(t) / t) / n     # ln(M) – ln(t!)/t

def R2(t, Ka, n):
    return ln_comb(Ka, t) / n

# -------------------------------------------------------------------------
# E₀(ρ₁)  (eq. 11)
# -------------------------------------------------------------------------
def E0(rho1, a, b):
    inner = 1 - 2 * b * rho1
    return -np.inf if inner <= 0 else rho1 * a + 0.5 * log(inner)

# -------------------------------------------------------------------------
# E(t)  – max over ρ,ρ₁   (eqs. 6–8)
# -------------------------------------------------------------------------
def E_t(Pprime, t, Ka, n, M, grid=41):
    P = Pprime
    best = -np.inf
    for rho in np.linspace(0, 1, grid):
        for rho1 in np.linspace(0, 1, grid):
            D = (P*t - 1)**2 + 4*P*t*(1 + rho*rho1)/(1 + rho)
            if D < 0: continue
            lam = (P*t - 1 + math.sqrt(D)) / (4*(1 + rho*rho1)*P*t)
            if lam <= 0: continue
            mu  = rho * lam / (1 + 2*P*t*lam)
            if mu <= 0: continue

            term1, term2 = 1 + 2*P*t*lam, 1 + 2*P*t*mu
            if term1 <= 0 or term2 <= 0: continue

            a = (rho/2)*log(term1) + 0.5*log(term2)
            b = rho*lam - mu/(1 + 2*P*t*mu)
            val = -rho*rho1*t*R1(t, M, n) - rho1*R2(t, Ka, n) + E0(rho1, a, b)
            best = max(best, val)
    return max(0.0, best)

# -------------------------------------------------------------------------
# Gallager branch  pₜ  (eq. 5)
# -------------------------------------------------------------------------
def p_t(Pprime, t, Ka, n, M):
    return math.exp(-n * E_t(Pprime, t, Ka, n, M))

# -------------------------------------------------------------------------
# Information-density branch  qₜ  (Chernoff surrogate)
# -------------------------------------------------------------------------
def q_t(Pprime, t, Ka, n, M):
    mu  = 0.5 * log(1 + Pprime*t)
    var = (Pprime*t) / (1 + Pprime*t)
    gamma = mu - var                     # ~1 σ below the mean
    tail  = math.exp(-var/2)             # exp(-(mu-gamma)^2/(2var)) = exp(-var/2)

    # log-safe computation of exp{ n(tR1+R2) – γ }
    expo_log = n*(t*R1(t, M, n) + R2(t, Ka, n)) - gamma
    exp_term = math.exp(expo_log) if expo_log < 700 else float('inf')
    return tail + exp_term

# -------------------------------------------------------------------------
# ε upper bound  (Theorem 1)
# -------------------------------------------------------------------------
def epsilon_bound(Ka, M, n, Pprime, Ppeak):
    series = sum((t/Ka) * min(p_t(Pprime, t, Ka, n, M),   # ← new
                          q_t(Pprime, t, Ka, n, M))
             for t in range(1, Ka+1))
    return min(1.0, p0(Ka, M, n, Pprime, Ppeak) + series)

# -------------------------------------------------------------------------
# P'  →  E_b/N_0  (eq. 1)
# -------------------------------------------------------------------------
def EbN0_dB(Pprime, M, n):
    R = log(M) / n
    return 10 * math.log10(Pprime / (2*R))

# -------------------------------------------------------------------------
# find smallest P′ s.t. ε ≤ ε_target
# -------------------------------------------------------------------------
# --- choose a *fixed* peak power once for all searches -------------
P_peak_abs = 1.1       # <-- pick 1.1 (linear), or 2, 10, etc.

def find_Pprime(Ka, M, n, eps_target=0.1):
    lo, hi = 1e-10, 1.0
    while epsilon_bound(Ka, M, n, hi, P_peak_abs) > eps_target:
        hi *= 2
    for _ in range(60):
        mid = 0.5*(lo+hi)
        if epsilon_bound(Ka, M, n, mid, P_peak_abs) <= eps_target:
            hi = mid
        else:
            lo = mid
    return hi

# -------------------------------------------------------------------------
# reproduce the red-dotted curve
# -------------------------------------------------------------------------
if __name__ == "__main__":
    n, M = 30_000, 2**100
    print("Ka   P'        Eb/N0(dB)")
    print("--   --------  ---------")
    Kas, Eb = [], []
    for Ka in range(2, 101, 10):
        Pp  = find_Pprime(Ka, M, n)
        EbN = EbN0_dB(Pp, M, n)
        Kas.append(Ka);  Eb.append(EbN)
        print(f"{Ka:2d}   {Pp:8.2e}   {EbN:9.2f}")

    plt.plot(Kas, Eb, "r--", lw=2)
    plt.xlabel("$K_a$  (active users)"); plt.ylabel("$E_b/N_0$  [dB]")
    plt.ylim(-2, 10); plt.grid(True); plt.tight_layout(); plt.show()
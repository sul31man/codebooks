import math
import numpy as np
from scipy.stats import chi2
from math import lgamma, log
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# helpers: log-factorials and log-binomials (to avoid overflow)
# ---------------------------------------------------------------------------
def ln_fact(t):
    """natural log of t!  via lgamma"""
    return lgamma(t + 1)

def ln_comb(n, k):
    """natural log of C(n,k)"""
    return ln_fact(n) - ln_fact(k) - ln_fact(n - k)


# ---------------------------------------------------------------------------
# p0  –  eq. (4)
# ---------------------------------------------------------------------------
def p0(Ka, M, n, P_prime, P):
    """collision + Gaussian tail term"""
    coll = math.comb(Ka, 2) / M
    # tail of χ²ₙ distribution:  P[ ΣZ_i² ≥ n·P/P' ]
    tail = chi2.sf(n * P / P_prime, df=n)
    return coll + Ka * tail


# ---------------------------------------------------------------------------
# R1 , R2   –  eqs. (9) & (10)
# ---------------------------------------------------------------------------
def R1(t, M, n):
    return (log(M) - ln_fact(t) / t) / n     # natural logs

def R2(t, Ka, n):
    return ln_comb(Ka, t) / n


# ---------------------------------------------------------------------------
# E₀(ρ₁)   –  eq. (11)
# ---------------------------------------------------------------------------
def E0(rho1, a, b):
    inner = 1 - 2 * b * rho1
    if inner <= 0:
        return -np.inf
    return rho1 * a + 0.5 * log(inner)


# ---------------------------------------------------------------------------
# E(t)   –  maximise (6)–(8)
# ---------------------------------------------------------------------------
def E_t(P_prime, t, Ka, n, M, grid=61):
    """max_{ρ,ρ₁∈[0,1]}  [-ρρ₁tR₁ - ρ₁R₂ + E₀]"""
    P_eff = P_prime            # each user sends with power P'
    best = -np.inf

    rhos  = np.linspace(0, 1, grid)
    rho1s = np.linspace(0, 1, grid)

    for rho in rhos:
        for rho1 in rho1s:
            # λ  –  eq. (8)
            D = (P_eff * t - 1) ** 2 + 4 * P_eff * t * (1 + rho * rho1) / (1 + rho)
            if D < 0:
                continue
            lam = (P_eff * t - 1 + math.sqrt(D)) / (4 * (1 + rho * rho1) * P_eff * t)
            if lam <= 0:
                continue

            # μ  –  eq. (7)
            mu = rho * lam / (1 + 2 * P_eff * t * lam)
            if mu <= 0:
                continue

            term1 = 1 + 2 * P_eff * t * lam
            term2 = 1 + 2 * P_eff * t * mu
            if term1 <= 0 or term2 <= 0:
                continue

            a = (rho / 2) * log(term1) + 0.5 * log(term2)
            b = rho * lam - mu / (1 + 2 * P_eff * t * mu)

            E0_val = E0(rho1, a, b)
            if not math.isfinite(E0_val):
                continue

            val = -rho * rho1 * t * R1(t, M, n) - rho1 * R2(t, Ka, n) + E0_val
            best = max(best, val)

    return max(0.0, best)


# ---------------------------------------------------------------------------
# p_t   –  eq. (5)
# ---------------------------------------------------------------------------
def p_t(P_prime, t, Ka, n, M):
    return math.exp(-n * E_t(P_prime, t, Ka, n, M))


# ---------------------------------------------------------------------------
# overall ε upper-bound (Theorem 1)
# ---------------------------------------------------------------------------
def epsilon_bound(Ka, M, n, P_prime, P):
    term0 = p0(Ka, M, n, P_prime, P)
    series = sum((t / Ka) * p_t(P_prime, t, Ka, n, M) for t in range(1, Ka + 1))
    return min(1.0, term0 + series)


# ---------------------------------------------------------------------------
# convert P′ →  E_b/N₀  (eq. (1))
# ---------------------------------------------------------------------------
def EbN0_dB(P_prime, M, n):
    R = log(M) / n                       # nats / channel-use
    EbN0_lin = P_prime / (2 * R)
    return 10 * math.log10(EbN0_lin)


# ---------------------------------------------------------------------------
# find the smallest P′ that gives ε ≤ ε_target
# ---------------------------------------------------------------------------
def find_Pprime(Ka, M, n, eps_target=0.1, ratio_P_to_Pprime=100, tol=1e-5):
    lo, hi = 1e-10, 50.0                 # search bracket (linear power)
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        eps = epsilon_bound(Ka, M, n, mid, mid * ratio_P_to_Pprime)
        if eps <= eps_target:
            hi = mid
        else:
            lo = mid
    return hi


# ---------------------------------------------------------------------------
# reproduce the red-dotted curve
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    n = 30_000
    M = 2 ** 100
    Ka_list, Eb_list = [], []

    for Ka in range(2, 101, 10):
        Pprime = find_Pprime(Ka, M, n)
        Eb_dB = EbN0_dB(Pprime, M, n)
        Ka_list.append(Ka)
        Eb_list.append(Eb_dB)
        print(f"Ka={Ka:2d}   P′={Pprime:.3e}   Eb/N0={Eb_dB:+.2f} dB")

    plt.figure(figsize=(7, 4))
    plt.plot(Ka_list, Eb_list, "r--", label="Theorem 1  (random-coding)")
    plt.xlabel("$K_a$  (active users)")
    plt.ylabel("$E_b/N_0$  [dB]")
    plt.grid(True);  plt.legend();  plt.tight_layout()
    plt.show()
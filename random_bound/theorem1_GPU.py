"""
theorem1_sim.py – numerical evaluation of Theorem 1 (random‑coding achievability)
from “A Perspective on Massive Random Access,” IEEE TIT 2021.

Implements
===========
• Gaussian codebook of size M with variance P′.
• Power screening: transmit 0-vector if ‖c‖² > nP.
• Active messages drawn **with replacement** (collisions allowed).
• Exact analytic p₀, p_t; Monte‑Carlo q_t.
• Binary search on P′ to satisfy ε ≤ ε_target.

Usage
-----
```bash
python theorem1_sim.py  # runs a quick Ka sweep demo
```
Adjust parameters in the `__main__` block.  Requires NumPy ≥ 1.20, SciPy ≥ 1.6.
"""
from __future__ import annotations
import math
from itertools import combinations
from typing import Optional

import numpy as np
from numpy.typing import NDArray
from scipy.stats import chi2  # type: ignore

# ---------------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------------

def ln_fact(t: int) -> float:
    return math.lgamma(t + 1.0)


def ln_comb(n: int, k: int) -> float:
    return ln_fact(n) - ln_fact(k) - ln_fact(n - k)

# ---------------------------------------------------------------------------
#  p₀ – Eq. (4)
# ---------------------------------------------------------------------------

def p0(Ka: int, M: int, n: int, P_prime: float, P: float) -> float:
    coll = math.comb(Ka, 2) / M
    thresh = n * P / P_prime
    tail = chi2.sf(thresh, df=n)
    return coll + Ka * tail

# ---------------------------------------------------------------------------
#  Rates
# ---------------------------------------------------------------------------

def R1(t: int, M: int, n: int) -> float:
    return (math.log(M) - ln_fact(t) / t) / n


def R2(t: int, Ka: int, n: int) -> float:
    return ln_comb(Ka, t) / n

# ---------------------------------------------------------------------------
#  Exponent pieces
# ---------------------------------------------------------------------------

def E0(rho1: float, a: float, b: float) -> float:
    inner = 1.0 - 2.0 * b * rho1
    return -math.inf if inner <= 0.0 else rho1 * a + 0.5 * math.log(inner)


def E_t(P_prime: float, t: int, Ka: int, n: int, M: int, grid: int = 61) -> float:
    best = -math.inf
    rhos = np.linspace(0.0, 1.0, grid)
    rho1s = np.linspace(0.0, 1.0, grid)
    for rho in rhos:
        for rho1 in rho1s:
            D = (P_prime * t - 1.0) ** 2 + 4.0 * P_prime * t * (1.0 + rho * rho1) / (
                1.0 + rho
            )
            if D < 0.0:
                continue
            lam = (P_prime * t - 1.0 + math.sqrt(D)) / (
                4.0 * (1.0 + rho * rho1) * P_prime * t
            )
            if lam <= 0.0:
                continue
            mu = rho * lam / (1.0 + 2.0 * P_prime * t * lam)
            if mu <= 0.0:
                continue
            a = (rho / 2.0) * math.log(1.0 + 2.0 * P_prime * t * lam) + 0.5 * math.log(
                1.0 + 2.0 * P_prime * t * mu
            )
            b = rho * lam - mu / (1.0 + 2.0 * P_prime * t * mu)
            e0 = E0(rho1, a, b)
            if not math.isfinite(e0):
                continue
            val = -rho * rho1 * t * R1(t, M, n) - rho1 * R2(t, Ka, n) + e0
            best = max(best, val)
    return best


def p_t(P_prime: float, t: int, Ka: int, n: int, M: int, grid: int = 61) -> float:
    return math.exp(-n * E_t(P_prime, t, Ka, n, M, grid))

# ---------------------------------------------------------------------------
#  Ensemble utilities
# ---------------------------------------------------------------------------

def generate_codebook(M: int, n: int, P_prime: float, rng: np.random.Generator) -> NDArray:
    return rng.normal(scale=math.sqrt(P_prime), size=(M, n))


def codeword_norms(C: NDArray) -> NDArray:
    return np.sum(C * C, axis=1)


def draw_active_indices(M: int, Ka: int, rng: np.random.Generator) -> NDArray:
    return rng.integers(0, M, size=Ka)


def encode_active(
    C: NDArray,
    norms: NDArray,
    active: NDArray,
    P: float,
    n: int,
) -> NDArray:
    X = np.zeros((active.size, n))
    mask = norms[active] <= n * P
    X[mask] = C[active[mask]]
    return X

# ---------------------------------------------------------------------------
#  I_t – Eq. (13)
# ---------------------------------------------------------------------------

def I_t_from_active(X: NDArray, y: NDArray, t: int, P_prime: float) -> float:
    Ka, n = X.shape
    C_t = 0.5 * math.log(1.0 + P_prime * t)
    best = math.inf
    for S0 in combinations(range(Ka), t):
        S0 = np.fromiter(S0, dtype=int)
        Sc0 = np.setdiff1d(np.arange(Ka), S0, assume_unique=True)
        a = X[S0].sum(axis=0)
        b = X[Sc0].sum(axis=0)
        term1 = np.linalg.norm(y - b) ** 2 / (1.0 + P_prime * t)
        term2 = np.linalg.norm(y - a - b) ** 2
        info = n * C_t + 0.5 * (term1 - term2)
        best = min(best, info)
    return best

# ---------------------------------------------------------------------------
#  q_t via Monte‑Carlo
# ---------------------------------------------------------------------------

def q_t(
    t: int,
    Ka: int,
    n: int,
    P_prime: float,
    M: int,
    P: float,
    N_mc: int = 5000,
    rng: Optional[np.random.Generator] = None,
) -> float:
    rng = np.random.default_rng() if rng is None else rng
    I_samples = np.empty(N_mc)
    for s in range(N_mc):
        C = generate_codebook(M, n, P_prime, rng)
        norms = codeword_norms(C)
        active = draw_active_indices(M, Ka, rng)
        X = encode_active(C, norms, active, P, n)
        y = X.sum(axis=0) + rng.normal(size=n)
        I_samples[s] = I_t_from_active(X, y, t, P_prime)
    R1v, R2v = R1(t, M, n), R2(t, Ka, n)
    a_star = n * (t * R1v + R2v)
    g_min = float(I_samples.min() - 5.0)
    g_max = float(max(a_star + 20.0, I_samples.max() + 5.0))
    gammas = np.linspace(g_min, g_max, 400)
    best = math.inf
    for g in gammas:
        prob = np.mean(I_samples <= g)
        union = math.exp(a_star - g)
        best = min(best, prob + union)
    return min(best, 1.0)

# ---------------------------------------------------------------------------
#  ε upper‑bound
# ---------------------------------------------------------------------------

def epsilon_bound(
    Ka: int,
    M: int,
    n: int,
    P_prime: float,
    P: float,
    grid_Et: int = 61,
    N_mc_qt: int = 5000,
    rng: Optional[np.random.Generator] = None,
) -> float:
    rng = np.random.default_rng() if rng is None else rng
    term0 = p0(Ka, M, n, P_prime, P)
    series = sum(
        (t / Ka) * p_t(P_prime, t, Ka, n, M, grid_Et) for t in range(2, Ka + 1)
    )
    pt1 = p_t(P_prime, 1, Ka, n, M, grid_Et)
    qt1 = q_t(1, Ka, n, P_prime, M, P, N_mc_qt, rng)
    term2 = (1.0 / Ka) * min(pt1, qt1)
    return min(1.0, term0 + series + term2)

# ---------------------------------------------------------------------------
#  P′ search
# ---------------------------------------------------------------------------

def find_Pprime(
    Ka: int,
    M: int,
    n: int,
    eps_target: float,
    P: float,
    grid_Et: int = 61,
    N_mc_qt: int = 5000,
    rng: Optional[np.random.Generator] = None,
) -> float:
    rng = np.random.default_rng() if rng is None else rng
    lo, hi = 1e-6, P * 0.999
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        eps = epsilon_bound(Ka, M, n, mid, P, grid_Et, N_mc_qt, rng)
        if eps <= eps_target:
            hi = mid
        else:
            lo = mid
    return hi

# ---------------------------------------------------------------------------
#  Main demo sweep
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    n = 300            # codeword length
    M = 2 ** 12        # messages per user (4096)
    eps_tgt = 0.1
    P = 1.0            # average transmit power constraint

    Ka_values = list(range(30, 81, 10))
    rng = np.random.default_rng(0)

    Eb_dB = []
    for Ka in Ka_values:
        P_prime = find_Pprime(
            Ka, M, n, eps_tgt, P, grid_Et=21, N_mc_qt=400, rng=rng
        )
        Eb_dB.append(10.0 * math.log10(P / (2.0 * math.log(M, 2) / n)))
        print(f"K_a={Ka}   P'={P_prime:.4g}   Eb/N0≈{Eb_dB[-1]:.2f} dB")

    plt.plot(Ka_values, Eb_dB, "bo-")
    plt.xlabel("$K_a$")
    plt.ylabel("$E_b/N_0$ [dB]")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

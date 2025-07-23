#!/usr/bin/env python3
# --------------------------------------------
# Random-coding achievability – fixed R1(t)
# --------------------------------------------
import math, numpy as np, matplotlib.pyplot as plt
from math import lgamma, log
from scipy.stats import chi2, norm

# ------------ utilities ----------------------------------------------------
ln_fact = lambda t: lgamma(t + 1)
ln_comb = lambda n,k: ln_fact(n) - ln_fact(k) - ln_fact(n-k)

def R1(t, M, n):                    # <-- FIXED!
    return (log(M) - ln_fact(t)) / n

def R2(t, Ka, n):
    return ln_comb(Ka, t) / n

# ------------ p0 -----------------------------------------------------------
def p0(Ka, M, n, Pp, Ppeak):
    return math.comb(Ka,2)/M + Ka*chi2.sf(n*Ppeak/Pp, df=n)

# ------------ E(t) & p_t ---------------------------------------------------
def E_t(Pp, t, Ka, n, M, grid=81):
    best = -np.inf
    for rho in np.linspace(0,1,grid):
        for rho1 in np.linspace(0,1,grid):
            D = (Pp*t-1)**2 + 4*Pp*t*(1+rho*rho1)/(1+rho)
            if D < 0: continue
            lam = (Pp*t-1 + math.sqrt(D)) / (4*(1+rho*rho1)*Pp*t)
            if lam<=0: continue
            mu  = rho*lam / (1+2*Pp*t*lam)
            if mu<=0:  continue
            term1,term2 = 1+2*Pp*t*lam, 1+2*Pp*t*mu
            if term1<=0 or term2<=0: continue
            a = (rho/2)*log(term1) + 0.5*log(term2)
            b = rho*lam - mu/(1+2*Pp*t*mu)
            inner = 1 - 2*b*rho1
            if inner<=0: continue
            E0 = rho1*a + 0.5*log(inner)
            val = -rho*rho1*t*R1(t,M,n) - rho1*R2(t,Ka,n) + E0
            best = max(best, val)
    return max(0.0, best)

def p_t(Pp, t, Ka, n, M):
    return math.exp(-n * E_t(Pp, t, Ka, n, M))

# ------------ q_t (Chernoff, correct R1) -----------------------------------
def q_t(Pp, t, Ka, n, M, g_pts=200):
    mu  = 0.5*log(1 + Pp*t)
    var = (Pp*t)/(1 + Pp*t)
    sig = math.sqrt(var)
    rate = n*(t*R1(t,M,n) + R2(t,Ka,n))

    gammas = np.linspace(mu-8*sig, mu+2*sig, g_pts)
    tail   = norm.cdf((gammas-mu)/sig)              # P[I_t ≤ γ]
    term   = np.exp(rate - gammas)                  # e^{n(tR1+R2)-γ}
    return float(np.min(tail + term))

# ------------ ε(P′) --------------------------------------------------------
def eps(Ka,M,n,Pp,Ppeak):
    series = sum((t/Ka)*min(p_t(Pp,t,Ka,n,M), q_t(Pp,t,Ka,n,M))
                 for t in range(1,Ka+1))
    return min(1.0, p0(Ka,M,n,Pp,Ppeak) + series)

# ------------ search for P′ -----------------------------------------------
def find_Pprime(Ka,M,n,eps_target=0.1,ratio=100):
    lo,hi = 1e-10, 1e-3
    while eps(Ka,M,n,hi,hi*ratio) > eps_target:
        hi *= 2
    for _ in range(60):
        mid = 0.5*(lo+hi)
        if eps(Ka,M,n,mid,mid*ratio) <= eps_target:
            hi = mid
        else:
            lo = mid
    return hi

# ------------ Eb/N0 --------------------------------------------------------
def EbN0_dB(Pp,M,n):
    R = log(M)/n
    return 10*math.log10(Pp/(2*R))

# ------------ main ---------------------------------------------------------
if __name__ == "__main__":
    n, M = 30_000, 2**100
    print("Ka   P'        Eb/N0(dB)")
    print("--   --------  ---------")
    Kas, Eb = [], []
    for Ka in range(2, 21, 2):
        Pp  = find_Pprime(Ka,M,n)
        EbN = EbN0_dB(Pp,M,n)
        Kas.append(Ka); Eb.append(EbN)
        print(f"{Ka:2d}   {Pp:8.3e}   {EbN:9.2f}")

    plt.plot(Kas,Eb,"r--",lw=2)
    plt.xlabel("$K_a$  (active users)"); plt.ylabel("$E_b/N_0$  [dB]")
    plt.ylim(-2,10); plt.grid(True); plt.tight_layout(); plt.show()
import scipy.special as sp
import numpy as np
import math
import tqdm

# ==================================
# example parameters (for debugging)
# ==================================
k      = 100   # number of bits per symbol
n      = 30000 # frame length
Ka     = 200   # number of active device
EbN0db = 2     # energy per bit
Niq    = 1000  # Number of samples for CDF evaluation

# =======================
# power and codebook size
# =======================
P = 10 ** (EbN0db / 10) * k / n
M = 2 ** k

# ==============
# initialization
# ==============
t_vec    = np.arange(1, Ka + 1)
rho_vec  = np.linspace(0, 1, 100)
rho1_vec = np.linspace(0, 1, 100)
t        = np.tile(t_vec, (len(rho1_vec), len(rho_vec), 1))
rho      = np.transpose(np.tile(rho_vec, (len(rho1_vec), len(t_vec), 1)), [0, 2, 1])
rho1     = np.transpose(np.tile(rho1_vec, (len(rho_vec), len(t_vec), 1)), [2, 0, 1])

# =========================================
# Optimization over P' (here denoted by P1)
# =========================================
P1_vec  = np.linspace(1e-9, P, 20)
epsilon = np.zeros_like(P1_vec)

for pp in tqdm.tqdm(range(len(P1_vec))):
    # extract P'
    P1 = P1_vec[pp]
    # compute of p0
    p0 = math.comb(Ka, 2) / M + Ka * sp.gammaincc(n, n * P / P1)
    # compute R1, R2
    R1 = 1 / n * np.log(float(M)) - 1 / (n * t) * sp.gammaln(t + 1)
    R2 = 1 / n * (sp.gammaln(Ka + 1) - sp.gammaln(t + 1) - sp.gammaln(Ka - t + 1))
    # compute D
    D = (P1 * t - 1) ** 2 + 4 * P1 * t * (1 + rho * rho1) / (1 + rho)
    # compute lambda_
    lambda_ = (P1 * t - 1 + np.sqrt(D)) / (2 * (1 + rho1 * rho) * P1 * t)
    # compute mu
    mu = rho * lambda_ / (1 + P1 * t * lambda_)
    # compute a, b
    a = rho * np.log(1 + P1 * t * lambda_) + np.log(1 + P1 * t * mu)
    b = rho * lambda_ - mu / (1 + P1 * t * mu)
    # compute E0
    E0 = rho1 * a + np.log(1 - b * rho1)
    # compute Et
    Et = np.squeeze(np.max(np.max(-rho * rho1 * t * R1 - rho1 * R2 + E0, axis=0), axis=0))
    # compute pt
    pt = np.exp(-n * Et)

    # compute of qt (for t = 1 only, as in [1])
    It = np.zeros(Niq)
    for II in range(Niq):
        Zi       = np.sqrt(0.5) * (np.random.randn(n) + 1j * np.random.randn(n))
        codebook = np.sqrt(0.5 * P1) * (np.random.randn(Ka, n) + 1j * np.random.randn(Ka, n))
        it       = n * np.log(1 + P1) + \
                  (np.sum(np.abs(np.tile(Zi, (Ka, 1)) + codebook) ** 2, axis=1) / (1 + P1) - \
                   np.sum(np.abs(np.tile(Zi, (Ka, 1))) ** 2, axis=1))
        It[II]   = np.min(it)

    # use numpy for empirical CDF
    gamma = np.sort(It)
    prob = np.arange(1, len(gamma) + 1) / len(gamma)
    R1 = 1 / n * np.log(float(M)) - 1 / (n * 1) * sp.gammaln(Ka)
    R2 = 1 / n * (sp.gammaln(Ka + 1) - sp.gammaln(2) - sp.gammaln(Ka))
    qt = np.min(prob + np.exp(n * (R1 + R2) - gamma))
    # compute RHS of [1, Eq. (3)] for a given P' < P
    epsilon[pp] = t_vec[0] / Ka * np.min([pt[0], qt]) + np.sum(t_vec[1:] / Ka * pt[1:]) + p0

# Find the minimum over P'
print(epsilon)
idx         = np.argmin(epsilon)
epsilon_min = epsilon[idx]
P1_opt      = P1_vec[idx]
print(epsilon_min, P1_opt)
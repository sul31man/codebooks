###this script will help us to determine q(t) 
#first we need to generate our gaussian codebook of size Ka,n 
import numpy as np
from itertools import combinations
from math import lgamma, log

def ln_fact(t):
    """natural log of t!  via lgamma"""
    return lgamma(t + 1)

def ln_comb(n, k):
    """natural log of C(n,k)"""
    return ln_fact(n) - ln_fact(k) - ln_fact(n - k)

def R1(t, M, n):
    return (log(M) - ln_fact(t) / t) / n     # natural logs

def R2(t, Ka, n):
    return ln_comb(Ka, t) / n


def generate_codebook(K_a, n, P_prime):
    C = np.random.randn(K_a, n)
    C /= np.linalg.norm(C, axis=1, keepdims=True)  # normalize each row to unit norm
    C *= np.sqrt(n * P_prime)                      # scale to correct power
    return C

C = generate_codebook(5, 3, 5)
print(C)

##now we need to use this codebook to generate a signal from this codebook of length Ka

def generate_signal(codebook, Ka, n):

    #first lets pick Ka random numbers (with repetition allowed)

    random_numbers = np.random.randint(0, len(codebook), size=Ka)

    signal = np.zeros(n)

    for number in random_numbers:

        signal += codebook[number]

    
    return signal

def add_noise(signal, P_prime, Ka):
    """
    Adds AWGN noise to 'signal', using total SNR = Ka * P_prime.
    """
    signal_power = np.mean(signal**2)
    SNR_linear = Ka * P_prime
    noise_power = signal_power / SNR_linear
    noise = np.random.randn(signal.shape[0]) * np.sqrt(noise_power)
    return signal + noise


def I_t(t, Ka, codebook, n, P_prime):
    # Step 1: Generate true signal
    signal = generate_signal(codebook, Ka, n)                   # Sum of all K_a codewords
    Y = add_noise(signal, P_prime, Ka)                  # Add Gaussian noise

    # Step 2: Loop through all subsets of t users
    C_t = 0.5 * np.log(1 + P_prime * t)
    min_info = float('inf')

    all_combos = list(combinations(range(Ka), t))

    for S0 in all_combos:
        S0 = list(S0)
        S0_c = list(set(range(Ka)) - set(S0))         # Complement subset

        a = codebook[S0].sum(axis=0)                  # Sum of t codewords
        b = codebook[S0_c].sum(axis=0)                # Sum of remaining Ka - t

        term1 = np.linalg.norm(Y - b)**2 / (1 + P_prime * t)
        term2 = np.linalg.norm(Y - a - b)**2 / (1 + P_prime * t)
        info = n * C_t + (np.log(np.e) / 2) * (term1 - term2)

        min_info = min(min_info, info)

    return min_info


def q_t(t, Ka, codebook, n, P_prime, M):


    I_t_samples = [I_t(t, Ka, codebook, n, P_prime) for _ in range(100)]

    gamma_vals = np.linspace(min(I_t_samples), max(I_t_samples), 100)
    
    R1_value = R1(t, M, n)
    R2_value = R2(t, Ka, n)
    smallest = float('inf')
    for gamma in gamma_vals:

        probability = np.mean(np.array(I_t_samples) < gamma)

        term = probability + np.exp(-n*(R1_value + R2_value)*t - gamma)

        smallest = min(term, smallest)


    return smallest

Ka = 10
n = 8
P_prime = 5
M = 4
t =2
codebook = generate_codebook(Ka, n, P_prime)
print(q_t(t, Ka, codebook, n, P_prime, M))








    





    



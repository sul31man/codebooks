import math 
import numpy as np
from scipy import stats
import itertools
import matplotlib.pyplot as plt
from scipy.special import factorial


def p_o(Ka, M, n, P_prime, P):
    """
    Calculate probability for sum of squared Gaussian noise terms
    
    Args:
        Ka: Parameter Ka
        M: Parameter M
        n: Number of samples
        P_prime: Power parameter P'
        P: Power parameter P
        
    Returns:
        Probability value
    """
    first_term = math.comb(Ka, 2)/M  # Using comb instead of combination for newer versions
    
    # For sum of n squared Gaussian random variables (chi-square distribution)
    threshold = P_prime / P
    
    # Degrees of freedom = n (number of squared terms being summed)
    df = n
    
    # Calculate probability using chi-square distribution
    # The sum of n squared standard normal variables follows chi-square distribution
    # We scale by 1/n to account for the averaging
    scaled_threshold = n * threshold
    prob = 1 - stats.chi2.cdf(scaled_threshold, df)
    
    return first_term  + Ka*prob

def mu(P_prime, t, ro, ro1):

    lamb = lamb(P_prime, t, ro, ro1)
    mu = (ro*lamb) / (1 + 2*P_prime*t*lamb)

    return mu

def b(P_prime, t, ro, ro1):
    
    lamb = lamb(P_prime, t, ro, ro1)
    mu = mu(ro, lamb, P_prime, t)
    b = ro*lamb - mu/ (1 + 2*P_prime*t*mu)

    return b

def lamb(P_prime, t, ro, ro1):
    
    D = D(P_prime, t, ro, ro1)
    lamb = P_prime*t-1+math.sqrt(D) / (4*(1 + ro1*ro)*P_prime*t)
    
    return lamb 

def D(P_prime, t, ro, ro1):

    D = (P_prime*t -1)**2 + 4*P_prime*t*(1 + ro1*ro)/(1 + ro)

    return D

def calculate_R1(t, M, n):
    """Calculate R1 from equation (9)"""
    # Use math.log2 for scalar values
    return (1/n) * math.log2(M) - (1/(n*t)) * math.log2(factorial(t))

def calculate_R2(t, n, Ka):
    """Calculate R2 from equation (10)"""
    # Use math.log2 for scalar values
    return (1/n) * math.log2(math.comb(Ka, t))

def Et(P_prime, t, Ka, n, M):
    """Calculate E(t) using equation in paper"""
    R1_val = calculate_R1(t, M, n)
    R2_val = calculate_R2(t, n, Ka)

    best = float('-inf')
    
    # Create a finer grid for better optimization
    X = np.linspace(0, 1, 50)  # rho values
    Y = np.linspace(0, 1, 50)  # rho1 values
    
    for x in X:  # rho
        for y in Y:  # rho1
            try:
                # Calculate E0
                a_val = a(P_prime, t, x, y)
                b_val = b(P_prime, t, x, y)
                if 1 - 2*b_val*y <= 0:  # Check for valid log argument
                    continue
                E0_val = y*a_val + 0.5*math.log2(1 - 2*b_val*y)
                
                # Calculate full term
                term = -x*y*t*R1_val - y*R2_val + E0_val
                
                if term > best:
                    best = term
            except:
                continue
    
    return best

def p_t(P_prime, t, Ka, n, M):
    """Calculate p_t using equation (5)"""
    E_value = Et(P_prime, t, Ka, n, M)
    return np.exp(-n * E_value)

def a(P_prime, t, ro, ro1):

    mu = mu(P_prime, t, ro, ro1)
    term = ro/2 * math.log2(1 + 2*P_prime*t*mu) + 1/2*math.log2(1 + 2*P_prime*t*mu)

    return term

def E0(P_prime, t, ro, ro1):

    a = a(P_prime, t, ro, ro1)
    b = b(P_prime, t, ro, ro1)

    E0 = ro1*a + 1/2*math.log2(1-2*b*ro1)

    return E0

def information_density(y, a, b, n, P_prime, t):
    """
    Calculate the information density i_t(a;y|b)
    Args:
        y: received signal vector (n-dimensional)
        a: transmitted signal vector (n-dimensional)
        b: interference signal vector (n-dimensional)
        n: number of samples
        P_prime: power constraint
        t: time index
    """
    Ct = 0.5 * np.log2(1 + P_prime * t)
    
    # Calculate the L2 norms
    term1 = np.linalg.norm(y - b)**2 / (1 + P_prime * t)
    term2 = np.linalg.norm(y - a - b)**2
    
    return n * Ct + (np.log(math.e)/2) * (term1 - term2)

def calculate_It(y, codewords, t, n, P_prime, Ka):
    """
    Calculate I_t = min_{S0} i_t(c(S0);Y|c(S0^c))
    Args:
        y: received signal vector (n-dimensional)
        codewords: dictionary mapping user indices to their codewords (each n-dimensional)
        t: time index (size of subsets to consider)
        n: number of samples
        P_prime: power constraint
        Ka: total number of users
    Returns:
        Minimum information density over all t-subsets
    """
    # Get all possible t-subsets of [Ka]
    all_subsets = list(itertools.combinations(range(Ka), t))
    
    min_value = float('inf')
    for S0 in all_subsets:
        # Construct c(S0) by summing codewords in S0
        c_S0_sum = np.zeros(n)
        for user in S0:
            c_S0_sum += codewords[user]
            
        # Construct c(S0^c) by summing codewords not in S0
        c_S0c_sum = np.zeros(n)
        for user in set(range(Ka)) - set(S0):
            c_S0c_sum += codewords[user]
            
        # Calculate information density for this subset
        value = information_density(y, c_S0_sum, c_S0c_sum, n, P_prime, t)
        min_value = min(min_value, value)
    
    return min_value

def q_t(P_prime, t, Ka, n, M, y=None, codewords=None):
    """
    Calculate q_t as defined in the paper
    Args:
        P_prime: power constraint
        t: time index
        Ka: total users
        n: number of samples
        M: parameter M
        y: received signal vector (optional)
        codewords: dictionary of user codewords (optional)
    """
    R1_val = calculate_R1(t, M, n)
    R2_val = calculate_R2(t, n, Ka)
    
    if y is None or codewords is None:
        # If no signals provided, return just the exponential term with some default gamma
        gamma = n * (t*R1_val + R2_val) / 2  # reasonable default
        return np.exp(n * (t*R1_val + R2_val) - gamma)
    
    # To find inf_γ, we'll try a range of γ values
    gamma_range = np.linspace(0, n * (t*R1_val + R2_val) * 2, 100)  # Adjust range as needed
    min_qt = float('inf')
    
    It_val = calculate_It(y, codewords, t, n, P_prime, Ka)
    
    for gamma in gamma_range:
        # P[It ≤ γ] is 1 if It_val ≤ γ, 0 otherwise
        prob_term = float(It_val <= gamma)
        exp_term = np.exp(n * (t*R1_val + R2_val) - gamma)
        
        qt_gamma = prob_term + exp_term
        min_qt = min(min_qt, qt_gamma)
    
    return min_qt

def calculate_epsilon(Ka, M, n, P_prime, P, y=None, codewords=None):
    """
    Calculate the error bound epsilon from equation (3):
    ε ≤ Σ(t=1 to Ka) [t/Ka * min(pt, qt)] + p0
    
    Args:
        Ka: number of users
        M: parameter M
        n: number of samples
        P_prime: power constraint P'
        P: power constraint P
        y: received signal (optional)
        codewords: dictionary of user codewords (optional)
    Returns:
        Upper bound on epsilon
    """
    # First calculate p0
    p0_value = p_o(Ka, M, n, P_prime, P)
    
    # Initialize sum
    sum_term = 0
    
    # For each t from 1 to Ka
    for t in range(1, Ka + 1):
        # Calculate pt using equation (5)
        pt_value = p_t(P_prime, t, Ka, n, M)
        
        # Calculate qt using equation (11)
        qt_value = q_t(P_prime, t, Ka, n, M, y, codewords)
        
        # Take minimum of pt and qt
        min_pq = min(pt_value, qt_value)
        
        # Multiply by t/Ka and add to sum
        sum_term += (t/Ka) * min_pq
    
    # Final result is sum plus p0
    epsilon = sum_term + p0_value
    
    return epsilon

def calculate_epsilon_for_parameters(Ka_range, M_range, n_range, P_prime, P):
    """
    Helper function to find good parameters that minimize epsilon
    
    Args:
        Ka_range: list of Ka values to try
        M_range: list of M values to try
        n_range: list of n values to try
        P_prime: power constraint P'
        P: power constraint P
    Returns:
        Dictionary with best parameters and resulting epsilon
    """
    best_epsilon = float('inf')
    best_params = {}
    
    for Ka in Ka_range:
        for M in M_range:
            for n in n_range:
                try:
                    eps = calculate_epsilon(Ka, M, n, P_prime, P)
                    if eps < best_epsilon:
                        best_epsilon = eps
                        best_params = {
                            'Ka': Ka,
                            'M': M,
                            'n': n,
                            'epsilon': eps
                        }
                except Exception as e:
                    continue
    
    return best_params

def plot_results(Ka_range, M_range, n_range, P_prime, P):
    """
    Plot the results of parameter optimization
    """
    # Create figure with multiple subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Store results for different parameters
    results = []
    
    for Ka in Ka_range:
        try:
            eps = calculate_epsilon(Ka, M_range[0], n_range[0], P_prime, P)
            results.append({
                'Ka': Ka,
                'epsilon': eps
            })
            print(f"Calculated epsilon for Ka={Ka}: {eps}")
        except Exception as e:
            print(f"Error for Ka={Ka}: {str(e)}")
            continue
    
    # Convert to numpy arrays for easier plotting
    if results:
        data = np.array([(r['Ka'], r['epsilon']) for r in results])
        
        # Plot 1: Epsilon vs Ka (linear scale)
        ax1.plot(data[:, 0], data[:, 1], marker='o')
        ax1.set_xlabel('Number of Users (Ka)')
        ax1.set_ylabel('Epsilon')
        ax1.set_title('Error Bound vs Number of Users (Linear Scale)')
        ax1.grid(True)
        
        # Plot 2: Epsilon vs Ka (log scale)
        ax2.semilogy(data[:, 0], data[:, 1], marker='o')
        ax2.set_xlabel('Number of Users (Ka)')
        ax2.set_ylabel('Epsilon (log scale)')
        ax2.set_title('Error Bound vs Number of Users (Log Scale)')
        ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('epsilon_analysis.png')
    plt.close()

def plot_power_analysis(Ka, M, n, P_prime_range, P_range):
    """
    Analyze and plot how epsilon varies with power levels
    
    Args:
        Ka: Fixed number of users
        M: Fixed M parameter
        n: Fixed number of samples
        P_prime_range: List of P' values to try
        P_range: List of P values to try
    """
    results = []
    
    # For each P, try all valid P' values (P' < P)
    for P in P_range:
        for P_prime in P_prime_range:
            if P_prime >= P:  # Skip invalid combinations
                continue
            try:
                eps = calculate_epsilon(Ka, M, n, P_prime, P)
                results.append({
                    'P': P,
                    'P_prime': P_prime,
                    'epsilon': eps
                })
                print(f"Calculated epsilon for P={P}, P'={P_prime}: {eps}")
            except Exception as e:
                print(f"Error for P={P}, P'={P_prime}: {str(e)}")
                continue
    
    if not results:
        print("No valid results to plot!")
        return
    
    # Create figure with multiple subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Convert results to numpy arrays
    data = np.array([(r['P'], r['P_prime'], r['epsilon']) for r in results])
    
    # Plot 1: 3D scatter of epsilon vs P and P'
    scatter = ax1.scatter(data[:, 0], data[:, 1], c=np.log10(data[:, 2]), 
                         cmap='viridis')
    ax1.set_xlabel('P')
    ax1.set_ylabel('P\'')
    ax1.set_title('Log10(Epsilon) vs Powers (P and P\')')
    plt.colorbar(scatter, ax=ax1, label='Log10(Epsilon)')
    
    # Plot 2: Epsilon vs P for different P' values
    unique_P_primes = np.unique(data[:, 1])
    for P_prime in unique_P_primes:
        mask = data[:, 1] == P_prime
        ax2.semilogy(data[mask, 0], data[mask, 2], 
                    label=f"P'={P_prime}", marker='o')
    ax2.set_xlabel('P')
    ax2.set_ylabel('Epsilon (log scale)')
    ax2.set_title('Epsilon vs P for different P\' values')
    ax2.legend()
    ax2.grid(True)
    
    # Plot 3: Epsilon vs P' for different P values
    unique_Ps = np.unique(data[:, 0])
    for P in unique_Ps:
        mask = data[:, 0] == P
        ax3.semilogy(data[mask, 1], data[mask, 2], 
                    label=f"P={P}", marker='o')
    ax3.set_xlabel('P\'')
    ax3.set_ylabel('Epsilon (log scale)')
    ax3.set_title('Epsilon vs P\' for different P values')
    ax3.legend()
    ax3.grid(True)
    
    # Plot 4: Epsilon vs P/P' ratio
    ratio = data[:, 0] / data[:, 1]
    ax4.semilogy(ratio, data[:, 2], 'o-')
    ax4.set_xlabel('P/P\' ratio')
    ax4.set_ylabel('Epsilon (log scale)')
    ax4.set_title('Epsilon vs P/P\' ratio')
    ax4.grid(True)
    
    plt.tight_layout()
    plt.savefig('power_analysis.png')
    plt.close()

if __name__ == "__main__":
    # Fixed parameters
    M_test = 2**100
    n_test = 30000
    Ka_test = 5  # Fix Ka to analyze power effects
    
    # Power ranges to try
    P_prime_range = np.linspace(0.1, 10, 10)  # P' from 0.1 to 10
    P_range = np.linspace(1, 20, 10)          # P from 1 to 20
    
    print(f"\nAnalyzing epsilon values for different power levels:")
    print(f"Fixed parameters: Ka={Ka_test}, M=2^100, n={n_test}")
    print(f"P' range: {min(P_prime_range)} to {max(P_prime_range)}")
    print(f"P range: {min(P_range)} to {max(P_range)}")
    print("\nCalculating values...")
    
    # Analyze how epsilon varies with power
    plot_power_analysis(Ka_test, M_test, n_test, P_prime_range, P_range)
    print("\nPlots have been saved as 'power_analysis.png'")






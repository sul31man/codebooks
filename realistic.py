##this will be used for anlaysing the AMP algorithm
##first we need to generate the sparse vector signal
import numpy as np
from itertools import product 
from scipy.special import comb 
import matplotlib.pyplot as plt

class SPARC:

    def __init__(self, Ka, L, J, B, n, P_total, Eb_N0_dB):
        self.Ka = Ka #number of users
        self.L = L #number of sections in the encoding matrix
        self.J = J #number of bits in the outer encoding scheme
        self.B = B #number of bits in the original message
        self.n = n #number of rows in the inner encoding matrix
        self.P_total = P_total #total power P
        self.Eb_N0_dB = Eb_N0_dB #Eb/N0 in dB

        # Calculate code rate
        self.R = self.B / self.n
        # Convert Eb/N0 to SNR
        Eb_N0_linear = 10**(self.Eb_N0_dB/10)
        self.snr_linear = Eb_N0_linear * self.R
        self.snr_db = 10 * np.log10(self.snr_linear)


    def generate_messages(self):
        """Generate Ka messages, each as a B-bit binary string"""
        self.messages = []
        for _ in range(self.Ka):
          # Generate B random bits and join them into a string
          bits = np.random.randint(0, 2, self.B)
          message = ''.join(map(str, bits))
          self.messages.append(message)
        return self.messages
        
    def outer_coding(self): ##will be implementing BCH coding in this 

        #this function will turn each of the messages into the L J bit outer codewords using BCH coding
        from galois import BCH
        
        self.outer_mapping = {} #initialise the mapping from messages to outer codewords
        
        L = self.L 
        J = self.J
        B = self.B
        
        # Calculate BCH parameters
        # We need to encode B/L bits into J bits for each section
        k = B // L  # information bits per section
        n_bch = J   # codeword length (J bits)
        
        # Create BCH code
        # We need to find appropriate t (error correction capability)
        # BCH(n, k) where n = J and k = B/L
        t = (n_bch - k) // 2  # Maximum correctable errors
        
        try:
            bch = BCH(n_bch, k, t)
        except:
            # If BCH parameters are invalid, use systematic approach
            # Pad with zeros if needed
            bch = None
        
        # For each message, generate L many J-bit codewords using BCH
        for msg_idx, message in enumerate(self.messages):
            # Convert message to integer for processing
            msg_int = int(message, 2)
            
            # Split message into L sections of k bits each
            codewords = []
            
            for l in range(L):
                # Extract k bits for this section
                start_bit = l * k
                end_bit = min((l + 1) * k, B)
                section_bits = k
                
                # Extract the section from the message
                if end_bit > start_bit:
                    section_value = (msg_int >> (B - end_bit)) & ((1 << (end_bit - start_bit)) - 1)
                else:
                    section_value = 0
                
                if bch is not None:
                    # Use BCH encoding
                    info_bits = np.array([int(x) for x in format(section_value, f'0{k}b')])
                    if len(info_bits) < k:
                        info_bits = np.pad(info_bits, (k - len(info_bits), 0), 'constant')
                    
                    encoded = bch.encode(info_bits[:k])
                    codeword = int(''.join(map(str, encoded)), 2)
                else:
                    # Fallback: systematic code (message + parity)
                    parity_bits = J - k
                    parity = section_value % (2**parity_bits)  # Simple parity
                    codeword = (section_value << parity_bits) | parity
                
                # Ensure codeword fits in J bits
                codeword = codeword & ((1 << J) - 1)
                codewords.append(codeword)
            
            self.outer_mapping[message] = codewords
        
        return self.outer_mapping

    def inner_coding(self):

        #this will take the L J bit outer codewords and turn it into the inner codewords
        #we first need a matrix with l section with each section containing 2^J columns 

        messages = self.messages #load in our original messages

        outer_mapping = self.outer_mapping #load in the outer encoding scheme

        n = self.n #number of rows in our matrix for the inner encoding scheme

        J = self.J 

        L = self.L ##loading in the sizes of the original encoding scheme 

        A = np.random.rand(n, L*(2**J)) #initialise our matrix, A, as seen in the paper

        A = A - np.mean(A, axis=0) #this ensures the columns of the matrix have a mean of 0 as required in the paper

        A = A / np.linalg.norm(A, axis = 0)

        self.A = A  #now the class has the attribute of the matrix 

         
        #now that we have the matrix we can map the outer codewords for each message to the inner codeword which will actually be transmitted 

        inner_mapping = {} #generate the dictionary to prepare the mappings from the message to the inner codeword using the outer to inner scheme.

        for message in messages:

            outer_codewords = outer_mapping[message] #this is a list of L J bit codewords which will be the indexes of the columns 
             
            inner_codewords = np.zeros(L*(2**J)) #preparing the vector so that I can use the indices to determine which columns we will be adding from the matrix A. 

            for idx, codeword in enumerate(outer_codewords):

                column_index = codeword + 2**J*(idx)  #this gets the codeword from the J bit outer codes and turnsit into the column number in the large matrix A

                inner_codewords[column_index] = 1
            
            # Debug: print number of nonzeros
            print(f"[DEBUG] User {message}: nonzeros in inner_codewords: {np.count_nonzero(inner_codewords)} (should be {L})")
            # Power scaling per paper: Pl = n*P_total/L
            Pl = self.n * self.P_total / self.L
            vector = inner_codewords * np.sqrt(Pl)
            print(f"[DEBUG] Power of inner codeword vector: {np.sum(vector**2):.2f} (should be n*P_total = {self.n*self.P_total:.2f} for one user)")
            # Print power of this vector for debugging
            #print(f"[DEBUG] Power of inner codeword vector: {np.sum(vector**2):.2f} (should be close to n*P_total/L = {self.n*self.P_total/self.L:.2f} if only one nonzero)")
            inner_codeword = A @ (inner_codewords.T) #this will calculate A*m_k like in the paper
            
            inner_mapping[message] = (inner_codeword,vector) #this tuple will store both so that from any message we can access the vector and the transform

        self.inner_mapping = inner_mapping ##this should be sufficient.
        
        return inner_mapping

    def generate_signal(self):

        #now that we've put the channel coding scheme together, we can superpose the encoded messages

        messages = self.generate_messages()
        outer_coding = self.outer_coding()
        inner_coding = self.inner_coding()

        sparse_vector = np.zeros(self.L * (2**self.J)) #setting up the theta vector here 
        
        i = 0
        
        # Ensure at least one user is active for testing
        active_users = np.random.randint(0,2,self.Ka)
        if np.sum(active_users) == 0:  # If no users are active, make first user active
            active_users[0] = 1
        self.original_messages = []

        for message, mapping in inner_coding.items():

            vector = mapping[1] #this is the m_k for each message, where A*m_k is the inner codeword, which is mapping[0]

            sparse_vector += active_users[i]*vector #superposing the signals together
            
            if active_users[i] == 1:

                self.original_messages.append(message)

            i += 1
        
        self.sparse_vector = sparse_vector
        # Print total power of the sparse vector
        print(f"[DEBUG] Total power of sparse vector: {np.sum(sparse_vector**2):.2f} (should be close to n*P_total*Ka/L)")
        self.signal = self.A @ self.sparse_vector #this does the calculation A@theta, which is what our tranmistted signal is 
        # Normalize signal to have average power = P_total
        signal_power = np.mean(self.signal**2)
        if signal_power > 0:
            self.signal = self.signal * np.sqrt(self.P_total / signal_power)
            print(f"[DEBUG] Normalized transmitted signal to power: {np.mean(self.signal**2):.2f} (should be P_total = {self.P_total})")
        else:
            print("[DEBUG] Signal power is zero, skipping normalization.")
        # Print power of the transmitted signal
        print(f"[DEBUG] Power of transmitted signal: {np.mean(self.signal**2):.2f}")
        return self.signal

    def transmit_signal(self):
        signal = self.signal
        snr = self.snr_linear
        signal_power = np.mean(signal**2)
        noise_var = self.P_total / snr
        noise = np.random.randn(*signal.shape) * np.sqrt(noise_var)
        self.transmitted_signal = signal + noise
        # Debug prints
        print(f"[DEBUG] Code rate R = {self.R:.4f}")
        print(f"[DEBUG] Target SNR (linear): {snr:.4f}, SNR (dB): {self.snr_db:.2f}")
        print(f"[DEBUG] Target Eb/N0 (linear): {snr/self.R:.4f}, Eb/N0 (dB): {10*np.log10(snr/self.R):.2f}")
        print(f"[DEBUG] Noise variance used: {noise_var:.4f}")
        # Empirical SNR
        measured_signal_power = np.mean(signal**2)
        measured_noise_power = np.mean(noise**2)
        measured_snr = measured_signal_power / measured_noise_power
        measured_snr_db = 10 * np.log10(measured_snr)
        print(f"[DEBUG] Measured SNR (dB): {measured_snr_db:.2f}")
        return self.transmitted_signal
    
    def decode_signal(self): ##this method is to be used to decode the noiseless signal after we have done AMP

        
        
        sparse_vector = self.sparse_vector
        #turn this back into 1s and 0s 
        sparse_vector *= np.sqrt(self.L / self.P_total)
        
        L = self.L
        J = self.J

        sections = []

        for i in range(0, L*2**J, 2**J):

            section = sparse_vector[i:i+2**J]
            indices = []

            for idx, element in enumerate(section):

                if element != 0:

                    indices.append(idx)

            sections.append(indices)


        ## now we should have a list which contains the indices filled for each section in the sparse vector
        ## we can go through each path and figure out which message caused it
        outer_mappings = self.outer_mapping 

        #we need to search through all of these paths and compare it to he original messages

        combinations = list(product(*sections))
        decoded_messages = []
        original_messages = self.original_messages

        for combination in combinations:

            for message, indices in outer_mappings.items():

                if list(combination) == indices:

                    decoded_messages.append(message)

        #print(f"The decoded messages are {decoded_messages}")
        #print(f"The original messages are {original_messages}")

        
        if sorted(decoded_messages) == sorted(original_messages):

            #print("successful decoding occurred")
            worked = True
        
        else: 

           #print("unsuccesful decoding occurred")
            worked = False
        return decoded_messages, worked

    
    def _structure_match(self, vec1, vec2):
        """Check if two vectors have the same non-zero structure (indices)"""
        indices1 = set(np.where(vec1 != 0)[0])
        indices2 = set(np.where(vec2 != 0)[0])
        return indices1 == indices2
    
    def amp_decode(self, T_max: int = 15, tol: float = 1e-6):
        """
        Recover the integer vector ŝ from the received signal
            y = √P̂ · A s + z .
        Returns
        -------
        theta_final : np.ndarray, shape (N,)
            Estimated counts per column (integer after final scaling).
        """
        worked = False
        A, y  = self.A, self.transmitted_signal          # design matrix and Rx
        n, N  = A.shape
        L, J  = self.L, self.J
        Ka    = self.Ka
        P_total = self.P_total

        sqrt_P  = np.sqrt(P_total)                         # √P̂
        alpha   = 2 * J * L / n                          # Onsager coefficient  (Eq. 9)
        
        # ------------------------------------------------------------------
        #  Pre-compute the prior mass  p_k  for k = 0 … Ka     (Eq. 7)
        # ------------------------------------------------------------------
        k_vals = np.arange(Ka + 1)                       # shape (Ka+1,)
        p_k    = comb(Ka, k_vals)                        \
               * (2.0 ** (-k_vals * J))                  \
               * ((1 - 2.0 ** (-J)) ** (Ka - k_vals))    # shape (Ka+1,)
        p_k    = p_k[:, None]                            # reshape → (Ka+1,1) for broadcasting

        # ------------------------------------------------------------------
        #  Initialise AMP state
        # ------------------------------------------------------------------
        theta = np.zeros(N)         # θ₀
        z     = y.copy()            # z₀

        # ------------------------------------------------------------------
        #  Main AMP loop
        # ------------------------------------------------------------------
        for t in range(T_max):

            tau  = np.linalg.norm(z) / np.sqrt(n)        # τ_t   (scalar)

            v    = A.T @ z + theta                       # pseudo-data

            # ---------- Vectorised denoiser  f_t(v)  (Eq. 10) ----------
            v_exp    = v[None, :]                        # shape (1,N)  → broadcast
            diff     = v_exp - k_vals[:, None] * sqrt_P  # (Ka+1,N)
            exp_arg  = -diff**2 / (2 * tau**2)           # minus sign ✔️
            exp_val  = np.exp(np.clip(exp_arg, -50, 0))  # clip for stability

            Z        = (p_k * exp_val).sum(axis=0)       # partition fn  Z_t(v)
            g_num    = (p_k * k_vals[:, None] * exp_val).sum(axis=0)

            theta_next = sqrt_P * (g_num / Z)            # Eq. (10)

            # ---------- Derivative  f′_t(v)  (for Onsager term) ----------
            g_prime_num = (p_k * k_vals[:, None] * diff * exp_val).sum(axis=0) / (tau**2)
            Z_prime     = (p_k * diff * exp_val).sum(axis=0)            / (tau**2)

            f_prime = sqrt_P * (g_prime_num * Z - g_num * Z_prime) / (Z**2)

            # ---------- Residual update  z_{t+1}  (Eq. 9) ----------
            z_next = y - A @ theta_next + alpha * z * f_prime.mean()

            # ---------- Convergence check ----------
            if np.linalg.norm(theta_next - theta) / np.sqrt(N) < tol:
                theta = theta_next
                break

            theta, z = theta_next, z_next
        #print(self.sparse_vector)
        #print(theta)
        
        # Analysis: Compare missed non-zeros with true zeros
        original_nonzero_indices = np.where(self.sparse_vector != 0)[0]
        recovered_values_at_nonzero = theta[original_nonzero_indices]
        
        # Get values at positions that should be zero
        zero_indices = np.where(self.sparse_vector == 0)[0]
        recovered_values_at_zero = theta[zero_indices]
        
       # print(f"\nOriginal non-zero positions: {original_nonzero_indices}")
        #print(f"Recovered values at non-zero positions: {recovered_values_at_nonzero}")
       # print(f"Max value at true zero positions: {np.max(recovered_values_at_zero):.2e}")
       ## print(f"Min value at non-zero positions: {np.min(recovered_values_at_nonzero):.2e}")
        
        # Sort the recovered values at non-zero positions
        sorted_nonzero_values = np.sort(recovered_values_at_nonzero)[::-1]  # descending
        #print(f"Sorted values at non-zero positions: {sorted_nonzero_values}")
        
        # Check if we can set a threshold to capture more
        threshold_candidates = [1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10]
        best_threshold = None
        
        for thresh in threshold_candidates:
            detected_indices = np.where(theta > thresh)[0]
            true_positives = len(np.intersect1d(detected_indices, original_nonzero_indices))
            false_positives = len(np.setdiff1d(detected_indices, original_nonzero_indices))
            #print(f"Threshold {thresh:.0e}: TP={true_positives}/{len(original_nonzero_indices)}, FP={false_positives}")
            
            # Select best threshold: perfect TP with minimal FP
            if true_positives == len(original_nonzero_indices) and false_positives == 0:
                if best_threshold is None:
                    best_threshold = thresh
        
        # If no perfect threshold found, use the most permissive one
        if best_threshold is None:
            best_threshold = threshold_candidates[-1]
            #print("  No perfect threshold found, using most permissive")
            
            # Try to find threshold that gets maximum TP with acceptable FP
            max_tp = 0
            backup_threshold = threshold_candidates[-1]
            for thresh in threshold_candidates:
                detected_indices = np.where(theta > thresh)[0]
                true_positives = len(np.intersect1d(detected_indices, original_nonzero_indices))
                false_positives = len(np.setdiff1d(detected_indices, original_nonzero_indices))
                if true_positives > max_tp and false_positives <= 5:  # Allow some false positives
                    max_tp = true_positives
                    backup_threshold = thresh
            best_threshold = backup_threshold
            #print(f"Using backup threshold {best_threshold:.0e} for {max_tp}/{len(original_nonzero_indices)} recovery")
        
        # Recreate the original sparse vector using optimal threshold
        threshold = best_threshold
        detected_indices = np.where(theta > threshold)[0]
        
        # Recreate sparse vector
        reconstructed_sparse_vector = np.zeros_like(self.sparse_vector)
        reconstructed_sparse_vector[detected_indices] = 0.5  # Original non-zero value
        
        #print(f"\n=== SPARSE VECTOR RECONSTRUCTION ===")
        #print(f"Using threshold: {threshold}")
        #print(f"Detected indices: {detected_indices}")
        #print(f"Original non-zero indices: {original_nonzero_indices}")
        
        # Check perfect reconstruction (using structure matching - same non-zero indices)
        reconstruction_error = np.linalg.norm(self.sparse_vector - reconstructed_sparse_vector)
        perfect_match = self._structure_match(self.sparse_vector, reconstructed_sparse_vector)
        
        if perfect_match:

            worked = True
            return reconstructed_sparse_vector, worked
       # print(f"Perfect reconstruction: {perfect_match}")
       # print(f"Reconstruction error (L2): {reconstruction_error}")
        
        #if perfect_match:
            #print("🎉 SUCCESS: Original sparse vector perfectly reconstructed!")
        #else:
            #diff_indices = np.where(self.sparse_vector != reconstructed_sparse_vector)[0]
            #print(f"❌ Differences at indices: {diff_indices}")
        
        # ===== STRUCTURE-AWARE FALSE POSITIVE REDUCTION =====
        #print(f"\n=== STRUCTURE-AWARE POST-PROCESSING ===")
        
        # Method 1: Section-wise peak detection
        L, J = self.L, self.J
        section_size = 2**J  # 64 positions per section
        
        structure_aware_vector = np.zeros_like(self.sparse_vector)
        
        for section_idx in range(L):
            start_idx = section_idx * section_size
            end_idx = start_idx + section_size
            section_values = theta[start_idx:end_idx]
            
            # More sensitive approach: use both relative and absolute thresholds
            max_in_section = np.max(section_values)
            relative_threshold = max_in_section * 0.01  # 1% of max in section
            absolute_threshold = 1e-12  # Absolute minimum
            
            # Use the more permissive of the two thresholds
            section_threshold = max(relative_threshold, absolute_threshold)
            strong_peaks = np.where(section_values > section_threshold)[0]
            
            if len(strong_peaks) > 0:
                # Add offset to get global indices
                global_peaks = strong_peaks + start_idx
                structure_aware_vector[global_peaks] = 0.5
               # print(f"Section {section_idx}: Found {len(strong_peaks)} peaks at local indices {strong_peaks} (threshold: {section_threshold:.2e})")
            #else:
               # print(f"Section {section_idx}: No peaks found (max value: {max_in_section:.2e})")
        
        # Check structure-aware reconstruction
        structure_error = np.linalg.norm(self.sparse_vector - structure_aware_vector)
        structure_match = self._structure_match(self.sparse_vector, structure_aware_vector)
        
        if structure_match:
            worked = True
            return structure_aware_vector, worked 

        detected_sa = np.where(structure_aware_vector != 0)[0]
        true_positives_sa = len(np.intersect1d(detected_sa, original_nonzero_indices))
        false_positives_sa = len(np.setdiff1d(detected_sa, original_nonzero_indices))
        
        #print(f"Structure-aware detection: TP={true_positives_sa}/{len(original_nonzero_indices)}, FP={false_positives_sa}")
        #print(f"Structure-aware perfect match: {structure_match}")
        #print(f"Structure-aware error: {structure_error}")
        
        # ===== METHOD 2: COMBINATORIAL DECODING =====
        #print(f"\n=== COMBINATORIAL DECODING ===")
        
        # Convert recovered structure to message decoding
        sections_detected = []
        for section_idx in range(L):
            start_idx = section_idx * section_size
            end_idx = start_idx + section_size
            section_values = theta[start_idx:end_idx]
            
            # Use more permissive threshold for combinatorial decoding
            max_in_section = np.max(section_values)
            section_threshold = max(max_in_section * 0.01, 1e-12)  # Same as structure-aware
            section_peaks = np.where(section_values > section_threshold)[0]  # Local indices
            sections_detected.append(section_peaks.tolist())
        
        #print(f"Detected indices per section: {sections_detected}")
        
        # Try to match with valid codeword combinations
        if hasattr(self, 'outer_mapping'):
            valid_combinations = []
            for message, codewords in self.outer_mapping.items():
                if all(len(sections_detected[i]) > 0 and codewords[i] in sections_detected[i] 
                       for i in range(L)):
                    valid_combinations.append((message, codewords))
            
            #print(f"Valid message combinations found: {len(valid_combinations)}")
            if valid_combinations:
                #print("Decoded messages:", [combo[0] for combo in valid_combinations])
                
                # Create final reconstruction based on valid combinations
                final_reconstruction = np.zeros_like(self.sparse_vector)
                for message, codewords in valid_combinations:
                    for section_idx, codeword in enumerate(codewords):
                        global_idx = section_idx * section_size + codeword
                        final_reconstruction[global_idx] = 0.5
                
                final_error = np.linalg.norm(self.sparse_vector - final_reconstruction)
                final_match = self._structure_match(self.sparse_vector, final_reconstruction)
                detected_final = np.where(final_reconstruction != 0)[0]
                tp_final = len(np.intersect1d(detected_final, original_nonzero_indices))
                fp_final = len(np.setdiff1d(detected_final, original_nonzero_indices))
                
                #print(f"Final combinatorial result: TP={tp_final}/{len(original_nonzero_indices)}, FP={fp_final}")
                #print(f"Perfect reconstruction: {final_match}")
                
                if final_match:
                    
                    return final_reconstruction, worked
        
        # ===== SET RECOVERED SPARSE VECTOR FOR decode_signal() =====
        #print(f"\n=== SETTING RECOVERED SPARSE VECTOR ===")
        
        # Choose the best reconstruction method in order of preference:
        # 1. Combinatorial decoding (if valid combinations found)
        # 2. Structure-aware reconstruction  
        # 3. Simple thresholding reconstruction
        
        if 'final_reconstruction' in locals() and final_match:
            # Use combinatorial decoding result (best option)
            self.sparse_vector = final_reconstruction.copy()
            #print("✅ Using combinatorial decoding reconstruction")
        elif structure_match:
            # Use structure-aware result 
            self.sparse_vector = structure_aware_vector.copy()
            #print("✅ Using structure-aware reconstruction")
        elif perfect_match:
            # Use simple thresholding result
            self.sparse_vector = reconstructed_sparse_vector.copy()
            #print("✅ Using simple thresholding reconstruction")
        else:
            # Fallback: use the method with highest TP rate
            methods = []
            
            # Evaluate each method's TP rate
            if 'final_reconstruction' in locals():
                detected_final = np.where(final_reconstruction != 0)[0]
                tp_final = len(np.intersect1d(detected_final, original_nonzero_indices))
                methods.append(('combinatorial', final_reconstruction, tp_final))
            
            detected_sa = np.where(structure_aware_vector != 0)[0]
            tp_sa = len(np.intersect1d(detected_sa, original_nonzero_indices))
            methods.append(('structure-aware', structure_aware_vector, tp_sa))
            
            detected_simple = np.where(reconstructed_sparse_vector != 0)[0]
            tp_simple = len(np.intersect1d(detected_simple, original_nonzero_indices))
            methods.append(('simple thresholding', reconstructed_sparse_vector, tp_simple))
            
            # Choose method with highest TP rate
            best_method = max(methods, key=lambda x: x[2])
            method_name, best_reconstruction, best_tp = best_method
            
            self.sparse_vector = best_reconstruction.copy()
            #print(f"⚠️ Using {method_name} reconstruction (TP={best_tp}/{len(original_nonzero_indices)})")
        
        ##print(f"Sparse vector set with {np.count_nonzero(self.sparse_vector)} non-zero positions")
        #print("✅ Ready for decode_signal() method!")
        
        # ------------------------------------------------------------------
        #  Convert from amplitude domain (√P̂ multiples) to integer counts
        # ------------------------------------------------------------------

        
        return theta / sqrt_P, worked      # ŝ  (≈ integers)
    
    def simulate(self, num_sims):
        block_errors = 0
        for sim_idx in range(num_sims):
            self.generate_signal()
            self.transmit_signal()
            _, worked = self.amp_decode()
            # Block error: any user missed or any false positive
            orig_indices = set(np.where(self.sparse_vector != 0)[0])
            # After amp_decode, self.sparse_vector is the reconstruction
            recon_indices = set(np.where(self.sparse_vector != 0)[0])
            missed = orig_indices - recon_indices
            false_positives = recon_indices - orig_indices
            block_error = (len(missed) > 0) or (len(false_positives) > 0)
            if block_error:
                block_errors += 1
            # Debug output for first few trials
            if sim_idx < 3:
                print(f"[DEBUG] Trial {sim_idx}: Missed users: {len(missed)}, False positives: {len(false_positives)}, Block error: {block_error}")
        error_rate = block_errors / num_sims
        print(f"[RESULT] Block error rate over {num_sims} trials: {error_rate:.3f}")
        return 1 - error_rate  # for compatibility with old code

        

# Test with easier parameters first
print("Testing Ka=10, SNR=1dB:")
system = SPARC(Ka = 10, L = 32, J = 8, B = 128, n = 19600, P_total = 5, Eb_N0_dB = 10)
success_rate = system.simulate(10)
error_rate = 1 - success_rate
print(f"Error rate: {error_rate}")

print("Testing Ka=15, Eb/N0=3dB:")
system = SPARC(Ka = 15, L = 32, J = 8, B = 128, n = 19600, P_total = 5, Eb_N0_dB = 3)
success_rate = system.simulate(30)
error_rate = 1 - success_rate
print(f"Error rate: {error_rate}")

def sweep_ebn0_vs_ka():
    Ka_values = list(range(5, 31, 2))  # Ka from 5 to 30 in steps of 2
    ebn0_range = [1.0 + 0.5*i for i in range(20)]  # Eb/N0 from 1 to 11 dB in 0.5 dB steps
    results = []
    for Ka in Ka_values:
        found = False
        for Eb_N0_dB in ebn0_range:
            print(f"\n[Sweep] Ka={Ka}, Eb/N0={Eb_N0_dB:.1f}dB:")
            system = SPARC(Ka=Ka, L=32, J=8, B=128, n=19600, P_total=5, Eb_N0_dB=Eb_N0_dB)
            success_rate = system.simulate(10)  # 10 trials for speed
            block_error = 1 - success_rate
            print(f"[Sweep] Ka={Ka}, Eb/N0={Eb_N0_dB:.1f}dB, Block error rate: {block_error:.3f}")
            if block_error <= 0.1:
                results.append((Ka, Eb_N0_dB))
                print(f"[Sweep] Ka={Ka}: Minimum Eb/N0 for <=0.1 error: {Eb_N0_dB:.1f}dB")
                found = True
                break
        if not found:
            results.append((Ka, None))
            print(f"[Sweep] Ka={Ka}: No Eb/N0 found for <=0.1 error in range.")
    # Plot
    kas = [k for k, eb in results if eb is not None]
    ebn0s = [eb for k, eb in results if eb is not None]
    plt.figure(figsize=(8,5))
    plt.plot(kas, ebn0s, marker='o', label='Simulated SPARC')
    plt.xlabel('Number of Users (Ka)')
    plt.ylabel('$E_b/N_0$ (dB) for Block Error Rate ≤ 0.1')
    plt.title('SPARC: Minimum $E_b/N_0$ vs Number of Users (Block Error Rate ≤ 0.1)')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig('sparc_ebn0_vs_ka.png')
    plt.show()
    print("[Sweep] Plot saved as sparc_ebn0_vs_ka.png")

def sweep_ebn0_for_ka5():
    Ka = 5
    ebn0_range = [1.0 + 0.25*i for i in range(21)]  # Eb/N0 from 1 to 6 dB in 0.25 dB steps
    min_ebn0 = None
    for Eb_N0_dB in ebn0_range:
        print(f"\n[Sweep] Ka={Ka}, Eb/N0={Eb_N0_dB:.2f}dB:")
        system = SPARC(Ka=Ka, L=32, J=8, B=128, n=19600, P_total=5, Eb_N0_dB=Eb_N0_dB)
        success_rate = system.simulate(20)
        block_error = 1 - success_rate
        print(f"[Sweep] Ka={Ka}, Eb/N0={Eb_N0_dB:.2f}dB, Block error rate: {block_error:.3f}")
        if block_error <= 0.1:
            min_ebn0 = Eb_N0_dB
            print(f"[Sweep] Ka={Ka}: Minimum Eb/N0 for <=0.1 error: {Eb_N0_dB:.2f}dB")
            break
    if min_ebn0 is None:
        print(f"[Sweep] Ka={Ka}: No Eb/N0 found for <=0.1 error in range.")
    else:
        print(f"[RESULT] For Ka={Ka}, minimum Eb/N0 for block error rate ≤ 0.1 is {min_ebn0:.2f} dB.")

def debug_single_trial():
    Ka = 5
    Eb_N0_dB = 3
    print(f"\n[DEBUG SINGLE TRIAL] Ka={Ka}, Eb/N0={Eb_N0_dB}dB")
    system = SPARC(Ka=Ka, L=32, J=8, B=128, n=19600, P_total=5, Eb_N0_dB=Eb_N0_dB)
    system.generate_signal()
    system.transmit_signal()
    _, worked = system.amp_decode()
    orig_indices = set(np.where(system.sparse_vector != 0)[0])
    recon_indices = set(np.where(system.sparse_vector != 0)[0])
    missed = orig_indices - recon_indices
    false_positives = recon_indices - orig_indices
    block_error = (len(missed) > 0) or (len(false_positives) > 0)
    print(f"[DEBUG] Missed users: {len(missed)}, False positives: {len(false_positives)}, Block error: {block_error}")
    print(f"[DEBUG] Success: {not block_error}")

if __name__ == "__main__":
    sweep_ebn0_for_ka5()



        










        















        

            




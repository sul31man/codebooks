##this will be used for anlaysing the AMP algorithm
##first we need to generate the sparse vector signal
import numpy as np
from itertools import product 
from scipy.special import comb 
class SPARC:

    def __init__(self, Ka, L, J, B, n, P_hat, snr_db):

        self.Ka = Ka #number of users
        self.L = L #number of sections in the encoding matrix
        self.J = J #number of bits in the outer encoding scheme
        self.B = B #number of bits in the original message
        self.n = n #number of rows in the inner encoding matrix
        self.P_hat = P_hat #per-user symbol power
        self.snr_db = snr_db #channel snr in db


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
            
            vector = inner_codewords*np.sqrt(self.P_hat / self.L) #this is m_k 
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
        
        active_users = np.random.randint(0,2,self.Ka)
        self.original_messages = []

        for message, mapping in inner_coding.items():

            vector = mapping[1] #this is the m_k for each message, where A*m_k is the inner codeword, which is mapping[0]

            sparse_vector += active_users[i]*vector #superposing the signals together
            
            if active_users[i] == 1:

                self.original_messages.append(message)

            i += 1
        
        self.sparse_vector = sparse_vector

        self.signal = self.A @ self.sparse_vector #this does the calculation A@theta, which is what our tranmistted signal is 
        
        return self.signal

    def transmit_signal(self):

        #this method will use the signal and add noise to it

        signal = self.signal 

        snr = 10**(self.snr_db/10)

        signal_power = np.mean(signal**2)

        noise_var = signal_power / snr 

        noise = np.random.randn(*signal.shape) * np.sqrt(noise_var)

        self.transmitted_signal = signal + noise 

        return self.transmitted_signal 
    
    def decode_signal(self): ##this method is to be used to decode the noiseless signal after we have done AMP

        
        
        sparse_vector = self.sparse_vector
        #turn this back into 1s and 0s 
        sparse_vector *= np.sqrt(self.L / self.P_hat)
        
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

        print(f"The decoded messages are {decoded_messages}")
        print(f"The original messages are {original_messages}")

        
        if decoded_messages.sort() == original_messages.sort():

            print("successful decoding occurred")
        
        else: 

            print("unsuccesful decoding occurred")
        return decoded_messages

    def amp_decode(self, T_max=15, tol=1e-6):
        """
        Return ŝ  (integer counts per column)  estimated from self.transmitted_signal
        """

        A, y   = self.A,  self.transmitted_signal          # design matrix and noisy Rx
        n, N   = A.shape                                   # rows, cols
        L, J   = self.L, self.J
        Ka     = self.Ka
        # amplitude of ONE user's non-zero entry  (match whatever you used in inner_coding)
        gamma  = np.sqrt(self.P_hat / L)                   # √( P̂ / L )
        alpha  = N / n                                     # = 2^{J} L / n   (Onsager coef)

        # ---------- establish functions needed for this AMP method  -------------  (eq. 7)
        
        def p_k(k): #equation 7

            term = comb(Ka, k)*(2**(-k*J))*((1-2**(-J))**(Ka-k))
            
            return term 
        
        def Z(x, z): #equation 11

            tao = tao_func(z)

            term = 0
            
            for k in range(Ka+1):

                p = p_k(k)

                term2 = p*np.exp((x-k*(np.sqrt(self.P_hat)))**2 / (2*(tao**2)))

                term += term2

            
            return term 
        

        def f(x, z):##equation 10 applied to every dimension

            tao = tao_func(z)
            denominator = Z(x, z)
            term = np.zeros_like(x)
            
            for k in range(Ka+1):

                p = p_k(k)

                term2 = p*k*np.exp((x-k*(np.sqrt(self.P_hat)))**2 / (2*(tao**2)))

                term += term2

            term *= np.sqrt(self.P_hat)
            
            term /= denominator 

            return term

        
            

        
        def tao_func(z): ##the tao function

            p_o = (1 - 2**(-J))**Ka

            z = np.linalg.norm(z)

            term = z / (n*p_o) 

            return np.sqrt(term)
            
        
        ##now we need to initialise the first parts of our iterable variables 

        theta = np.zeros_like(sparse_vector) #initialise the theta variable

        z = y #initialise the z variable to be the received signal
        
        
        for _ in range(T_max):
        
           parameter1 = A.T@(z) + theta  #this will be input ot the function f we defined earlier in equation 10

           theta_next = f(parameter1, z)  #the next iteration of theta

           z_next = y - A@theta_next + (((2**J)*L)/n)*(z)*np.mean(theta_next) #the next iteration of the residal 

           
           z = z_next 
           theta = theta_next

        theta  = theta / gamma 

        print(theta)

        return theta
        













    




            

        

system = SPARC(3, 4, 6, 24, 128, 1, 5)


sparse_vector = system.generate_signal()
system.decode_signal()
system.transmit_signal()
system.amp_decode()
print(system.transmitted_signal)





##need to implement the AMP decoding algorithm 
##need to vary noise by playing around with SNR input parameters
##need to have a better analysis of one of the previous papers equations (THeorem 1)

        

        










        















        

            




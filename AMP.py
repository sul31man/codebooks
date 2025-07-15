##this will be used for anlaysing the AMP algorithm
##first we need to generate the sparse vector signal
import numpy as np

class SPARC:

    def __init__(self, Ka, L, J, B, n):

        self.Ka = Ka #number of users
        self.L = L #number of sections in the encoding matrix
        self.J = J #number of bits in the outer encoding scheme
        self.B = B #number of bits in the original message
        self.n = n #number of rows in the inner encoding matrix

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

        #this function will turn each of the messages into the L J bit outer codewords 
        self.outer_mapping = {} #initialise the mapping from messages to outer codewords
        # Generate all possible B-bit messages (2^B total)
        

        L = self.L 
        J = self.J
        
        # For each message, generate L many J-bit codewords
        for msg_idx, message in enumerate(self.messages):
            # Create L codewords for this message
            codewords = []
            for l in range(L):
                # Generate a unique J-bit codeword for this message and position l
                # Use a deterministic approach based on message and position
                seed_value = hash((message, l)) % (2**32)
                np.random.seed(seed_value)  # Deterministic but unique per (message, l)
                codeword = np.random.randint(0, 2**J) #this will direcctly produce the index value instead of the string of bits as this is easier to deal with, won't have to convert between binary and decimal. 
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

            vector = inner_codewords #this is m_k 
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
        
        
        for message, mapping in inner_coding.items():

            vector = mapping[1] #this is the m_k for each message, where A*m_k is the inner codeword, which is mapping[0]

            sparse_vector += vector #superposing the signals together
        
        self.sparse_vector = sparse_vector

        self.signal = self.A @ self.sparse_vector #this does the calculation A@theta, which is what our tranmistted signal is 
        
        return sparse_vector

    def transmit_signal(self):

        #this method will use the signal and add noise to it

        signal = self.signal 

        noise = np.random.randn(*signal.shape)

        self.transmitted_signal = signal + noise 

        return self.transmitted_signal 
    


system = SPARC(10, 2, 2, 2, 4)


sparse_vector = system.generate_signal()

print(system.inner_mapping.values())




##need to add error correcting codes - how can I go from my messages to building a scheme such that
#only ceratin message sequences cna lead to a certain superposition of messages


##need to implement the AMP decoding algorithm 
##need to vary noise by playing around with SNR input parameters
##need to have a better analysis of one of the previous papers equations (THeorem 1)

        

        










        















        

            




in these files we shall be working on the first iteration of using reinforcement learning to generate codebooks

We will be using a SAC algorithm, which will have a state determining the effectivity of the codebook, the action being the new codeword the agent adds to the codebook, where the environment then analyses the performance of the codebook with
the current amount of messages it has built. 

The environment will be a custom high performance environemnt built for performing parallel simulations extremely quickly so that we can iterate really quickly


This agent will be generating the indices for sparse codeword generation. 

hence it will spit out L indices which go from the value 0 to 2**J. 
we will use a normal distribution to 
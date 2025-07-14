These files contain scripts for generating the curve for theorem 1 in the paper https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8006984

The file named theorem1.py is the most accurate iteration thus far, including both functions q(t) and p(t). This takes a lot of time to run due to the large number of combinations required for 
the Monte Carlo simulation of the It rv. 

The file named analytical_paper_debug.py is without q(t) and hence without the MC simulation, making it much faster to run. 

Overall the latter file is around 3dB larger than expected and the former is 2dB larger than expected, however the generally increasing trend in the results they produce promises that there is good core logic still. 

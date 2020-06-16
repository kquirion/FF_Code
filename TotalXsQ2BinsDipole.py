"""
Here I plot the total cross sections 
calculated from various bins of 
Q^2 to see how much each interval of Q^2 
contributes to the total cross section 
"""
import time
from numpy import logspace,log10
from XsFunctionsBinned import XsBinned
import datetime
start_time = time.time()
print(datetime.datetime.now())
NumSigma  = 20
Elow  = -1.
Ehigh = log10(20)
NewEnu = logspace(Elow,Ehigh,NumSigma)
SIGMA = XsBinned(NewEnu,1.35)
print("--- %s Minutes Until Finish" % ((time.time() - start_time)/60.0))

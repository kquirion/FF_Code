## Here I plot the total cross sections calculated from various bins of Q^2 to see how much each interval of Q^2 contributes to the total cross section ##

from scipy.interpolate import interp1d
import os
import psutil
import time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 
from matplotlib import cm
from numpy import array,logspace,longdouble,log,log10
from XSFunctions import *
from XSFunctions_binned import *
import datetime

start_time = time.time()   
print(datetime.datetime.now())         
    
    
#############################################################################################################
#############################################################################################################
#############################################################################################################
num_SIGMA  = 20
E_low  = -1.
E_high = log10(20)
new_E_nu = logspace(E_low,E_high,num_SIGMA)

    
## Make Neutrino Cross Section ##
SIGMA = make_total_xs_binned(new_E_nu,1.35)

duration = 1  # seconds
freq = 440  # Hz
#os.system('play -nq -t alsa synth {} sine {}'.format(duration, freq))
          
print("--- %s Minutes Until Finish" % ((time.time() - start_time)/60.0))

from math import log10, floor 
from numpy import array,linspace,longdouble,where,sqrt,broadcast_to,swapaxes,log,power,nanmin,nanmax,conjugate,random,amin,amax,histogram,ravel
from math import pi
from sys import exit
import matplotlib.pyplot as plt
import time
from XSFunctions import *

######################################################################## 
## Create a function to make the less constrained kinematic variables ##
########################################################################
def make_variables(N_T,N_cos,E_nu):
    m_N = (0.9389)                                            # mass of the Nucleon
    m_mu = (0.1057)                                           # mass of Muon GeV
    p_F = (0.220)                                             # Fermi Momentum
    E_hi = sqrt(sq(m_N) + sq(p_F))                            # Upper limit of neutron energy integration    
    
    T_mu_max = E_nu + E_hi - m_mu - m_N
    T_mu_min = 0.05 
    T_mu = linspace(T_mu_min,T_mu_max,N_T,endpoint=False,dtype=longdouble)
    DELTA_T_mu = (T_mu_max-T_mu_min)/N_T
    E_mu = T_mu + m_mu
    P_mu = sqrt(sq(E_mu) - sq(m_mu))
    
    ## Restrict cos values to those satisfying Q2 > 0 ##
    cos_max = E_mu/P_mu - m_mu**2/(2.0*E_nu*P_mu)
    cos_max = where(cos_max < 1.0, cos_max, 1.0)
    #cos_max = 20.*pi/180.
    cos_mu = array([linspace(-cos_max[i],cos_max[i],2*N_cos,endpoint=False) for i in range(N_T)],dtype=longdouble)
    #cos_mu = array([linspace(-cos_max,cos_max,2*N_cos,endpoint=False) for i in range(N_T)],dtype=longdouble)
    
    DELTA_cos_mu = array([0.0  for i in range(N_T)],dtype=longdouble)
    for i in range(N_T):
        DELTA_cos_mu[i] = abs(cos_mu[i][1] - cos_mu[i][0])
    
    T_mu = broadcast_to(T_mu,(int(2*N_cos),N_T))
    T_mu = swapaxes(T_mu,0,1)
    E_mu = broadcast_to(E_mu,(int(2*N_cos),N_T))
    E_mu = swapaxes(E_mu,0,1)
    P_mu = broadcast_to(P_mu,(int(2*N_cos),N_T))
    P_mu = swapaxes(P_mu,0,1)
                   
    return T_mu,E_mu,P_mu,E_nu,cos_mu,DELTA_cos_mu,DELTA_T_mu


start_time = time.time()   
m_mu = 0.1057

E_nu = 10.0
N_T = 500
N_cos = 250
################################
## Create Kinematic Variables ##
################################
T_mu,E_mu,P_mu,E_nu,cos_mu,DELTA_cos_mu,DELTA_T_mu = make_variables(N_T,N_cos,E_nu)

Q2 = 2.0*E_mu*E_nu - 2.0*E_nu*P_mu*cos_mu - m_mu**2
Q2 = Q2.ravel()

Q2_low = amin(Q2)
Q2_high = amax(Q2)
Q2_average = sum(Q2)/(len(Q2)+1)
print Q2_average
binning = linspace(Q2_low,Q2_high,40)   
      
fig, axs = plt.subplots(1,1)
axs.set_title("Q^2 at E_nu = %s GeV" % E_nu)
axs.set_xlabel("Q^2 (GeV)")

hist = axs.hist(Q2,binning,label=("Average Q^2 is %s GeV" % Q2_average))
axs.legend()

plt.show()

fig.savefig("Desktop/Research/Axial FF/FF PDFs/Q2 Histos/Enu=%s.pdf" % E_nu)
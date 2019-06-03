import time
from scipy.integrate import quad
from scipy import stats
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import math 
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import pprint
from math import log10, floor
import pandas
from pandas import DataFrame
from scipy.optimize import curve_fit

start_time = time.time()
  
## Create a function for rounding to 4 significant figures (for readability) ##   
def round_sig(x, sig=4):
   return round(x, sig-int(floor(log10(abs(x))))-1)
   
## Define a function that creates an array of axial form factor points ##
def make_GA(xdata,M_A):
    g_A = 1.212
    return (g_A/(1+xdata/M_A**2)**2)

###########################################################################################################################################  
###########################################################################################################################################  
###########################################################################################################################################  

## START THE CROSS SECTION FITTING PORTION OF THE CODE ##

## A function for creating data points to plot for a double differential cross section given an axial mass ##    
def plot_cross_section(Array,M_A):
    E_mu = []
    P_mu = []
    E_nu = []
    
    p_F = 0.220                                             # Fermi Momentum
    A = 12                                                  # number of Nucleons  
    m_N = 0.9389                                            # mass of the Nucleon
    E_b = 0.025                                             # binding energy GeV
    m_T = A*(m_N-E_b)                                       # mass of target Nucleus
    V = (3*math.pi**2*A)/(2*p_F**3)                         # Volume of the tagret
    m_mu = 0.1057                                           # mass of Muon GeV
    mu_p = 2.793                                            # proton magnetic moment
    mu_n = -1.913                                           # neutron magnetic moment 
    M = m_N                                                 # Mass of the W Boson
    M_V2 = 0.71                                             # Vector mass parameter 
    g_A = -1.269                                            # F_A(q^2 = 0)
    M_pi = 0.1396                                           # mass of the pion
    G_F = 1.166*10**(-5)                                    # Fermi Constant
    V_ud = 0.9742                                           # Mixing element for up and down quarks
    GeV_To_Cm = 5.06773076*10**(13)                             # Conversion factor for GeV to cm
    a2 = [3.253,3.104,3.043,3.226,3.188]                    # coefficient for form factor denominator (see Arrington Table 2)
    a4 = [1.422,1.428,0.8548,1.508,1.354]
    a6 = [0.08582,0.1112,0.6806,-0.3773,0.1511]
    a8 = [0.3318,-0.006981,-0.1287,0.6109,-0.01135]
    a10 = [-0.09371,0.0003705,0.008912,-0.1853,0.0005330]
    a12 = [0.01076,-0.7063*10**(-5),0.0,0.01596,-0.9005*10**(-5)]
    E_hi = math.sqrt(m_N**2 + p_F**2)
     
        
    ## Convert the flux values to the correct order of magnitude (per proton on target)
    Flux = np.array([45.4,171,222,267,332,364,389,
        409,432,448,456,458,455,451,443,
        431,416,398,379,358,335,312,288,
        264,239,214,190,167,146,126,108,
        92,78,65.7,55.2,46.2,38.6,32.3,
        27.1,22.8,19.2,16.3,13.9,11.9,
        10.3,8.96,7.87,7,6.3,5.73,5.23,
        4.82,4.55,4.22,3.99,3.84,3.63,
        3.45,3.33,3.20] )*10**(-12)  
           
    ## Calculate the integrated flux per proton on target (POT) through a Riemann sum ##
    Total_Flux_POT = 0
    for i in range(len(Flux)):
        Total_Flux_POT = Total_Flux_POT + 0.05*Flux[i]    
    
    N_T = 18
    N_cos = 20
    N_nu = len(Flux)
    
    ## Define the weight functions needed to integrate over the flux ##
    weight = np.array([[[0.0 for k in range(N_nu)] for j in range(N_cos)] for i in range(N_T)])
    for i in range(N_T):
        for j in range(N_cos):
            for k in range(N_nu):
                weight[i][j][k] = Flux[k]/Total_Flux_POT
                
    T_mu = np.array([[[0.0 for k in range(N_nu)] for j in range(N_cos)] for i in range(N_T)])
    cos_mu = np.array([[[0.0 for k in range(N_nu)] for j in range(N_cos)] for i in range(N_T)])
    P_mu = np.array([[[0.0 for k in range(N_nu)] for j in range(N_cos)] for i in range(N_T)])
    E_nu = np.array([[[0.0 for k in range(N_nu)] for j in range(N_cos)] for i in range(N_T)])
    E_mu = np.array([[[0.0 for k in range(N_nu)] for j in range(N_cos)] for i in range(N_T)])
    
    for i in range(N_T):
        for j in range(N_cos):
            for k in range(N_nu):
                cos_mu[i][j][k] = Array[360+j]
                T_mu[i][j][k] = Array[i*20]       
    #print(T_mu)
    #print(cos_mu)
        
    ## Create the array for neutrino Energy and reconstructed neutrino energy ##
    for i in range(N_T):
        for j in range(N_cos):
            for k in range(N_nu):         
                E_nu[i][j][k] = (0.025 + 3.0*k/N_nu)
    #print(E_nu)
                     
    ## fill in the muon kinetic energy [.2,2], total energy, and 3-momentum values ##
    for i in range(N_T): 
        for j in range(N_cos):
            for k in range(N_nu):   
                E_mu[i][j][k] = T_mu[i][j][k] + m_mu 
                P_mu[i][j][k] = math.sqrt(E_mu[i][j][k]**2 - m_mu**2)  
                       
    ## Create 3D arrays of the physical quantities to be calculated ##                                                                                                                                                                                    
    q = np.array([[[0.0 for k in range(N_nu)] for j in range(N_cos)] for i in range(N_T)])           # 3-Momentum of the W Boson                                                                # Upper limit of neutron energy integration
    E_lo = np.array([[[0.0 for k in range(N_nu)] for j in range(N_cos)] for i in range(N_T)])        # Lower limit of neutron energy integration
                                                                                
    ## The following are used to define the W and a coefficients in the differential cross section ##
    c = np.array([[[0.0 for k in range(N_nu)] for j in range(N_cos)] for i in range(N_T)])
    d = np.array([[[0.0 for k in range(N_nu)] for j in range(N_cos)] for i in range(N_T)])
    
    ## Sachs form factors used to define the Pauli-Dirac form factors ##
    GEp = np.array([[[0.0 for k in range(N_nu)] for j in range(N_cos)] for i in range(N_T)])
    GEn = np.array([[[0.0 for k in range(N_nu)] for j in range(N_cos)] for i in range(N_T)])
    GMp = np.array([[[0.0 for k in range(N_nu)] for j in range(N_cos)] for i in range(N_T)])
    GMn = np.array([[[0.0 for k in range(N_nu)] for j in range(N_cos)] for i in range(N_T)])
    GEV = np.array([[[0.0 for k in range(N_nu)] for j in range(N_cos)] for i in range(N_T)])
    GMV = np.array([[[0.0 for k in range(N_nu)] for j in range(N_cos)] for i in range(N_T)])
        
    ###################################################################################################################################
    ###################################################################################################################################
    ###################################################################################################################################
        
    ## Here we repeat the calculation of many of the relevant quantities with many more neutrino energy sample points ##                    
                
    ## fill in the Q^2 = -q^2 values ##
    Q2 = 2.0*np.multiply(E_mu,E_nu) - 2.0*np.multiply(np.multiply(E_nu,P_mu),cos_mu) - m_mu**2
    
    ## fill in values of the W Boson Energy ##
    w = E_nu - E_mu
            
    ## fill in the values of the W boson energy ##
    w_eff = w - E_b 
        
    ## fill in the values for the 3-momentum of the Boson ##
    for i in range(N_T):
        for j in range(N_cos):
            for k in range(N_nu):
                q[i][j][k] = math.sqrt(Q2[i][j][k] + w[i][j][k]**2)
        
    ## fill in the values for some useful coefficients ##
    c = np.divide(-w_eff,q)                    
    d = np.divide(-(np.square(w_eff) - np.square(q)),2.0*q*m_N)  
        
    for i in range(N_T):
        for j in range(N_cos):
            for k in range(N_nu):  
                ## fill in values for the limits of integration being careful to avoid imaginary arguments ##
                ## The nested if statement ensures E_hi > E_lo ##
                alpha = ( 1.0 - c[i][j][k]**2 + d[i][j][k]**2 )
                if ((alpha > 0.0) & ((1-c[i][j][k]**2) > 0)) :
                    E_lo[i][j][k] = max((E_hi - w_eff[i][j][k]), m_N*(c[i][j][k]*d[i][j][k] + math.sqrt(alpha))/(1-c[i][j][k]**2))
                elif ((alpha < 0.0)):
                    alpha = 0.0
                    E_lo[i][j][k] = max((E_hi - w_eff[i][j][k]), m_N*(c[i][j][k]*d[i][j][k] + math.sqrt(alpha))/(1-c[i][j][k]**2))
                if E_lo[i][j][k] > E_hi:
                    E_lo[i][j][k] = E_hi
                    
    #print(np.min(E_lo))
                            
    ## AnalyticExpressions for the b elements analytically ##
    b_0 = (m_T*V)/(2.0*math.pi)*np.divide((E_hi - E_lo) + E_b*np.log(np.divide((E_hi - E_b),(E_lo - E_b))),q)
    b_1 = (m_T*V)/(2.0*math.pi*m_N)*np.divide(0.5*(E_hi**2 - np.square(E_lo)) + E_b*(E_hi - E_lo) + (E_b**2)*np.log(np.divide(E_hi - E_b,E_lo - E_b)),q)
    b_2 = (m_T*V)/(2.0*math.pi*(m_N**2))*np.divide((1/3.0)*(E_hi**3 - np.power(E_lo,3)) + E_b*(0.5*(E_hi**2 - np.square(E_lo)) + E_b*(E_hi - E_lo) + (E_b**2)*np.log(np.divide((E_hi - E_b),(E_lo - E_b)))),q)
    
    ## using the b elements as well as c and d to calculate the a elements ##
    a_1 = b_0
    a_2 = b_2 - b_0
    a_3 = np.multiply(np.square(c),b_2) + 2.0*np.multiply(np.multiply(c,d),b_1) + np.multiply(np.square(d),b_0)
    a_4 = b_2 - (2.0*E_b*b_1)/m_N + (E_b**2)/(m_N**2)*b_0
    a_5 = np.multiply(-c,b_2) + np.multiply((E_b*c/m_N - d),b_1) + np.multiply((E_b*d/m_N),b_0)
    a_6 = np.multiply(-c,b_1) - np.multiply(d,b_0)
    a_7 = b_1 - E_b*b_0/m_N
        
    ########## print time to checkpoint 3 ###############
    #print("--- %s seconds --- 3" % (time.time() - start_time))
                                    
    for i in range(N_T):
        for j in range(N_cos):
            for k in range(N_nu):                                     
                ## Fill in the Sachs Form Factors using Q2 restrictions seen in Arrington##
                if Q2[i][j][k] < 6.0:
                    GEp[i][j][k] = 1.0/(1 + a2[0]*Q2[i][j][k] + a4[0]*Q2[i][j][k]**2 + a6[0]*Q2[i][j][k]**3 + a8[0]*Q2[i][j][k]**4 + a10[0]*Q2[i][j][k]**5 + a12[0]*Q2[i][j][k]**6)
                elif Q2[i][j][k] >= 6.0:
                    GEp[i][j][k] = (mu_p/(1+a2[1]*Q2[i][j][k]+a4[1]*Q2[i][j][k]**2+a6[1]*Q2[i][j][k]**3+a8[1]*Q2[i][j][k]**4+a10[1]*Q2[i][j][k]**5+a12[1]*Q2[i][j][k]**6)) * (0.5/(1+a2[0]*6.0+a4[0]*6.0**2+a6[0]*6.0**3+a8[0]*6.0**4+a10[0]*6.0**5+a12[0]*6.0**6)) / (0.5*mu_p/(1+a2[1]*6.0+a4[1]*6.0**2+a6[1]*6.0**3+a8[1]*6.0**4+a10[1]*6.0**5+a12[1]*6.0**6))
                    
    GMp = mu_p/(1+a2[1]*Q2+a4[1]*np.square(Q2)+a6[1]*np.power(Q2,3)+a8[1]*np.power(Q2,4)+a10[1]*np.power(Q2,5)+a12[1]*np.power(Q2,6))
    GMn = mu_n/(1+a2[2]*Q2+a4[2]*np.square(Q2)+a6[2]*np.power(Q2,3)+a8[2]*np.power(Q2,4)+a10[2]*np.power(Q2,5)+a12[2]*np.power(Q2,6))  
    GEn = -mu_n*0.942*(Q2/(4*M**2)) / (1+(Q2/(4*M**2)*4.61)) * (1/(1 + (np.square(Q2)/M_V2**2)))                
                
    GEV = (GEp - GEn)
    GMV = (GMp - GMn)            
                                    
    ## Create the form factors as a function of Q^2 = -q^2 ##
    F_1 = np.divide((GEV + np.multiply(Q2/(4*M**2),GMV)),(1 + Q2/(4*M**2)))
    F_2 = np.divide((GMV - GEV),((1 + Q2/(4*M**2))))
    F_A = (g_A/np.square(1 + Q2/M_A**2))
    F_P = np.divide((2.0*m_N**2)*F_A,(M_pi**2 + Q2))
                                
    ## Use the Form Factors to Define the H Elements ##
    H_1 = 8.0*m_N**2*np.square((g_A/np.square(1 + Q2/M_A**2))) + np.multiply(2.0*Q2,np.square(F_1 + F_2) + np.square(g_A/np.square(1 + Q2/M_A**2)))
    H_2 = 8.0*m_N**2*(np.square(F_1) + np.square(g_A/np.square(1 + Q2/M_A**2))) + np.multiply(2.0*Q2,np.square(F_2))
    H_3 = np.multiply(-16.0*m_N**2*(g_A/np.square(1 + Q2/M_A**2)),(F_1 + F_2))
    H_4 = np.multiply(Q2/2.0,np.square(F_2) + 4.0*np.square(np.divide((2.0*m_N**2)*(g_A/np.square(1 + Q2/M_A**2)),(M_pi**2 + Q2)))) - 2.0*m_N**2*np.square(F_2) - 4.0*m_N**2*(np.multiply(F_1,F_2) + np.multiply(2.0*g_A/np.square(1 + Q2/M_A**2),np.divide((2.0*m_N**2)*(g_A/np.square(1 + Q2/M_A**2)),(M_pi**2 + Q2))))
    H_5 = 8.0*m_N**2*(np.square(F_1) + np.square(g_A/np.square(1 + Q2/M_A**2))) + np.multiply(2.0*Q2,np.square(F_2))
           
    ########## print time to checkpoint 4 ###############
    #print("--- %s seconds --- 4" % (time.time() - start_time))
                                                                                         
    ## Use the a and H values to determine the W values ##
    W_1 = np.multiply(a_1,8.0*m_N**2*np.square((g_A/np.square(1 + Q2/M_A**2))) + np.multiply(2.0*Q2,(np.square(F_1 + F_2) + np.square((g_A/np.square(1 + Q2/M_A**2)))))) + 0.5*np.multiply((a_2 - a_3),(8.0*m_N**2*(np.square(F_1) + np.square((g_A/np.square(1 + Q2/M_A**2)))) + np.multiply(2.0*Q2,np.square(F_2))))
    W_2 = np.multiply((a_4 + np.multiply(np.divide(np.square(w),np.square(q)),a_3) - 2*np.multiply(np.divide(w,q),a_5) + 0.5*np.multiply(1-np.divide(np.square(w),np.square(q)),a_2-a_3)),(8.0*m_N**2*(np.square(F_1) + np.square((g_A/np.square(1 + Q2/M_A**2)))) + np.multiply(2.0*Q2,np.square(F_2))))
    W_3 = (m_T/m_N)*np.multiply(a_7 - np.multiply(np.divide(w,q),a_6),np.multiply(-16.0*m_N**2*(g_A/np.square(1 + Q2/M_A**2)),(F_1 + F_2)))
    W_4 = (m_T**2/m_N**2)*(np.multiply(a_1,np.multiply((Q2/2.0),(np.square(F_2) + 4.0*np.square(np.divide((2.0*m_N**2)*(g_A/np.square(1 + Q2/M_A**2)),(M_pi**2 + Q2))))) - 2.0*m_N**2*np.square(F_2) - 4.0*m_N**2*(np.multiply(F_1,F_2) + np.multiply(2.0*(g_A/np.square(1 + Q2/M_A**2)),np.divide((2.0*m_N**2)*(g_A/np.square(1 + Q2/M_A**2)),(M_pi**2 + Q2))))) + m_N*np.divide(np.multiply(a_6,8.0*m_N**2*(np.square(F_1) + np.square((g_A/np.square(1 + Q2/M_A**2)))) + np.multiply(2.0*Q2,np.square(F_2))),q) + (m_N**2/2)*np.divide(np.multiply(3*a_3 - a_2,(8.0*m_N**2*(np.square(F_1) + np.square((g_A/np.square(1 + Q2/M_A**2)))) + np.multiply(2.0*Q2,np.square(F_2)))),np.square(q)))
    W_5 = (m_T/m_N)*np.multiply((a_7 - np.multiply(np.divide(w,q),a_6)),8.0*m_N**2*(np.square(F_1) + np.square((g_A/np.square(1 + Q2/M_A**2)))) + np.multiply(2.0*Q2,np.square(F_2))) + m_T*np.multiply((2*a_5 + np.multiply(np.divide(w,q),a_2 - 3*a_3)),np.divide((8.0*m_N**2*(np.square(F_1) + np.square((g_A/np.square(1 + Q2/M_A**2)))) + np.multiply(2.0*Q2,np.square(F_2))),q))
                                   
    ## We can now write the double differential cross section ##
    S = np.multiply((G_F**2*P_mu*V_ud**2)/(16.0*math.pi**2*m_T*(GeV_To_Cm**2)),( 2.0*np.multiply(E_mu-np.multiply(P_mu,cos_mu),W_1) + np.multiply(E_mu+np.multiply(P_mu,cos_mu),W_2) + (1/m_T)*np.multiply(np.multiply(E_mu-np.multiply(P_mu,cos_mu),E_nu+E_mu) - m_mu**2,W_3) + (m_mu**2/m_T**2)*np.multiply(E_mu-np.multiply(P_mu,cos_mu),W_4) - (m_mu**2/m_T)*W_5))

    S_Flux_Integrated = np.trapz(np.multiply(weight/(A/2),S),dx=0.05,axis=2)
    return S_Flux_Integrated
          
############################################################################################
############################################################################################
############################################################################################


## A function for fitting the double differential cross section ##
def fit_cross_section(Array,M_A):
    E_mu = []
    P_mu = []
    E_nu = []
    
    p_F = 0.220                                             # Fermi Momentum
    A = 12                                                  # number of Nucleons  
    m_N = 0.9389                                            # mass of the Nucleon
    E_b = 0.025                                             # binding energy GeV
    m_T = A*(m_N-E_b)                                       # mass of target Nucleus
    V = (3*math.pi**2*A)/(2*p_F**3)                         # Volume of the tagret
    m_mu = 0.1057                                           # mass of Muon GeV
    mu_p = 2.793                                            # proton magnetic moment
    mu_n = -1.913                                           # neutron magnetic moment 
    M = m_N                                                 # Mass of the W Boson
    M_V2 = 0.71                                             # Vector mass parameter 
    g_A = -1.269                                            # F_A(q^2 = 0)
    M_pi = 0.1396                                           # mass of the pion
    G_F = 1.166*10**(-5)                                    # Fermi Constant
    V_ud = 0.9742                                           # Mixing element for up and down quarks
    GeV_To_Cm = 5.06773076*10**(13)                             # Conversion factor for GeV to cm
    a2 = [3.253,3.104,3.043,3.226,3.188]                    # coefficient for form factor denominator (see Arrington Table 2)
    a4 = [1.422,1.428,0.8548,1.508,1.354]
    a6 = [0.08582,0.1112,0.6806,-0.3773,0.1511]
    a8 = [0.3318,-0.006981,-0.1287,0.6109,-0.01135]
    a10 = [-0.09371,0.0003705,0.008912,-0.1853,0.0005330]
    a12 = [0.01076,-0.7063*10**(-5),0.0,0.01596,-0.9005*10**(-5)]
    E_hi = math.sqrt(m_N**2 + p_F**2)
     
        
    ## Convert the flux values to the correct order of magnitude (per proton on target)
    Flux = np.array([45.4,171,222,267,332,364,389,
        409,432,448,456,458,455,451,443,
        431,416,398,379,358,335,312,288,
        264,239,214,190,167,146,126,108,
        92,78,65.7,55.2,46.2,38.6,32.3,
        27.1,22.8,19.2,16.3,13.9,11.9,
        10.3,8.96,7.87,7,6.3,5.73,5.23,
        4.82,4.55,4.22,3.99,3.84,3.63,
        3.45,3.33,3.20] )*10**(-12)  
           
    ## Calculate the integrated flux per proton on target (POT) through a Riemann sum ##
    Total_Flux_POT = 0
    for i in range(len(Flux)):
        Total_Flux_POT = Total_Flux_POT + 0.05*Flux[i]    
    
    N_T = 18
    N_cos = 20
    N_nu = len(Flux)
    
    ## Define the weight functions needed to integrate over the flux ##
    weight = np.array([[[0.0 for k in range(N_nu)] for j in range(N_cos)] for i in range(N_T)])
    for i in range(N_T):
        for j in range(N_cos):
            for k in range(N_nu):
                weight[i][j][k] = Flux[k]/Total_Flux_POT
                
    T_mu = np.array([[[0.0 for k in range(N_nu)] for j in range(N_cos)] for i in range(N_T)])
    cos_mu = np.array([[[0.0 for k in range(N_nu)] for j in range(N_cos)] for i in range(N_T)])
    P_mu = np.array([[[0.0 for k in range(N_nu)] for j in range(N_cos)] for i in range(N_T)])
    E_nu = np.array([[[0.0 for k in range(N_nu)] for j in range(N_cos)] for i in range(N_T)])
    E_mu = np.array([[[0.0 for k in range(N_nu)] for j in range(N_cos)] for i in range(N_T)])
    
    for i in range(N_T):
        for j in range(N_cos):
            for k in range(N_nu):
                cos_mu[i][j][k] = Array[360+j]
                T_mu[i][j][k] = Array[i*20]       
    #print(T_mu)
    #print(cos_mu)
        
    ## Create the array for neutrino Energy and reconstructed neutrino energy ##
    for i in range(N_T):
        for j in range(N_cos):
            for k in range(N_nu):         
                E_nu[i][j][k] = (0.025 + 3.0*k/N_nu)
    #print(E_nu)
                     
    ## fill in the muon kinetic energy [.2,2], total energy, and 3-momentum values ##
    for i in range(N_T): 
        for j in range(N_cos):
            for k in range(N_nu):   
                E_mu[i][j][k] = T_mu[i][j][k] + m_mu 
                P_mu[i][j][k] = math.sqrt(E_mu[i][j][k]**2 - m_mu**2)  
                       
    ## Create 3D arrays of the physical quantities to be calculated ##                                                                                                                                                                                    
    q = np.array([[[0.0 for k in range(N_nu)] for j in range(N_cos)] for i in range(N_T)])           # 3-Momentum of the W Boson                                                                # Upper limit of neutron energy integration
    E_lo = np.array([[[0.0 for k in range(N_nu)] for j in range(N_cos)] for i in range(N_T)])        # Lower limit of neutron energy integration
                                                                                
    ## The following are used to define the W and a coefficients in the differential cross section ##
    c = np.array([[[0.0 for k in range(N_nu)] for j in range(N_cos)] for i in range(N_T)])
    d = np.array([[[0.0 for k in range(N_nu)] for j in range(N_cos)] for i in range(N_T)])
    
    ## Sachs form factors used to define the Pauli-Dirac form factors ##
    GEp = np.array([[[0.0 for k in range(N_nu)] for j in range(N_cos)] for i in range(N_T)])
    GEn = np.array([[[0.0 for k in range(N_nu)] for j in range(N_cos)] for i in range(N_T)])
    GMp = np.array([[[0.0 for k in range(N_nu)] for j in range(N_cos)] for i in range(N_T)])
    GMn = np.array([[[0.0 for k in range(N_nu)] for j in range(N_cos)] for i in range(N_T)])
    GEV = np.array([[[0.0 for k in range(N_nu)] for j in range(N_cos)] for i in range(N_T)])
    GMV = np.array([[[0.0 for k in range(N_nu)] for j in range(N_cos)] for i in range(N_T)])
        
    ###################################################################################################################################
    ###################################################################################################################################
    ###################################################################################################################################
        
    ## Here we repeat the calculation of many of the relevant quantities with many more neutrino energy sample points ##                    
                
    ## fill in the Q^2 = -q^2 values ##
    Q2 = 2.0*np.multiply(E_mu,E_nu) - 2.0*np.multiply(np.multiply(E_nu,P_mu),cos_mu) - m_mu**2
    
    ## fill in values of the W Boson Energy ##
    w = E_nu - E_mu
            
    ## fill in the values of the W boson energy ##
    w_eff = w - E_b 
        
    ## fill in the values for the 3-momentum of the Boson ##
    for i in range(N_T):
        for j in range(N_cos):
            for k in range(N_nu):
                q[i][j][k] = math.sqrt(Q2[i][j][k] + w[i][j][k]**2)
        
    ## fill in the values for some useful coefficients ##
    c = np.divide(-w_eff,q)                    
    d = np.divide(-(np.square(w_eff) - np.square(q)),2.0*q*m_N)  
        
    for i in range(N_T):
        for j in range(N_cos):
            for k in range(N_nu):  
                ## fill in values for the limits of integration being careful to avoid imaginary arguments ##
                ## The nested if statement ensures E_hi > E_lo ##
                alpha = ( 1.0 - c[i][j][k]**2 + d[i][j][k]**2 )
                if ((alpha > 0.0) & ((1-c[i][j][k]**2) > 0)) :
                    E_lo[i][j][k] = max((E_hi - w_eff[i][j][k]), m_N*(c[i][j][k]*d[i][j][k] + math.sqrt(alpha))/(1-c[i][j][k]**2))
                elif ((alpha < 0.0)):
                    alpha = 0.0
                    E_lo[i][j][k] = max((E_hi - w_eff[i][j][k]), m_N*(c[i][j][k]*d[i][j][k] + math.sqrt(alpha))/(1-c[i][j][k]**2))
                if E_lo[i][j][k] > E_hi:
                    E_lo[i][j][k] = E_hi
                    
    #print(np.min(E_lo))
                            
    ## AnalyticExpressions for the b elements analytically ##
    b_0 = (m_T*V)/(2.0*math.pi)*np.divide((E_hi - E_lo) + E_b*np.log(np.divide((E_hi - E_b),(E_lo - E_b))),q)
    b_1 = (m_T*V)/(2.0*math.pi*m_N)*np.divide(0.5*(E_hi**2 - np.square(E_lo)) + E_b*(E_hi - E_lo) + (E_b**2)*np.log(np.divide(E_hi - E_b,E_lo - E_b)),q)
    b_2 = (m_T*V)/(2.0*math.pi*(m_N**2))*np.divide((1/3.0)*(E_hi**3 - np.power(E_lo,3)) + E_b*(0.5*(E_hi**2 - np.square(E_lo)) + E_b*(E_hi - E_lo) + (E_b**2)*np.log(np.divide((E_hi - E_b),(E_lo - E_b)))),q)
    
    ## using the b elements as well as c and d to calculate the a elements ##
    a_1 = b_0
    a_2 = b_2 - b_0
    a_3 = np.multiply(np.square(c),b_2) + 2.0*np.multiply(np.multiply(c,d),b_1) + np.multiply(np.square(d),b_0)
    a_4 = b_2 - (2.0*E_b*b_1)/m_N + (E_b**2)/(m_N**2)*b_0
    a_5 = np.multiply(-c,b_2) + np.multiply((E_b*c/m_N - d),b_1) + np.multiply((E_b*d/m_N),b_0)
    a_6 = np.multiply(-c,b_1) - np.multiply(d,b_0)
    a_7 = b_1 - E_b*b_0/m_N
                                    
    for i in range(N_T):
        for j in range(N_cos):
            for k in range(N_nu):                                     
                ## Fill in the Sachs Form Factors using Q2 restrictions seen in Arrington##
                if Q2[i][j][k] < 6.0:
                    GEp[i][j][k] = 1.0/(1 + a2[0]*Q2[i][j][k] + a4[0]*Q2[i][j][k]**2 + a6[0]*Q2[i][j][k]**3 + a8[0]*Q2[i][j][k]**4 + a10[0]*Q2[i][j][k]**5 + a12[0]*Q2[i][j][k]**6)
                elif Q2[i][j][k] >= 6.0:
                    GEp[i][j][k] = (mu_p/(1+a2[1]*Q2[i][j][k]+a4[1]*Q2[i][j][k]**2+a6[1]*Q2[i][j][k]**3+a8[1]*Q2[i][j][k]**4+a10[1]*Q2[i][j][k]**5+a12[1]*Q2[i][j][k]**6)) * (0.5/(1+a2[0]*6.0+a4[0]*6.0**2+a6[0]*6.0**3+a8[0]*6.0**4+a10[0]*6.0**5+a12[0]*6.0**6)) / (0.5*mu_p/(1+a2[1]*6.0+a4[1]*6.0**2+a6[1]*6.0**3+a8[1]*6.0**4+a10[1]*6.0**5+a12[1]*6.0**6))
                    
    GMp = mu_p/(1+a2[1]*Q2+a4[1]*np.square(Q2)+a6[1]*np.power(Q2,3)+a8[1]*np.power(Q2,4)+a10[1]*np.power(Q2,5)+a12[1]*np.power(Q2,6))
    GMn = mu_n/(1+a2[2]*Q2+a4[2]*np.square(Q2)+a6[2]*np.power(Q2,3)+a8[2]*np.power(Q2,4)+a10[2]*np.power(Q2,5)+a12[2]*np.power(Q2,6))  
    GEn = -mu_n*0.942*(Q2/(4*M**2)) / (1+(Q2/(4*M**2)*4.61)) * (1/(1 + (np.square(Q2)/M_V2**2)))                
                
    GEV = (GEp - GEn)
    GMV = (GMp - GMn)            
                                    
    ## Create the form factors as a function of Q^2 = -q^2 ##
    F_1 = np.divide((GEV + np.multiply(Q2/(4*M**2),GMV)),(1 + Q2/(4*M**2)))
    F_2 = np.divide((GMV - GEV),((1 + Q2/(4*M**2))))
    F_A = (g_A/np.square(1 + Q2/M_A**2))
    F_P = np.divide((2.0*m_N**2)*F_A,(M_pi**2 + Q2))
                                
    ## Use the Form Factors to Define the H Elements ##
    H_1 = 8.0*m_N**2*np.square((g_A/np.square(1 + Q2/M_A**2))) + np.multiply(2.0*Q2,np.square(F_1 + F_2) + np.square(g_A/np.square(1 + Q2/M_A**2)))
    H_2 = 8.0*m_N**2*(np.square(F_1) + np.square(g_A/np.square(1 + Q2/M_A**2))) + np.multiply(2.0*Q2,np.square(F_2))
    H_3 = np.multiply(-16.0*m_N**2*(g_A/np.square(1 + Q2/M_A**2)),(F_1 + F_2))
    H_4 = np.multiply(Q2/2.0,np.square(F_2) + 4.0*np.square(np.divide((2.0*m_N**2)*(g_A/np.square(1 + Q2/M_A**2)),(M_pi**2 + Q2)))) - 2.0*m_N**2*np.square(F_2) - 4.0*m_N**2*(np.multiply(F_1,F_2) + np.multiply(2.0*g_A/np.square(1 + Q2/M_A**2),np.divide((2.0*m_N**2)*(g_A/np.square(1 + Q2/M_A**2)),(M_pi**2 + Q2))))
    H_5 = 8.0*m_N**2*(np.square(F_1) + np.square(g_A/np.square(1 + Q2/M_A**2))) + np.multiply(2.0*Q2,np.square(F_2))
           
                                                                                         
    ## Use the a and H values to determine the W values ##
    W_1 = np.multiply(a_1,8.0*m_N**2*np.square(g_A/np.square(1+Q2/M_A**2))+np.multiply(2.0*Q2,np.square(F_1+F_2)+np.square(g_A/np.square(1+Q2/M_A**2))))+0.5*np.multiply(a_2-a_3,8.0*m_N**2*(np.square(F_1)+np.square((g_A/np.square(1+Q2/M_A**2))))+np.multiply(2.0*Q2,np.square(F_2)))
    W_2 = np.multiply(a_4+np.multiply(np.divide(np.square(w),np.square(q)),a_3)-2*np.multiply(np.divide(w,q),a_5)+0.5*np.multiply(1-np.divide(np.square(w),np.square(q)),a_2-a_3),8.0*m_N**2*(np.square(F_1)+np.square(g_A/np.square(1+Q2/M_A**2)))+np.multiply(2.0*Q2,np.square(F_2)))
    W_3 = (m_T/m_N)*np.multiply(a_7-np.multiply(np.divide(w,q),a_6),np.multiply(-16.0*m_N**2*(g_A/np.square(1+Q2/M_A**2)),(F_1+F_2)))
    W_4 = (m_T**2/m_N**2)*(np.multiply(a_1,np.multiply(Q2/2.0,np.square(F_2)+4.0*np.square(np.divide(2.0*m_N**2*g_A/np.square(1+Q2/M_A**2),(M_pi**2+Q2))))-2.0*m_N**2*np.square(F_2)-4.0*m_N**2*(np.multiply(F_1,F_2)+np.multiply(2.0*g_A/np.square(1+Q2/M_A**2),np.divide(2.0*m_N**2*g_A/np.square(1+Q2/M_A**2),(M_pi**2+Q2)))))+m_N*np.divide(np.multiply(a_6,8.0*m_N**2*(np.square(F_1)+np.square(g_A/np.square(1+Q2/M_A**2)))+np.multiply(2.0*Q2,np.square(F_2))),q)+m_N**2/2*np.divide(np.multiply(3*a_3-a_2,8.0*m_N**2*(np.square(F_1)+np.square(g_A/np.square(1+Q2/M_A**2)))+np.multiply(2.0*Q2,np.square(F_2))),np.square(q)))
    W_5 = (m_T/m_N)*np.multiply(a_7-np.multiply(np.divide(w,q),a_6),8.0*m_N**2*(np.square(F_1)+np.square(g_A/np.square(1+Q2/M_A**2)))+np.multiply(2.0*Q2,np.square(F_2)))+m_T*np.multiply(2*a_5+np.multiply(np.divide(w,q),a_2-3*a_3),np.divide(8.0*m_N**2*(np.square(F_1)+np.square(g_A/np.square(1+Q2/M_A**2)))+np.multiply(2.0*Q2,np.square(F_2)),q))
                                   
    #S = np.multiply((0.05*G_F**2*P_mu*V_ud**2)/(16.0*math.pi**2*m_T*(GeV_To_Cm**2)*(A/2)),2.0*np.multiply(E_mu-np.multiply(P_mu,cos_mu),np.multiply(a_1,8.0*m_N**2*np.square(g_A/np.square(1+Q2/M_A**2))+np.multiply(2.0*Q2,np.square(F_1+F_2)+np.square(g_A/np.square(1+Q2/M_A**2))))+0.5*np.multiply(a_2-a_3,8.0*m_N**2*(np.square(F_1)+np.square((g_A/np.square(1+Q2/M_A**2))))+np.multiply(2.0*Q2,np.square(F_2)))) + np.multiply(E_mu+np.multiply(P_mu,cos_mu),np.multiply(a_4+np.multiply(np.divide(np.square(w),np.square(q)),a_3)-2*np.multiply(np.divide(w,q),a_5)+0.5*np.multiply(1-np.divide(np.square(w),np.square(q)),a_2-a_3),8.0*m_N**2*(np.square(F_1)+np.square(g_A/np.square(1+Q2/M_A**2)))+np.multiply(2.0*Q2,np.square(F_2)))) + (1/m_T)*np.multiply(np.multiply(E_mu-np.multiply(P_mu,cos_mu),E_nu+E_mu) - m_mu**2,(m_T/m_N)*np.multiply(a_7-np.multiply(np.divide(w,q),a_6),np.multiply(-16.0*m_N**2*(g_A/np.square(1+Q2/M_A**2)),(F_1+F_2)))) + (m_mu**2/m_T**2)*np.multiply(E_mu-np.multiply(P_mu,cos_mu),(m_T**2/m_N**2)*(np.multiply(a_1,np.multiply(Q2/2.0,np.square(F_2)+4.0*np.square(np.divide(2.0*m_N**2*g_A/np.square(1+Q2/M_A**2),(M_pi**2+Q2))))-2.0*m_N**2*np.square(F_2)-4.0*m_N**2*(np.multiply(F_1,F_2)+np.multiply(2.0*g_A/np.square(1+Q2/M_A**2),np.divide(2.0*m_N**2*g_A/np.square(1+Q2/M_A**2),(M_pi**2+Q2)))))+m_N*np.divide(np.multiply(a_6,8.0*m_N**2*(np.square(F_1)+np.square(g_A/np.square(1+Q2/M_A**2)))+np.multiply(2.0*Q2,np.square(F_2))),q)+m_N**2/2*np.divide(np.multiply(3*a_3-a_2,8.0*m_N**2*(np.square(F_1)+np.square(g_A/np.square(1+Q2/M_A**2)))+np.multiply(2.0*Q2,np.square(F_2))),np.square(q)))) - (m_mu**2/m_T)*(m_T/m_N)*np.multiply(a_7-np.multiply(np.divide(w,q),a_6),8.0*m_N**2*(np.square(F_1)+np.square(g_A/np.square(1+Q2/M_A**2)))+np.multiply(2.0*Q2,np.square(F_2)))+m_T*np.multiply(2*a_5+np.multiply(np.divide(w,q),a_2-3*a_3),np.divide(8.0*m_N**2*(np.square(F_1)+np.square(g_A/np.square(1+Q2/M_A**2)))+np.multiply(2.0*Q2,np.square(F_2)),q))).ravel()

    ## Now we need to integrate over the neutrino flux to get the flux integrated double cross section ##
    ## do the Riemann integration as a sum. The A/2 accounts for the dfact that the cross section is per neutron ##  
    #for i in range(N_T):
       # for j in range(N_cos):
           # Int1 = 0
           # for k in range(N_nu):
             #   Int1 = Int1 + 0.05*0.5*weight[i][j][k]*S[i][j][k]/(A/2)
             #   if Int1 == 0.0:
             #       S_Flux_Integrated[i][j] = (0.0)
             #   elif T_mu[i] > 2.0:
             #       S_Flux_Integrated[i][j] = (0.0)
             #   else:
             #       S_Flux_Integrated[i][j] =  round_sig(Int1,4)

    
    ## 1d return option for fitting ##         
    return np.trapz(np.multiply(weight,np.multiply((G_F**2*P_mu*V_ud**2)/(16.0*math.pi**2*m_T*(GeV_To_Cm**2)*(A/2)),( 2.0*np.multiply(E_mu-np.multiply(P_mu,cos_mu),W_1) + np.multiply(E_mu+np.multiply(P_mu,cos_mu),W_2) + (1/m_T)*np.multiply(np.multiply(E_mu-np.multiply(P_mu,cos_mu),E_nu+E_mu) - m_mu**2,W_3) + (m_mu**2/m_T**2)*np.multiply(E_mu-np.multiply(P_mu,cos_mu),W_4) - (m_mu**2/m_T)*W_5))),dx=0.05,axis=2).ravel()
    
############################################################################################
############################################################################################
############################################################################################

## Initialize some 1D arrays ##
T_mu = []
cos_mu = []                                             # Energy range of angle between muon and lepton
E_nu_Extended = []                                      # Neutrino Energy for calculating total cross section 


## create the 2d array of miniBooNE data for the double fifferential cross section ##
Big_miniBooNE = [[289.2,348.7,418.3,497.6,600.2,692.3,778.1,557.5,891.8,919.3,1003.0,1007.0,992.3,910.2,871.9,765.6,681.9,553.6,401.9,190.0],
    [15.18,25.82,44.84,85.80,135.2,202.2,292.1,401.6,503.3,686.6,813.1,970.2,1148.0,1157.0,1279.0,1233.0,1222.0,981.1,780.6,326.5],
    [0.0,0.0,0.0,0.164,3.624,17.42,33.69,79.10,134.7,272.3,404.9,547.9,850.0,1054.0,1301.0,1495.0,1546.0,1501.0,1258.0,539.2],
    [0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.947,10.92,40.63,92.93,201.5,394.4,628.8,989.9,1289.0,1738.0,1884.0,1714.0,901.8],
    [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.071,2.176,11.63,36.51,105.0,231.0,469.1,872.2,1365.0,1847.0,2084.0,1288.0],
    [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.844,16.96,57.95,147.4,392.3,909.6,1629.0,2100.0,1633.0],
    [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,10.93,10.69,45.02,157.5,526.7,1203.0,2035.0,1857.0],
    [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,12.44,49.23,222.8,723.8,1620.0,1874.0],
    [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.012,9.241,81.65,359.8,1118.0,1803.0],
    [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.229,35.61,156.2,783.6,1636.0],
    [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,4.162,11.36,66.90,451.9,1354.0],
    [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.131,26.87,239.4,1047.0],
    [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.527,116.4,794.0],
    [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,19.50,73.07,687.9],
    [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,41.67,494.3],
    [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,36.55,372.5],
    [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,278.3],
    [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,227.4]]
    
Shape_Uncertainties = np.array([[349.8,339.7,408.8,450.2,520.8,596.9,698.0,922.7,917.7,980.4,1090.0,1351.0,1293.0,1267.0,1477.0,1380.0,1435.0,1134.0,905.0,684.3],
                                [63.32,107.6,184.4,236.6,360.7,482.6,553.3,586.4,746.9,783.6,1078.0,1246.0,1105.0,1154.0,1273.0,1372.0,1455.0,1557.0,1352.0,1071.0],
                                [0.0,0.0,0.0,31.22,34.63,57.73,135.3,215.6,337.5,515.7,695.5,1048.0,1041.0,1155.0,1365.0,1434.0,1581.0,1781.0,1754.0,1378.0],
                                [0.0,0.0,0.0,0.0,0.0,0.0,0.0,55.88,50.92,114.6,238.2,415.1,742.5,965.3,1369.0,1370.0,1648.0,1845.0,2009.0,1664.0],
                                [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,3.422,20.92,45.96,114.3,250.6,574.7,1021.0,1201.0,1791.0,1769.0,2222.0,1883.0],
                                [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,41.02,77.66,149.2,475.5,870.2,1513.0,1823.0,2334.0,2193.0],
                                [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,110.3,53.23,161.6,432.3,1068.0,1873.0,2711.0,2558.0],
                                [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,55.58,162.2,598.2,1464.0,2870.0,3037.0],
                                [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,16.32,71.88,267.2,963.8,2454.0,3390.0],
                                [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,49.10,155.1,601.6,1880.0,3320.0],
                                [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,54.01,69.28,339.6,1391.0,3037.0],
                                [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,89.01,184.1,1036.0,3110.0],
                                [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,170.1,758.7,2942.0],
                                [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,230.6,544.3,2424.0],
                                [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,505.5,2586.0],
                                [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,359.6,2653.0],
                                [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,3254.0],
                                [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,3838.0]])*10**(-42)
    

for i in range(18):
    for j in range(20):
        if Shape_Uncertainties[i][j] == 0:
            Shape_Uncertainties[i][j] = np.inf
        
## scale to size of actual data ##    
miniBooNE = np.array(Big_miniBooNE)*10**(-41)                                                                              

        
  

## fill in the cos values ranging from -1 to 1##
for i in range(0,20):
    cos_mu.append(-0.95+i/10.0)

## fill in the muon kinetic energy [.2,2], total energy, and 3-momentum values ##
for i in range(2,20):    
    T_mu.append(0.05 + i/10.0) 
    
#Array = np.vstack((T_mu,cos_mu))      

## create the cartesian product of these points 
X1,X2 = np.meshgrid(cos_mu,T_mu)
size = X1.shape
cos = X1.reshape(1,np.prod(size)).ravel()
T = X2.reshape(1,np.prod(size)).ravel()
#print(X1)
#print(cos)

#print(cos)
#print(T)

## create the input xdata for the cross section function and data to fit to##
xdata = np.array(np.vstack((T,cos)))
ydata = np.array(miniBooNE.reshape(size)) 

#print(ydata)

## do the fit ##
popt,pcov = curve_fit(fit_cross_section,xdata.ravel(),ydata.ravel(),sigma=Shape_Uncertainties.ravel(),absolute_sigma=True)

S_1d = plot_cross_section(xdata.ravel(),popt[0])
#print(S_1d)
model1 = (S_1d.ravel()+1.0)
data = (miniBooNE.ravel()+1.0)
   
#print(popt)
#print(pcov)
             
## Create the plot for the double differential cross section ##    
figax1 = plt.figure()

ax1 = figax1.gca(projection='3d')  
ax1.set_xlabel(r"$\cos\theta _ {\mu}$")
ax1.set_ylabel('$T_{mu}$')
ax1.set_zlabel(r"$\frac{d\sigma}{dT_{\mu} \, d\cos\theta_{\mu}}$")
ax1.set_title('Double Differential Cross Section with $M_A =$ %s' % popt[0])
ax1.set_xlim(1.0,-1.0)
ax1.set_ylim(0.2,2.0)
ax1.set_zlim(0.0,2.5*10**(-38))

## Plot the Cross Section ##
for i in range(len(T_mu)):
    for j in range(len(cos_mu)):        
        ys = cos_mu[j]
        xs = T_mu[i]
        zs = (S_1d[i][j])
        zs2 = miniBooNE[i][j]
            
        ax1.scatter(ys,xs,zs,color='green',marker='s')
        ax1.scatter(ys,xs,zs2,color='black',marker='s')

ax1.legend((r'RFG Model: $M_A = $ %s' % round_sig(popt[0],4),'MiniBooNE Data'),loc=(0.52,0.65))


###########################################################################################################################################  
###########################################################################################################################################  
###########################################################################################################################################  

## START THE FORM FACTOR FITTING SECTION OF THE CODE ##

## Data from the Paper 
Q2_Lattice = [0.0,0.0753,0.1477,0.2174,0.2849,0.3502,0.4135,0.5351,0.5936,0.6506,0.7064,0.7609,0.8143,0.8666,0.9683]
GA_Lattice = [1.212,1.110,1.035,0.970,0.911,0.855,0.802,0.701,0.678,0.636,0.588,0.573,0.520,0.533,0.096]
GA_error = [0.033,0.020,0.014,0.018,0.020,0.018,0.020,0.025,0.023,0.028,0.028,0.050,0.030,0.037,1.665]
GA_systematic = [0.022,0.016,0.017,0.010,0.005,0.009,0.009,0.021,0.018,0.053,0.065,0.045,0.073,0.052,0.161]

total_error = []
## add the statistical and systematic uncertainties together to get a total uncertainty ##
for i in range(len(GA_error)):
    total_error.append(math.sqrt(GA_error[i]**2 + GA_systematic[i]**2))

Q2_mod = [0.1477,0.2174,0.2849,0.3502,0.4135,0.5351,0.5936,0.6506,0.7064,0.7609,0.8143,0.8666,0.9683]
GA_mod = [1.035,0.970,0.911,0.855,0.802,0.701,0.678,0.636,0.588,0.573,0.520,0.533,0.096]
GA_error_mod = [0.014,0.018,0.020,0.018,0.020,0.025,0.023,0.028,0.028,0.050,0.030,0.037,1.665]

popt,pcov = curve_fit(make_GA,Q2_Lattice,GA_Lattice,sigma=GA_error,absolute_sigma=True)
sigma = math.sqrt(pcov)

#print(popt)
#print(math.sqrt(pcov[0][0]))

FA = make_GA(Q2_Lattice,popt[0],)

X2,p_val = stats.chisquare(FA,GA_Lattice)
print('Chi Squared Value: %s' % X2)

X2_XS,pvalXS = stats.chisquare(model1,data)
print('Chi-Squared for DDXS: %s' % X2_XS) 

X2_total = X2_XS+X2
print('Total Chi-Squared: %s' %X2_total)


fitplot = plt.figure()
fit_graph = fitplot.gca()
fit_graph.set_xlabel(r'$Q^2$')
fit_graph.set_ylabel(r'$G^{u-d}_A(Q^2)$')
fit_graph.set_title(r'Fit for $G^{u-d}_A$ with $M_A$ = %s ' % popt[0])
fit_graph.errorbar(Q2_Lattice,GA_Lattice,yerr=GA_error,fmt='o',color='red',label='Lattice Data')
fit_graph.scatter(Q2_Lattice,FA,color='blue',label='Fit Line')
fit_graph.legend()

###########################################################################################################################################  
###########################################################################################################################################  
###########################################################################################################################################          
                
                                
plt.show()   
        

###########################################################################################################################################  
###########################################################################################################################################  
###########################################################################################################################################  




print("--- %s seconds ---" % (time.time() - start_time))
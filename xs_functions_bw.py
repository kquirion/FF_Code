## XS_functions_binned  ##
## Here we make the  functions for calculating cross sections with Breit Wigner FF ##

from math import log10, floor
from numpy import array,linspace,longdouble,where,sqrt,broadcast_to,swapaxes,log,power,nanmin,nanmax,conjugate,sum,maximum,minimum,empty,meshgrid,arccos,amin,amax,exp,zeros,logspace,log10
from math import pi
from scipy.integrate import quad
from sys import exit
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from XSFunctions import make_a_elements,calc_cross_section
from misc_fns import *
from variable_fns import *


#########################
## Make FFs for BW fit ##
#########################
def make_form_factors_BW(Q2,GAMMA,M_A):
    m_N = 0.9389                                            # mass of the Nucleon
    mu_p = 2.793                                            # proton magnetic moment
    mu_n = -1.913                                           # neutron magnetic moment
    a2 = [3.253,3.104,3.043,3.226,3.188]                    # coefficient for form factor denominator (see Arrington Table 2)
    a4 = [1.422,1.428,0.8548,1.508,1.354]
    a6 = [0.08582,0.1112,0.6806,-0.3773,0.1511]
    a8 = [0.3318,-0.006981,-0.1287,0.6109,-0.01135]
    a10 = [-0.09371,0.0003705,0.008912,-0.1853,0.0005330]
    a12 = [0.01076,-0.7063*10**(-5),0.0,0.01596,-0.9005*10**(-5)]
    M_V2 = 0.71                                             # Vector mass parameter
    g_A = -1.267
    #g_A = -1.2723                                            # F_A(q^2 = 0)
    M_pi = 0.1396                                           # mass of the pion
    M_ro = 0.775

    GEp = where(Q2 < 6.0,1.0/(1 + a2[0]*Q2 + a4[0]*sq(Q2) + a6[0]*power(Q2,3) + a8[0]*power(Q2,4) + a10[0]*power(Q2,5) + a12[0]*power(Q2,6)),(mu_p/(1+a2[1]*Q2+a4[1]*sq(Q2)+a6[1]*power(Q2,3)+a8[1]*power(Q2,4)+a10[1]*power(Q2,5)+a12[1]*power(Q2,6))) * (0.5/(1+a2[0]*6.0+a4[0]*6.0**2+a6[0]*6.0**3+a8[0]*6.0**4+a10[0]*6.0**5+a12[0]*6.0**6)) / (0.5*mu_p/(1+a2[1]*6.0+a4[1]*6.0**2+a6[1]*6.0**3+a8[1]*6.0**4+a10[1]*6.0**5+a12[1]*6.0**6)))
    GMp = mu_p/(1.+a2[1]*Q2+a4[1]*sq(Q2)+a6[1]*power(Q2,3)+a8[1]*power(Q2,4)+a10[1]*power(Q2,5)+a12[1]*power(Q2,6))
    GMn = mu_n/(1.+a2[2]*Q2+a4[2]*sq(Q2)+a6[2]*power(Q2,3)+a8[2]*power(Q2,4)+a10[2]*power(Q2,5)+a12[2]*power(Q2,6))
    GEn = -mu_n*0.942*(Q2/(4.*sq(m_N))) / (1.+(Q2/(4*m_N**2)*4.61)) * (1./(1. + (sq(Q2)/M_V2**2)))
    GEV = (GEp - GEn)
    GMV = (GMp - GMn)

    g_1 = where( Q2 < sq(M_ro+M_pi), 4.1*power(Q2-9*sq(M_pi),3)*(1-3.3*(Q2-9*sq(M_pi))+5.8*sq(Q2-9*sq(M_pi))), Q2*(1.623+10.38/Q2-9.32/sq(Q2)+0.65/power(Q2,3)) )
    g_1 = where( Q2 > sq(3.*M_pi), g_1, 0.)
    if(sq(3.*M_pi) <  M_A < sq(M_ro+M_pi) ):
        g_MA =  4.1*power(M_A-9*sq(M_pi),3)*(1-3.3*(M_A-9*sq(M_pi))+5.8*sq(M_A-9*sq(M_pi)))
    elif( M_A > sq(M_ro + M_pi)):
        g_MA = M_A*(1.623+10.38/M_A-9.32/sq(M_A)+0.65/power(M_A,3))
    ## Create the form factors as a function of Q^2 = -q^2 ##
    F_1 = (GEV + (Q2/(4.0*m_N**2)*GMV))/(1. + Q2/(4.*sq(m_N)))
    F_2 = (GMV - GEV)/((1. + Q2/(4*m_N**2)))
    F_A = g_A*sq(M_A)/(sq(M_A) + Q2 - (1j*M_A*GAMMA*g_1/g_MA))
    F_P = (2.0*sq(m_N))*F_A/(M_pi**2 + Q2)

    return F_1,F_2,F_A,F_P,M_A

#################################
## make ddxs for fitting BW FF ##
#################################
def make_double_diff_BW(E_mu,E_nu,P_mu,cos_mu,GAMMA,M_A,opt):
    ## parameters ##
    A = 12                                                  # number of Nucleons
    m_N = 0.9389                                            # mass of the Nucleon
    E_b = 0.034                                             # binding energy GeV
    m_T = A*(m_N-E_b)                                       # mass of target Nucleus
    m_mu = 0.1057                                           # mass of Muon GeV
    V_ud = 0.9742                                           # Mixing element for up and down quarks
    GeV_To_Cm = 5.06773076*10**(13)                         # Conversion factor for GeV to cm
    G_F = 1.166*10**(-5)                                    # Fermi Constant

    ## fill in the Q^2 = -q^2 values ##
    Q2 = 2.0*E_mu*E_nu - 2.0*E_nu*P_mu*cos_mu - m_mu**2
    ## fill in values of the W Boson Energy ##
    w = E_nu - E_mu
    ## fill in the values of the W boson energy ##
    w_eff = w - E_b
    ## fill in the values for the 3-momentum of the Boson ##
    q = sqrt(Q2 + sq(w))

    ## calculate the a elements ##
    a_1,a_2,a_3,a_4,a_5,a_6,a_7 = make_a_elements(Q2,q,w,w_eff)
    ## calculate the form factors ##
    F_1,F_2,F_A,F_P,M_A = make_form_factors_BW(Q2,GAMMA,M_A)
    ## Use the Form Factors to Define the H Elements ##
    H_1 = 8.0*sq(m_N)*sq(F_A) + 2.0*Q2*(sq(F_1 + F_2) + sq(F_A))
    H_2 = 8.0*m_N**2*(sq(F_1) + sq(F_A)) + 2.0*Q2*sq(F_2)
    H_3 = -16.0*sq(m_N)*F_A*(F_1 + F_2)
    H_4 = Q2/2.0*(sq(F_2) + 4.0*sq(F_P)) - 2.0*sq(m_N)*sq(F_2) - 4.0*sq(m_N)*(F_1*F_2 + 2.0*F_A*F_P)
    H_5 = 8.0*sq(m_N)*(sq(F_1) + sq(F_A)) + 2.0*Q2*sq(F_2)
    ## Use the a and H values to determine the W values ##
    W_1 = a_1*H_1 + 0.5*(a_2 - a_3)*H_2
    W_2 = (a_4 + (sq(w)/sq(q))*a_3 - 2.0*(w/q)*a_5 + 0.5*((1-(sq(w)/sq(q)))*(a_2-a_3)))*H_2
    W_3 = (m_T/m_N)*(a_7 - (w/q)*a_6)*H_3
    W_4 = (sq(m_T)/sq(m_N))*(a_1*H_4 + m_N*(a_6*H_5)/q + (sq(m_N)/2.0)*((3.0*a_3 - a_2)*H_2)/sq(q))
    W_5 = (m_T/m_N)*(a_7 - (w/q)*a_6)*H_5 + m_T*(2.0*a_5 + (w/q)*(a_2 - 3.0*a_3))*(H_2/q)

    if opt == 1:
        double_diff = (sq(G_F)*P_mu*V_ud**2)/(16.0*sq(pi)*m_T*(GeV_To_Cm**2))*( 2.0*(E_mu-P_mu*cos_mu)*W_1 + (E_mu+P_mu*cos_mu)*W_2 + (1/m_T)*((E_mu-P_mu*cos_mu)*(E_nu+E_mu) - sq(m_mu))*W_3 + sq(m_mu/m_T)*(E_mu-P_mu*cos_mu)*W_4 - (sq(m_mu)/m_T)*W_5)
    elif opt == 2:
        double_diff = (sq(G_F)*P_mu*V_ud**2)/(16.0*sq(pi)*m_T*(GeV_To_Cm**2))*( 2.0*(E_mu-P_mu*cos_mu)*W_1 + (E_mu+P_mu*cos_mu)*W_2 - (1/m_T)*((E_mu-P_mu*cos_mu)*(E_nu+E_mu) - sq(m_mu))*W_3 + sq(m_mu/m_T)*(E_mu-P_mu*cos_mu)*W_4 - (sq(m_mu)/m_T)*W_5)
    else:
        print("Please Enter 1 for neutrino or 2 for antineutrino in the last argument of make_double_diff_BW")
        exit()
    double_diff = double_diff
    return double_diff.real,M_A


###############################
## function to do the BW fit ##
###############################
def make_total_xs_BW(E_nu,GAMMA,M_A):

    E_nu_array = E_nu
    N = len(E_nu_array)
    SIGMA = array([0.0 for i in range(N)])
    for m  in range(N):
        N_cos = int(E_nu_array[m]+1)*500
        N_T = 100+9*int(E_nu_array[m])
        bin_size = int(2*N_cos/100)
        num_bins = int(2*N_cos/bin_size)
        cos_bin =array([[0.0 for j in range(bin_size)] for i in range(N_T)])
        ################################
        ## Create Kinematic Variables ##
        ################################
        T_mu,E_mu,P_mu,E_nu,cos_mu,DELTA_cos_mu,DELTA_T_mu = make_variables(N_T,N_cos,E_nu_array[m])

        for l in range(num_bins):
            for i in range(N_T):
                for j in range(bin_size):
                    cos_bin[i][j] = cos_mu[i][j+l*bin_size]

            #####################################################################
            ## Create RFG Variables #############################################
            ## For the last entry, put 1 for neutrinos, or 2 for antineutrinos ##
            #####################################################################
            #double_diff = make_double_diff_BW(E_mu,E_nu,P_mu,cos_bin,GAMMA,M_A,2)
            double_diff,M_A = make_double_diff_BW(E_mu,E_nu,P_mu,cos_bin,GAMMA,M_A,1)

            #############################
            ## Calculate Cross Section ##
            #############################
            SIGMA_Temp,new_E_nu = calc_cross_section(E_mu,P_mu,cos_bin,E_nu,N_T,int(N_cos/100),DELTA_cos_mu,DELTA_T_mu,double_diff)

            #################################
            ## Add to total value of SIGMA ##
            #################################
            SIGMA[m] = SIGMA[m]+SIGMA_Temp

    return SIGMA

## This file contains many functions for running cross section calculations ##
from math import log10, floor
from numpy import array,linspace,longdouble,where,sqrt,broadcast_to,swapaxes,log,power,nanmin,nanmax,conjugate,sum,maximum,minimum,empty,meshgrid,arccos,amin,amax,exp,zeros,logspace,log10,cos
from math import pi
from scipy.integrate import quad
from sys import exit
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from misc_fns import *
from variable_fns import *
from XSFunctions import *

#############################
## Make FFs for dipole fit ##
#############################
def make_form_factors_sergi(Q2,L):
    m_N = 0.9389                                            # mass of the Nucleon
    mu_p = 2.793                                            # proton magnetic moment
    mu_n = -1.913                                           # neutron magnetic moment
    a2 = [3.253,3.104,3.043,3.226,3.188]                    # coefficient for form factor denominator (see Arrington Table 2)
    a4 = [1.422,1.428,0.8548,1.508,1.354]
    a6 = [0.08582,0.1112,0.6806,-0.3773,0.1511]
    a8 = [0.3318,-0.006981,-0.1287,0.6109,-0.01135]
    a10 = [-0.09371,0.0003705,0.008912,-0.1853,0.0005330]
    a12 = [0.01076,-0.7063*10**(-5),0.0,0.01596,-0.9005*10**(-5)]
    M = m_N                                                 # Mass of the W Boson
    M_V2 = 0.71                                             # Vector mass parameter
    M_V_nuance  = 0.84
    #g_A = -1.269
    g_A = -1.2723                                            # F_A(q^2 = 0)
    #g_A = 2.3723                                           # fake parameter for single xs
    M_pi = 0.1396                                           # mass of the pion
    a = 0.942
    b = 4.61
    tau = Q2/4./sq(m_N)

    #G_D = 1./(1+Q2/sq(M_V_nuance))**2
    #GEn = -mu_n*(a*tau)/(1.+b*tau)*G_D

    GEp = where(Q2 < 6.0,1.0/(1 + a2[0]*Q2 + a4[0]*sq(Q2) + a6[0]*power(Q2,3) + a8[0]*power(Q2,4) + a10[0]*power(Q2,5) + a12[0]*power(Q2,6)),(mu_p/(1+a2[1]*Q2+a4[1]*sq(Q2)+a6[1]*power(Q2,3)+a8[1]*power(Q2,4)+a10[1]*power(Q2,5)+a12[1]*power(Q2,6))) * (0.5/(1+a2[0]*6.0+a4[0]*6.0**2+a6[0]*6.0**3+a8[0]*6.0**4+a10[0]*6.0**5+a12[0]*6.0**6)) / (0.5*mu_p/(1+a2[1]*6.0+a4[1]*6.0**2+a6[1]*6.0**3+a8[1]*6.0**4+a10[1]*6.0**5+a12[1]*6.0**6)))
    GMp = mu_p/(1+a2[1]*Q2+a4[1]*sq(Q2)+a6[1]*power(Q2,3)+a8[1]*power(Q2,4)+a10[1]*power(Q2,5)+a12[1]*power(Q2,6))
    GMn = mu_n/(1+a2[2]*Q2+a4[2]*sq(Q2)+a6[2]*power(Q2,3)+a8[2]*power(Q2,4)+a10[2]*power(Q2,5)+a12[2]*power(Q2,6))
    GEn = -mu_n*0.942*(Q2/(4*sq(M))) / (1.+(Q2/(4*M**2)*4.61)) * (1./(1. + (sq(Q2)/M_V_nuance**2)))
    GEV = (GEp - GEn)
    GMV = (GMp - GMn)
    ## Create the form factors as a function of Q^2 = -q^2 ##
    F_1 = (GEV + (Q2/(4.*M**2)*GMV))/(1. + Q2/(4.*sq(M)))
    F_2 = (GMV - GEV)/((1. + Q2/(4.*M**2)))
    
    M_A = 1.05
    F_A = g_A / (1. + Q2/sq(M_A)) / (1.+Q2/sq(L))
    F_P = 2.0*sq(m_N)*F_A/(M_pi**2 + Q2)
    #F_P = 2.*sq(m_N)/(-Q2)*(g_A/(1+Q2/sq(M_pi)) - F_A)       # from 1972 paper

    return F_1,F_2,F_A,F_P

##############################################
## Create double differential cross section ##
### opt = 1 neutrino, opt = 2 antineutrino ###
##############################################
def make_double_diff_sergi(L):
    ## parameters ##
    A = 12.
    m_N = 0.9389                                            # mass of the Nucleon
    E_b = 0.034                                             # binding energy GeV
    m_T = A*(m_N-E_b)                                       # mass of target Nucleus
    m_mu = 0.1057                                           # mass of Muon GeV
    V_ud = 0.9742                                           # Mixing element for up and down quarks
    GeV_To_Cm = 5.06773076*10**(13)                         # Conversion factor for GeV to cm
    G_F = 1.166*10**(-5)                                    # Fermi Constant
    
    ## Create an array of the neutrino flux ##
    Flux = array([45.4,171,222,267,332,364,389,409,432,448,456,458,455,451,443,
        431,416,398,379,358,335,312,288,264,239,214,190,167,146,126,108,
        92,78,65.7,55.2,46.2,38.6,32.3,27.1,22.8,19.2,16.3,13.9,11.9,
        10.3,8.96,7.87,7,6.3,5.73,5.23,4.82,4.55,4.22,3.99,3.84,3.63,
        3.45,3.33,3.20])*10**(-12)
        
    T_mu_1d = linspace(0.25,1.95,18,endpoint=True)
    cos_mu_1d = linspace(-.95,.95,20,endpoint=True)
    E_nu_1d = linspace(0.05, (len(Flux))/20.0,len(Flux),endpoint=True)  
    
    NT = len(T_mu_1d)
    NC = len(cos_mu_1d)

    ## fill in the Q^2 = -q^2 values ##
    E_nu = linspace(0., 3.,len(Flux),endpoint=True)
    Func = interp1d(E_nu,Flux,kind='cubic')

    num_flux = 500
    E_nu_new = linspace(0.001,3.,num_flux,endpoint=True)
    Flux_new = Func(E_nu_new)

    Total_Flux = 0
    for i in range(len(Flux_new)):
        Total_Flux = Total_Flux + 3./num_flux*(Flux_new[i])

    ## Define the weight functions needed to integrate over the flux ##
    weight = []
    for i in range(len(Flux_new)):
        weight.append( (3./num_flux)*(Flux_new[i]/Total_Flux)/(A/2))

    weight = broadcast_to(weight,(NT,NC,num_flux))
    E_nu_new = broadcast_to(E_nu_new,(NT,NC,num_flux))

    T_mu = broadcast_to(T_mu_1d,(num_flux,NC,NT))
    T_mu = swapaxes(T_mu,0,2)

    cos_mu = broadcast_to(cos_mu_1d,(NT,num_flux,NC))
    cos_mu = swapaxes(cos_mu,1,2)

    E_mu = T_mu + m_mu
    P_mu = sqrt(sq(E_mu) - sq(m_mu))
    
    ## fill in the Q^2 = -q^2 values ##
    Q2 = 2.0*E_mu*E_nu_new - 2.0*E_nu_new*P_mu*cos_mu - m_mu**2
    ## fill in values of the W Boson Energy ##
    w = E_nu_new - E_mu
    ## fill in the values of the W boson energy ##
    w_eff = w - E_b
    ## fill in the values for the 3-momentum of the Boson ##
    q = sqrt(Q2 + sq(w))
    ## calculate the a elements ##
    a_1,a_2,a_3,a_4,a_5,a_6,a_7 = make_a_elements(Q2,q,w,w_eff)
    ## calculate the form factors ##
    F_1,F_2,F_A,F_P = make_form_factors_sergi(Q2,L)

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

    double_diff = (sq(G_F)*P_mu*V_ud**2)/(16.0*sq(pi)*m_T*(GeV_To_Cm**2))*( 2.0*(E_mu-P_mu*cos_mu)*W_1 + (E_mu+P_mu*cos_mu)*W_2 + (1/m_T)*((E_mu-P_mu*cos_mu)*(E_nu_new+E_mu) - sq(m_mu))*W_3 + sq(m_mu/m_T)*(E_mu-P_mu*cos_mu)*W_4 - (sq(m_mu)/m_T)*W_5)
    double_diff = where(Q2 > 30., 0., double_diff)
    #double_diff = where(cos_mu < cos(20.*pi/180), 0., double_diff)
    double_diff = weight_sum_3d(double_diff.real,weight)
    return double_diff

##############################################
## Create double differential cross section ##
### opt = 1 neutrino, opt = 2 antineutrino ###
##############################################
def make_double_diff_sergi_2(E_mu,E_nu,P_mu,cos_mu,L):
    ## parameters ##
    A = 12.
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
    M_A = 1.35
    F_1,F_2,F_A,F_P = make_form_factors_sergi(Q2,L)

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

    double_diff = (sq(G_F)*P_mu*V_ud**2)/(16.0*sq(pi)*m_T*(GeV_To_Cm**2))*( 2.0*(E_mu-P_mu*cos_mu)*W_1 + (E_mu+P_mu*cos_mu)*W_2 + (1/m_T)*((E_mu-P_mu*cos_mu)*(E_nu+E_mu) - sq(m_mu))*W_3 + sq(m_mu/m_T)*(E_mu-P_mu*cos_mu)*W_4 - (sq(m_mu)/m_T)*W_5)
    double_diff = where(Q2 > 30., 0., double_diff)
    #double_diff = where(cos_mu < cos(20.*pi/180), 0., double_diff)
    return double_diff

###################################
## function to do the dipole fit ##
###################################
def make_total_xs_sergi(E_nu,L):

    E_nu_array = E_nu
    N = len(E_nu_array)
    SIGMA = array([0.0 for i in range(N)])
    for m  in range(N):
        print "Starting Calculation for E_nu = %s out of E_nu = %s" % (round_sig(E_nu_array[m]),round_sig(nanmax(E_nu_array)))
        N_cos = 400 + int(E_nu_array[m]+1)*1000
        N_T = 150 + 30*int(E_nu_array[m])
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
            double_diff = make_double_diff_sergi_2(E_mu,E_nu,P_mu,cos_bin,L)

            ## apply the angle cut of Minerva ##
            #double_diff = where((arccos(cos_bin)*180/pi <= 20) & (arccos(cos_bin)*180/pi >= -20), double_diff, 0.)

            #############################
            ## Calculate Cross Section ##
            #############################
            SIGMA_Temp,new_E_nu = calc_cross_section(E_nu,N_T,int(N_cos/100),DELTA_cos_mu,DELTA_T_mu,double_diff)

            #################################
            ## Add to total value of SIGMA ##
            #################################
            SIGMA[m] = SIGMA[m]+SIGMA_Temp

    print(SIGMA)
    return SIGMA

## file  containing functions for creating the  single  pole XS  ##

from math import log10, floor
from numpy import array,linspace,longdouble,where,sqrt,broadcast_to,swapaxes,log,power,nanmin,nanmax,conjugate,sum,maximum,minimum,empty,meshgrid,arccos,amin,amax,exp,zeros,logspace,log10,cos
from math import pi
from scipy.integrate import quad
from sys import exit
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from XSFunctions import make_a_elements,make_form_factors_dipole,calc_cross_section
from misc_fns import *
from variable_fns import *

## Interpolate the flux function that we are  integrating over to calculate a better ddxs  ##
def flux_interpolate_monopole(M_A):

    m_mu = .1057
    num_flux = 500
    Flux_FHC = array([2.57,6.53,17.,25.1,33.1,40.7,42.8,34.2,20.4,11.1,6.79,4.87,3.95,3.34,2.91,2.55,2.29,2.05,1.85,1.7,1.54,1.41,1.28,1.18,1.07,
        .989,.906,.842,.761,.695,.619,.579,.532,.476,.44,.403,.371,.34,.317,.291])*3.34*10**(14)
    Flux_RHC = array([1.26,1.69,1.78,1.88,1.90,1.96,1.9,1.82,1.73,1.65,1.64,1.70,1.75,1.80,1.76,1.73,1.65,1.57,1.47,1.37,1.28,1.17,1.08,.998,.919,
        .832,.76,.677,.643,.574,.535,.479,.445,.397,.336,.33,.311,.285,.264,.239])
    Flux_minerva = Flux_FHC + Flux_RHC

    ## Lower edges of the bins ##
    p_T_1D_low = array([0.,.075,.15,.25,.325,.4,.475,.55,.7,.85,1.,1.25,1.5])
    p_P_1D_low = array([1.5,2.,2.5,3.,3.5,4.,4.5,5.,6.,8.,10.,15.])

    ## higher edges of the bins ##
    p_T_1D_high = array([.075,.15,.25,.325,.4,.475,.55,.7,.85,1.,1.25,1.5,2.5])
    p_P_1D_high = array([2.,2.5,3.,3.5,4.,4.5,5.,6.,8.,10.,15.,20.])

    ## middle of each bin ##
    p_T_1D = (p_T_1D_low + p_T_1D_high)/2.
    p_P_1D = (p_P_1D_low + p_P_1D_high)/2.

    ## define the  flux for  each case ##
    Flux = Flux_minerva

    E_nu_Flux = linspace(0.,20.,len(Flux))
    Func = interp1d(E_nu_Flux,Flux,kind='cubic')
    E_nu_new = linspace(0.,20.,num_flux)
    Flux_new = Func(E_nu_new)

    Total_Flux = 0
    for i in range(len(Flux_new)):
        Total_Flux = Total_Flux + 20./num_flux*(Flux_new[i])

    ## Define the weight functions needed to integrate over the flux ##
    weight = []
    for i in range(len(Flux_new)):
        weight.append( (20./num_flux)*Flux_new[i]/Total_Flux)

    ## define the kinematic inputs for each case ##
    p_P_2D,p_T_2D = meshgrid(p_P_1D,p_T_1D,indexing='ij')
    cos_mu_2D = p_P_2D/sqrt(sq(p_P_2D) + sq(p_T_2D))
    T_mu_2D = sqrt(sq(p_P_2D) + sq(p_T_2D) + sq(m_mu)) - m_mu
    Jac = p_T_2D/(T_mu_2D+m_mu)/sqrt(sq(p_P_2D) + sq(p_T_2D))

    p_P_3D,p_T_3D,E_nu_3D = meshgrid(p_P_1D,p_T_1D,E_nu_new,indexing = 'ij')
    T_mu_3D = sqrt(sq(p_P_3D) + sq(p_T_3D) + sq(m_mu)) - m_mu
    cos_mu_3D = p_P_3D/sqrt(sq(p_T_3D) + sq(p_P_3D))
    E_mu_3D = T_mu_3D + m_mu
    P_mu_3D = sqrt(sq(p_T_3D) + sq(p_P_3D))

    weight = broadcast_to(weight,(len(p_P_1D),len(p_T_1D),len(Flux_new)))

    ######################################################
    ## make double diff from fit to total cross section ##
    ######################################################
    double_diff,M_A = make_double_diff_monopole(E_mu_3D,E_nu_3D,P_mu_3D,cos_mu_3D,M_A,1)
    double_diff = weight_sum_3d(double_diff.real,weight)/12.
    double_diff = where(cos_mu_2D < cos(20.*pi/180), 0., double_diff)
    double_diff  = double_diff*Jac

    return double_diff

#############################
## Make FFs for dipole fit ##
#############################
def make_form_factors_monopole(Q2,M_A):
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

    G_D = 1./(1+Q2/sq(M_V_nuance))**2
    GEn = -mu_n*(a*tau)/(1.+b*tau)*G_D

    GEp = where(Q2 < 6.0,1.0/(1 + a2[0]*Q2 + a4[0]*sq(Q2) + a6[0]*power(Q2,3) + a8[0]*power(Q2,4) + a10[0]*power(Q2,5) + a12[0]*power(Q2,6)),(mu_p/(1+a2[1]*Q2+a4[1]*sq(Q2)+a6[1]*power(Q2,3)+a8[1]*power(Q2,4)+a10[1]*power(Q2,5)+a12[1]*power(Q2,6))) * (0.5/(1+a2[0]*6.0+a4[0]*6.0**2+a6[0]*6.0**3+a8[0]*6.0**4+a10[0]*6.0**5+a12[0]*6.0**6)) / (0.5*mu_p/(1+a2[1]*6.0+a4[1]*6.0**2+a6[1]*6.0**3+a8[1]*6.0**4+a10[1]*6.0**5+a12[1]*6.0**6)))
    GMp = mu_p/(1+a2[1]*Q2+a4[1]*sq(Q2)+a6[1]*power(Q2,3)+a8[1]*power(Q2,4)+a10[1]*power(Q2,5)+a12[1]*power(Q2,6))
    GMn = mu_n/(1+a2[2]*Q2+a4[2]*sq(Q2)+a6[2]*power(Q2,3)+a8[2]*power(Q2,4)+a10[2]*power(Q2,5)+a12[2]*power(Q2,6))
    #GEn = -mu_n*0.942*(Q2/(4*sq(M))) / (1.+(Q2/(4*M**2)*4.61)) * (1./(1. + (sq(Q2)/M_V_nuance**2)))
    GEV = (GEp - GEn)
    GMV = (GMp - GMn)
    ## Create the form factors as a function of Q^2 = -q^2 ##
    F_1 = (GEV + (Q2/(4.*M**2)*GMV))/(1. + Q2/(4.*sq(M)))
    F_2 = (GMV - GEV)/((1. + Q2/(4.*M**2)))
    F_A = g_A / (1. + Q2/sq(M_A))
    #F_P = 2.0*sq(m_N)*F_A/(M_pi**2 + Q2)
    F_P = 2.*sq(m_N)/(-Q2)*(g_A/(1+Q2/sq(M_pi)) - F_A)       # from 1972 paper

    return F_1,F_2,F_A,F_P,M_A

##############################################
## Create double differential cross section ##
### opt = 1 neutrino, opt = 2 antineutrino ###
##############################################
def make_double_diff_monopole(E_mu,E_nu,P_mu,cos_mu,M_A,opt):
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
    F_1,F_2,F_A,F_P,M_A = make_form_factors_monopole(Q2,M_A)

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
        print("Please Enter 1 for neutrino or 2 for antineutrino in the last argument of make_double_diff_dipole")
        exit()
    double_diff = where(Q2 > 6., 0., double_diff)
    return double_diff,M_A

###################################
## function to do the dipole fit ##
###################################
def make_total_xs_monopole(E_nu,M_A):

    E_nu_array = E_nu
    N = len(E_nu_array)
    SIGMA = array([0.0 for i in range(N)])
    for m  in range(N):
        print "Starting Calculation for E_nu = %s out of E_nu = %s" % (round_sig(E_nu_array[m]),round_sig(nanmax(E_nu_array)))
        N_cos = int(E_nu_array[m]+1)*800
        N_T = 50+20*int(E_nu_array[m])
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
            double_diff,M_A = make_double_diff_monopole(E_mu,E_nu,P_mu,cos_bin,M_A,1)

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

#################################
## make sdcs for fitting BW FF ##
#################################
def make_single_diff(Q2_passed,N):
    ## parameters ##
    num = 12                                                # number of Nucleons
    m_N = 0.9389                                            # mass of the Nucleon
    m_mu = 0.1057                                           # mass of Muon GeV
    G_F = 1.166*10**(-5)                                    # Fermi Constant
    cos = 0.974                                             # cosine of cabibbo angle
    GeV_To_Cm = 5.06773076*10**(13)                         # Conversion factor for GeV to cm
    M_A = 1.35

    ## Create an array of the neutrino flux ##
    Flux = array([45.4,171.,222.,267.,332.,364.,389.,
        409.,432.,448.,456.,458.,455.,451.,443.,
        431.,416.,398.,379.,358.,335.,312.,288.,
        264.,239.,214.,190.,167.,146.,126.,108.,
        92.,78.,65.7,55.2,46.2,38.6,32.3,
        27.1,22.8,19.2,16.3,13.9,11.9,
        10.3,8.96,7.87,7,6.3,5.73,5.23,
        4.82,4.55,4.22,3.99,3.84,3.63,
        3.45,3.33,3.20])*10**(-12)

    ## fill in the Q^2 = -q^2 values ##
    E_nu = linspace(0., 3.,len(Flux),endpoint=True)
    Func = interp1d(E_nu,Flux,kind='cubic')

    num_flux = 1000
    E_nu_new = linspace(0.02,3.,num_flux,endpoint=True)
    Flux_new = Func(E_nu_new)

    Total_Flux = 0
    for i in range(len(Flux_new)):
        Total_Flux = Total_Flux + (3.-.02)/num_flux*(Flux_new[i])

    Q2,E_nu_new = meshgrid(Q2_passed,E_nu_new,indexing='ij')

    N_Q = len(Q2_passed)

    ## Define the weight functions needed to integrate over the flux ##
    weight = ((3.-.02)/num_flux)*(Flux_new/Total_Flux)/(num/2)
    weight = broadcast_to(weight,(N_Q,num_flux))

    #Q2 = broadcast_to(Q2_passed,(N_E,N_Q))
    #Q2 = swapaxes(Q2,0,1)
    #E_nu = broadcast_to(E_nu_new,(N_Q,N_E))

    Q2 = Q2_passed
    E_nu_new = 2.5

    ## calculate the form factors ##
    F_1,F_2,F_A,F_P,M_A = make_form_factors_monopole(Q2,M_A)

    ## Define some convenient variables for single differential cross section ##
    ## x = (s-u)  ##
    x = 4.0*m_N*E_nu_new - Q2 - sq(m_mu)
    y = Q2/(4.*sq(m_N))
    A = (m_mu**2 + Q2)/m_N**2 * ( (1. + y)*sq(F_A) - (1.-y)*sq(F_1) + (y-y**2)*sq(F_2) + 4*y*F_1*F_2 - (m_mu/2/m_N)**2*( sq(F_1 + F_2) + sq(F_A + 2.*F_P) - (Q2/m_N**2 + 4.)*sq(F_P) ))
    B = Q2/m_N**2*F_A*(F_1+F_2)
    C = 0.25*( sq(F_A) + sq(F_1) + y*sq(F_2))

    R = 1.
    #R = (1. + 6.*Q2*exp(-Q2/0.34))

    ## Calculate the single differential XS ##
    single_diff = N*((sq(m_N)*sq(G_F)*sq(cos))/(8.*pi*sq(E_nu_new)*sq(GeV_To_Cm)))*(A - x*B/sq(m_N) + sq(x)*C/sq(sq(m_N)))*R
    #single_diff = ((sq(m_N)*sq(G_F))/(8.*pi*sq(E_nu_new)*sq(GeV_To_Cm)))*(A + x*B/sq(m_N) + sq(x)*C/sq(sq(m_N)))

    #single_diff = weight_sum_2d(single_diff,weight).real

    return single_diff

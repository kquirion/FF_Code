## XS_functions_binned  ##
## Here wehave the functions for checking how total cross section depends on the range of Q^2 being used ##

from math import log10, floor
from numpy import array,linspace,longdouble,where,sqrt,broadcast_to,swapaxes,log,power,nanmin,nanmax,conjugate,sum,maximum,minimum,empty,meshgrid,arccos,amin,amax,exp,zeros,logspace,log10,vstack,hstack
from math import pi
from scipy.integrate import quad
from sys import exit
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from XSFunctions import make_a_elements,make_form_factors_dipole,calc_cross_section
from misc_fns import *
from variable_fns import *

########################################################################
## Create a function to make the less constrained kinematic variables ##
########################################################################
def make_variables_unbinned(N_T,N_cos,E_nu):
    m_N = (0.9389)                                            # mass of the Nucleon
    m_mu = (0.1057)                                           # mass of Muon GeV
    p_F = (0.220)                                             # Fermi Momentum
    E_hi = sqrt(sq(m_N) + sq(p_F))                            # Upper limit of neutron energy integration

    T_mu_max = E_nu + E_hi - m_mu - m_N
    T_mu_min = 0.05
    T_mu = linspace(T_mu_min,T_mu_max,N_T,endpoint=False)
    DELTA_T_mu = (T_mu_max-T_mu_min)/N_T
    E_mu = T_mu + m_mu
    P_mu = sqrt(sq(E_mu) - sq(m_mu))

    ## Restrict cos values to those satisfying Q2 > 0 ##
    cos_max = E_mu/P_mu - m_mu**2/(2.0*E_nu*P_mu)
    cos_max = where(cos_max < 1.0, cos_max, 1.0)
    #cos_max = 20.*pi/180.
    cos_mu = array([linspace(-cos_max[i],cos_max[i],2*N_cos,endpoint=False) for i in range(N_T)])
    #cos_mu = array([linspace(-cos_max,cos_max,2*N_cos,endpoint=False) for i in range(N_T)])

    DELTA_cos_mu = array([0.0  for i in range(N_T)])
    for i in range(N_T):
        DELTA_cos_mu[i] = abs(cos_mu[i][1] - cos_mu[i][0])

    T_mu = broadcast_to(T_mu,(2*N_cos,N_T))
    T_mu = swapaxes(T_mu,0,1)
    E_mu = broadcast_to(E_mu,(2*N_cos,N_T))
    E_mu = swapaxes(E_mu,0,1)
    P_mu = broadcast_to(P_mu,(2*N_cos,N_T))
    P_mu = swapaxes(P_mu,0,1)

    return T_mu,E_mu,P_mu,E_nu,cos_mu,DELTA_cos_mu,DELTA_T_mu

##############################################
## Create double differential cross section ##
### opt = 1 neutrino, opt = 2 antineutrino ###
##############################################
def make_double_diff_binned(E_mu,E_nu,P_mu,cos_mu,M_A,lower,upper):
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
    F_1,F_2,F_A,F_P,M_A = make_form_factors_dipole(Q2,M_A)

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
    double_diff = where(lower < Q2, double_diff, 0.)
    double_diff = where(upper > Q2, double_diff, 0.)

    return double_diff,M_A

###################################
## function to do the dipole fit ##
###################################
def make_total_xs_binned(E_nu,M_A):

    Miniboone_XData = array([.425,.475,.525,.575,.625,.675,.725,.775,.85,.95,1.05,1.2,1.4,1.75])
    Miniboone_XS = array([7.985,8.261,8.809,9.530,10.13,10.71,11.11,11.55,12.02,12.30,12.58,12.58,12.78,12.36])*10**(-39)
    Miniboone_Error = array([1.997,1.532,1.330,1.209,1.24,1.089,1.065,1.078,1.129,1.217,1.359,1.662,2.116,2.613])*10**(-39)

    Nomad_XData = array([4.7,7.7,10.5,13.5,17.8,23.8,29.8,35.8,45.3,71.7])
    Nomad_XS = array([9.94,9.42,10.14,8.59,8.43,9.91,8.88,9.70,8.96,9.11])*10**(-39)
    Nomad_Error = array([1.25,0.72,0.61,0.57,0.40,0.52,0.64,.86,.70,.73])*10**(-39)

    A_Minerva_XData = array([1.75,2.25,2.75,3.25,3.75,4.5,5.5,6.5,7.5,9.0])
    A_Minerva_XS = array([2.9773,3.7445,4.4340,4.7043,4.2805,4.1718,4.8057,6.2044,5.8574,5.6274])*10**(-39)
    A_Minerva_Error = array([17.21,13.69,11.64,11.14,11.22,11.75,20.93,31.25,34.57,33.25])*10**(-41)

    Minerva_XData = array([1.75,2.25,2.75,3.25,3.75,4.25,4.75,5.5,7.,9.,12.5,17.5])
    Minerva_XS = 2.0*array([3.72,4.67,5.24,5.36,5.16,5.13,5.63,6.15,6.91,6.56,6.74,7.79])*10**(-39)
    Minerva_Error = 2.0*array([8.88,5.58,5.34,4.66,5.50,7.15,8.15,6.91,7.33,7.56,7.62,10.2])*10**(-40)

    A_Miniboone_XData = array([.425,.475,.525,.575,.625,.675,.725,.775,.85,.95,1.05,1.2,1.4,1.75])
    A_Miniboone_XS = array([1.808,1.89,2.019,2.258,2.501,2.728,2.932,3.091,3.372,3.815,4.254,4.789,5.784,7.086])*10**(-39)
    A_Miniboone_Error = array([6.267,4.471,4.433,4.384,4.335,4.559,4.39,4.56,4.821,5.663,6.704,9.831,17.42,31.26])*10**(-40)

    m_mu = 0.1057
    E_nu_array = E_nu
    N = len(E_nu_array)
    num_Q2  = 24
    N_cos_max = int(amax(E_nu_array)+1)*800
    N_T_max = 50+20*int(amax(E_nu_array))
    T_mu,E_mu,P_mu,E_nu,cos_mu,DELTA_cos_mu,DELTA_T_mu = make_variables_unbinned(N_T_max,N_cos_max,E_nu_array[N-1])
    Q2 = 2.0*E_mu*E_nu - 2.0*E_nu*P_mu*cos_mu - m_mu**2
    bin_edges = linspace(0.,8.,num_Q2+1)
    E_low  = -1.
    E_high = log10(20.)

    SIGMA_TOT = zeros(200)
    y = []
    y_labels = []

    for k in range(num_Q2-6):
        SIGMA = zeros(N)
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
                double_diff,M_A = make_double_diff_binned(E_mu,E_nu,P_mu,cos_bin,M_A,bin_edges[k],bin_edges[k+1])

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

        ###############################################
        ## plot the contribution of  each Q^2  range ##
        ###############################################

        Func = interp1d(E_nu_array,SIGMA,kind='cubic')
        newer_E_nu = logspace(E_low,E_high,200)
        SIGMA_new = Func(newer_E_nu)

        #Func = interp1d(E_nu_array,SIGMA,kind='cubic')
        #newer_E_nu = logspace(-1.,log10(20.),100)
        #SIGMA_new = Func(newer_E_nu)

        SIGMA_TOT  = SIGMA_TOT + SIGMA_new
        y.append(SIGMA_TOT)
        y_labels.append("Q2 < %s GeV^2" % bin_edges[k+1])


    fig = plt.figure()
    SIGMA_graph = fig.gca()
    SIGMA_graph.set_xlabel(r'$E_{\nu}$ ($GeV$)')
    SIGMA_graph.set_ylabel(r'$\sigma$ ($cm^2$)')
    SIGMA_graph.set_title(r'Neutrino $^{12}C$ Cross Section ')
    SIGMA_graph.set_xlim(0.1,20.0)
    SIGMA_graph.set_ylim(0.0,2.0*10**(-38))

    SIGMA_graph.stackplot(newer_E_nu,y,linestyle='-',linewidth=2,labels=y_labels)
    SIGMA_graph.errorbar(Minerva_XData,Minerva_XS,yerr=Minerva_Error,marker='s',color='m',fmt='o',label='Minerva XS')
    SIGMA_graph.errorbar(Miniboone_XData,Miniboone_XS,yerr=Miniboone_Error,marker='s',color='black',fmt='o',label='Miniboone XS')
    SIGMA_graph.errorbar(Nomad_XData,Nomad_XS,yerr=Nomad_Error,marker='s',color='grey',fmt='o',label='Nomad XS')
    SIGMA_graph.legend()

    fig.savefig("Desktop/Research/Axial FF/Plots/Q2 Conts 2./Q2_Stacks.pdf" )

    return SIGMA_TOT

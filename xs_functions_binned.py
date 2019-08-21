## XS_functions_binned  ##
## Here wehave the functions for checking how total cross section depends on the range of Q^2 being used ##

from math import log10, floor
from numpy import array,linspace,longdouble,where,sqrt,broadcast_to,swapaxes,log,power,nanmin,nanmax,conjugate,sum,maximum,minimum,empty,meshgrid,arccos,amin,amax,exp,zeros,logspace,log10,vstack,hstack
from math import pi
from scipy.integrate import quad
from sys import exit
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from xs_functions_dipole import *
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

## Interpolate the flux function that we are  integrating over to calculate a better ddxs  ##
def flux_interpolate_binned((N,num_flux),M_A,lower,upper):

    ## Mass of the muon in GeV and angle  cut in degrees ##
    m_mu = .1057
    angle_cut = 20.
    p = 1
    Flux_FHC = array([2.57,6.53,17.,25.1,33.1,40.7,42.8,34.2,20.4,11.1,6.79,4.87,3.95,3.34,2.91,2.55,2.29,2.05,1.85,1.7,1.54,1.41,1.28,1.18,1.07,
        .989,.906,.842,.761,.695,.619,.579,.532,.476,.44,.403,.371,.34,.317,.291])*3.34*10**(14)
    Flux_RHC = array([1.26,1.69,1.78,1.88,1.90,1.96,1.9,1.82,1.73,1.65,1.64,1.70,1.75,1.80,1.76,1.73,1.65,1.57,1.47,1.37,1.28,1.17,1.08,.998,.919,
        .832,.76,.677,.643,.574,.535,.479,.445,.397,.336,.33,.311,.285,.264,.239])
    Flux_minerva = Flux_FHC + Flux_RHC

    ## Lower edges of the bins ##
    #p_T_1D_low = array([0.,.075,.15,.25,.325,.4,.475,.55,.7,.85,1.,1.25,1.5])
    p_T_1D_low = array([0.,.075,.15,.25,.325,.4,.475,.55,.7,.85,1.,1.25])
    p_P_1D_low = array([1.5,2.,2.5,3.,3.5,4.,4.5,5.,6.,8.,10.,15.])
    p_P_1D_low = linspace(1.,15.,p*N)
    ## higher edges of the bins ##
    #p_T_1D_high = array([.075,.15,.25,.325,.4,.475,.55,.7,.85,1.,1.25,1.5,2.5])
    p_T_1D_high = array([.075,.15,.25,.325,.4,.475,.55,.7,.85,1.,1.25,1.5])
    p_P_1D_high = array([2.,2.5,3.,3.5,4.,4.5,5.,6.,8.,10.,15.,20.])
    p_P_1D_high = linspace(2.,20.,p*N)
    ## middle of each bin ##
    p_T_1D = (p_T_1D_low + p_T_1D_high)/2.
    p_P_1D = (p_P_1D_low + p_P_1D_high)/2.

    len_pt = len(p_T_1D)
    len_pp = len(p_P_1D)
    
    ## Number of bins to average  the ddxs over ##
    p_T_1d = array([linspace(p_T_1D_low[i],p_T_1D_high[i],N) for i in range(len_pt)])
    p_P_1d = array([linspace(p_P_1D_low[i],p_P_1D_high[i],N) for i in range(p*N)])
    
    ## Interpolate the flux data ##
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

    ## an array to hold the different ddxs arrays to be averaged over ##
    ddxs_holder = zeros((N,p*N,len_pt))
    
    ###############################################
    ## define the kinematic inputs for each case ##
    ###############################################
    for o in range(N):
        #if (o+1)%100 == 0.:
        #    print ("%s percent done" % o/N/(len(Q2_bins)-1))
        p_P_2D,p_T_2D = meshgrid(p_P_1d[:,o],p_T_1d[:,o],indexing='ij')
        cos_mu_2D = p_P_2D/sqrt(sq(p_P_2D) + sq(p_T_2D))
        T_mu_2D = sqrt(sq(p_P_2D) + sq(p_T_2D) + sq(m_mu)) - m_mu
        Jac = p_T_2D/(T_mu_2D+m_mu)/sqrt(sq(p_P_2D) + sq(p_T_2D))
    
        p_P_3D,p_T_3D,E_nu_3D = meshgrid(p_P_1d[:,o],p_T_1d[:,o],E_nu_new,indexing = 'ij')
        T_mu_3D = sqrt(sq(p_P_3D) + sq(p_T_3D) + sq(m_mu)) - m_mu
        cos_mu_3D = p_P_3D/sqrt(sq(p_T_3D) + sq(p_P_3D))
        E_mu_3D = T_mu_3D + m_mu
        P_mu_3D = sqrt(sq(p_T_3D) + sq(p_P_3D))
    
        weight = broadcast_to(weight,(len_pp,len_pt,len(Flux_new)))
    
        ######################################################
        ## make double diff from fit to total cross section ##
        ######################################################
        double_diff = make_double_diff_dipole(E_mu_3D,E_nu_3D,P_mu_3D,cos_mu_3D,M_A,1)
    
        #######################################################################
        ## find the bounds on the indices where the cross section is nonzero ##
        #######################################################################
        a = empty([len_pp,len_pt], dtype=int)
        b = empty([len_pp,len_pt],dtype=int)
        for i in range(len_pp):
            for j in range(len_pt):
                A = 0
                B = num_flux-1
                while double_diff[i,j,A] == 0.:
                    A += 1
                    if A ==  num_flux:
                        break
                while  double_diff[i,j,B] == 0. :
                    B -= 1
                    if B == 0:
                        break
                a[i,j] = A-1
                b[i,j] = B+1
        b = where( b == num_flux, num_flux-1, b)
    
        ## Find new ranges of flux for each combo of p_T and p_|| ##
        even_newer_E_nu = empty([len_pp,len_pt,num_flux])
        for i in range(len_pp):
            for j in range(len_pt):
                temp_flux = linspace(E_nu_new[a[i,j]],E_nu_new[b[i,j]],num_flux)
                for k in range(num_flux):
                    even_newer_E_nu[i,j,k] = temp_flux[k]
        ## Create new weight funcntions for each combo of p_T and p_|| ##
        newer_flux_new = Func(even_newer_E_nu)
        new_weights = empty([len_pp,len_pt,num_flux])
        for i in range(len_pp):
            for j in range(len_pt):
                temp_max = amax(even_newer_E_nu[i,j])
                temp_min = amin(even_newer_E_nu[i,j])
                for k in range(num_flux):
                    new_weights[i,j,k] = ((temp_max-temp_min)/num_flux)*(newer_flux_new[i,j,k]/Total_Flux)
    
        double_diff = make_double_diff_binned(E_mu_3D,even_newer_E_nu,P_mu_3D,cos_mu_3D,M_A,lower,upper)
        double_diff_temp = weight_sum_3d(double_diff.real,new_weights)/12.
        double_diff_temp = where(cos_mu_2D < cos(angle_cut*pi/180), 0., double_diff_temp)
        double_diff_temp  = double_diff_temp*Jac
        for i in range(p*N):
            for j in range(len_pt):
                ddxs_holder[o][i,j] = double_diff_temp[i,j]
    
    ddxs_avg = array(sum(ddxs_holder,0)/N)
    
    return ddxs_avg


##############################################
## Create double differential cross section ##
##############################################
def make_double_diff_binned(E_mu,E_nu,P_mu,cos_mu,M_A,lower,upper):
    ## parameters ##
    A = 12                                                  # number of Nucleons
    m_N = 0.9389                                            # mass of the Nucleon
    E_b = 0.025                                             # binding energy GeV
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
    F_1,F_2,F_A,F_P = make_form_factors_dipole(Q2,M_A)

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
    double_diff = where(lower <= Q2, double_diff, 0.)
    double_diff = where(upper >= Q2, double_diff, 0.)
    #double_diff = where(40.<= Q2, 0., double_diff)
    #double_diff = where(cos_mu < cos(20.*pi/180), 0., double_diff)

    return double_diff

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
    N_cos_max = 200 + int(amax(E_nu_array)+1)*800
    N_T_max = 100+20*int(amax(E_nu_array))
    T_mu,E_mu,P_mu,E_nu,cos_mu,DELTA_cos_mu,DELTA_T_mu = make_variables_unbinned(N_T_max,N_cos_max,E_nu_array[N-1])
    Q2 = 2.0*E_mu*E_nu - 2.0*E_nu*P_mu*cos_mu - m_mu**2
    print amin(Q2)
    bin_edges = array([0.0,0.2,0.4,0.6,0.9,1.2,1.5,2.0,3.,7.5,round_sig(amax(Q2))])
    num_Q2 = len(bin_edges)
    E_low  = -1.
    E_high = log10(20.)

    SIGMA_TOT = zeros(200)
    y = []
    y_labels = []
    k = 0
    while k < num_Q2-1:
        SIGMA = zeros(N)
        for m  in range(N):
            print(" %s%% complete" % (1.*(k*len(E_nu_array) + m)/(1.*(num_Q2-1)*len(E_nu_array))*100.))
            N_cos = 100 + int(E_nu_array[m]+1)*800
            N_T = 50 + 30*int(E_nu_array[m])

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
                double_diff = make_double_diff_binned(E_mu,E_nu,P_mu,cos_bin,M_A,bin_edges[k],bin_edges[k+1])

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
        ## uncomment these below if  you want to  get a rough idea  what the graph should look like, with lower number of T_mu, cos_mu points ##
        #for j in range(100):
        #    for i in range(len(newer_E_nu)):
        #        if newer_E_nu[i] >= 9.:
        #            SIGMA_new[i] = SIGMA_new[i-1]
        y.append(SIGMA_new)
        y_labels.append(r"%s < $Q^2$ < %s " % (bin_edges[k],bin_edges[k+1]))
        SIGMA_TOT  = SIGMA_TOT + SIGMA_new
        k += 1

    #SIGMA_old = make_total_xs_dipole(E_nu_array,M_A)

    fig = plt.figure()
    SIGMA_graph = fig.gca()
    SIGMA_graph.set_xlabel(r'$E_{\nu}$ ($GeV$)')
    SIGMA_graph.set_ylabel(r'$\sigma$ ($cm^2$)')
    SIGMA_graph.set_title(r'Neutrino Cross Section: $M_A = %s GeV$ ' % M_A, y=1.05)
    SIGMA_graph.set_xlim(0.1,20.0)
    SIGMA_graph.set_ylim(0.0,2.0*10**(-38))
    SIGMA_graph.set_xscale('log')

    if M_A == 1.05:
        col = 'green'
    elif M_A == 1.35:
        col = 'red'
    elif M_A ==  1.45:
        col =  'cyan'

    SIGMA_graph.stackplot(newer_E_nu,y,linestyle='-',linewidth=2,labels=y_labels)
    SIGMA_graph.plot(newer_E_nu,SIGMA_TOT,color=col,linestyle='-')
    SIGMA_graph.errorbar(Miniboone_XData,Miniboone_XS,yerr=Miniboone_Error,marker='s',color='black',fmt='o',label='Miniboone XS')
    #SIGMA_graph.errorbar(Minerva_XData,Minerva_XS,yerr=Minerva_Error,marker='s',color='m',fmt='o',label='Minerva XS')
    SIGMA_graph.errorbar(Nomad_XData,Nomad_XS,yerr=Nomad_Error,marker='s',color='grey',fmt='o',label='Nomad XS')
    chartBox = SIGMA_graph.get_position()
    SIGMA_graph.set_position([chartBox.x0, chartBox.y0, chartBox.width*0.9, chartBox.height])
    SIGMA_graph.legend(loc='upper center', title = r"$Q^2$ in $GeV^2$", bbox_to_anchor=(1.12, 1.), shadow=True, ncol=1, prop={'size': 6})
    fig.savefig("Desktop/Research/Axial FF/Plots/Q2 Conts 2./Q2_Stacks_%s_v6..pdf" % M_A )


    return SIGMA_TOT

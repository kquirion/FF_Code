from numpy import array,linspace,zeros,meshgrid,sqrt,where,pi,cos,amax,log10,logspace,inf,sum
from misc_fns import sq,round_sig
from xs_functions_dipole import make_double_diff_dipole,calc_cross_section,make_total_xs_dipole
from xs_functions_binned import make_variables_unbinned,make_double_diff_binned
from variable_fns import make_variables
import matplotlib.pyplot as plt

## Interpolate the flux function that we are  integrating over to calculate a better ddxs  ##
def make_ddxs_unintegrated(N,E_nu,M_A,lower,upper):
    
    ## Mass of the muon in GeV and angle  cut in degrees ##
    m_mu = .1057
    angle_cut = 20.
    
    ###############################################
    ## define the kinematic inputs for each case ##
    ###############################################  
    T_mu,E_mu,P_mu,E_nu,cos_mu,DELTA_cos_mu,DELTA_T_mu = make_variables_unbinned(N,N,E_nu)
    Jac = sqrt(1.-cos_mu**2)/E_mu
    
    Q2 = 2.*E_nu*(E_mu-P_mu*cos_mu) - m_mu**2
    
    ######################################################
    ## make double diff from fit to total cross section ##
    ######################################################
    double_diff = make_double_diff_dipole(E_mu,E_nu,P_mu,cos_mu,M_A,1)
    double_diff = double_diff/2.
    double_diff = where(cos_mu <= cos(angle_cut*pi/180.), 0., double_diff)
    
    double_diff = where(lower < Q2, double_diff, 0.)
    double_diff = where(upper > Q2, double_diff, 0.)

    return double_diff
    
#######################################
## function to do the dipole fit ######
#######################################
def make_total_xs_unintegrated(E_nu,M_A):
    angle_cut = 20.
    m_mu = .1057
    print "Starting Calculation "
    SIGMA = zeros((len(E_nu)))
    SIGMA_TOT = 0.
    N_cos = 100
    N_T = 2*N_cos
    N = 2*N_cos
    ################################
    ## Create Kinematic Variables ##
    ################################
    T_mu,E_mu,P_mu,E_nu_prime,cos_mu,DELTA_cos_mu,DELTA_T_mu = make_variables_unbinned(N_T,N_cos,E_nu[len(E_nu)-1])
    Q2 = 2.0*E_mu*E_nu[len(E_nu)-1] - 2.0*E_nu[len(E_nu)-1]*P_mu*cos_mu - m_mu**2 
    Q2_bins = array([0.0,0.2,0.4,0.6,0.9,1.2,1.5,2.0,3.,7.5,round_sig(amax(Q2))])
    y = zeros((len(Q2_bins)-1,len(E_nu)))
    y_labels = []
    for i in range(len(E_nu)):
        N_cos = 50
        N_T = 2*N_cos
        N = 2*N_cos
        for l in range(len(Q2_bins)-1): 
            SIGMA_temp = 0.    
            T_mu,E_mu,P_mu,E_nu_prime,cos_mu,DELTA_cos_mu,DELTA_T_mu = make_variables_unbinned(N_T,N_cos,E_nu[i])
            print "Q2 in [%s,%s]"  % (Q2_bins[l],Q2_bins[l+1])
            #####################################################################
            ## Create RFG Variables #############################################
            ## For the last entry, put 1 for neutrinos, or 2 for antineutrinos ##
            #####################################################################
            #double_diff = make_double_diff_binned(E_mu,E_nu[i],P_mu,cos_mu,M_A,Q2_bins[l],Q2_bins[l+1])
            double_diff = make_ddxs_unintegrated(N,E_nu[i],M_A,Q2_bins[l],Q2_bins[l+1])
            #############################
            ## Calculate Cross Section ##
            #############################
            SIGMA_temp,new_E_nu = calc_cross_section(E_nu[i],N_T,N_cos*2,DELTA_cos_mu,DELTA_T_mu,double_diff)
            ################################################
            ## append new xs value to  array for stacking ##
            ################################################
            y_labels.append(r"%s < $Q^2$ < %s " % (Q2_bins[l],Q2_bins[l+1]))
            y[l,i] = SIGMA_temp
    SIGMA = sum(y,0)
                  
    Minerva_XData = array([1.75,2.25,2.75,3.25,3.75,4.25,4.75,5.5,7.,9.,12.5,17.5])
    Minerva_XS = 2.0*array([3.72,4.67,5.24,5.36,5.16,5.13,5.63,6.15,6.91,6.56,6.74,7.79])*10**(-39)
    Minerva_Error = 2.0*array([8.88,5.58,5.34,4.66,5.50,7.15,8.15,6.91,7.33,7.56,7.62,10.2])*10**(-40)  
    
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

    num_SIGMA = 20
    E_low  = -1.
    E_high = log10(10.)
    new_E_nu = logspace(E_low,E_high,num_SIGMA)
    SIGMA_TOT = make_total_xs_dipole(new_E_nu,M_A)
    
    SIGMA_graph.stackplot(E_nu,y,linestyle='-',linewidth=2,labels=y_labels)
    SIGMA_graph.plot(new_E_nu,SIGMA_TOT,color=col,linestyle='-')
    #SIGMA_graph.scatter(E_nu,SIGMA,marker='s',color='black')
    #SIGMA_graph.errorbar(Miniboone_XData,Miniboone_XS,yerr=Miniboone_Error,marker='s',color='black',fmt='o',label='Miniboone XS')
    SIGMA_graph.errorbar(Minerva_XData,Minerva_XS,yerr=Minerva_Error,marker='s',color='m',fmt='o',label='Minerva XS')
    #SIGMA_graph.errorbar(Nomad_XData,Nomad_XS,yerr=Nomad_Error,marker='s',color='grey',fmt='o',label='Nomad XS')
    chartBox = SIGMA_graph.get_position()
    SIGMA_graph.set_position([chartBox.x0, chartBox.y0, chartBox.width*0.9, chartBox.height])
    SIGMA_graph.legend(loc='upper center', title = r"$Q^2$ in $GeV^2$", bbox_to_anchor=(1.12, 1.), shadow=True, ncol=1, prop={'size': 6})
    fig.savefig("Desktop/Research/Axial FF/Plots/Q2_Stacks_unintegrated.pdf"  )
    print(SIGMA)
    return SIGMA
    
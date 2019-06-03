## This file contains many functions for running cross section calculations ##
from math import log10, floor 
from numpy import array,linspace,longdouble,where,sqrt,broadcast_to,swapaxes,log,power,nanmin,nanmax,conjugate,sum,maximum,minimum,empty,meshgrid,arccos,amin,amax,exp,zeros,logspace,log10
from math import pi
from scipy.integrate import quad
from sys import exit
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt



## Create the integrands for the b elements ##
def Integrand_Zero(x,E_b,m_N,m_T,V,q):
    return (m_T*V)/(2*pi*q)*(x)/(x-E_b)*(x/m_N)**0
        
def Integrand_One(x,E_b,m_N,m_T,V,q):
    return (m_T*V)/(2*pi*q)*(x)/(x-E_b)*(x/m_N)**1
    
def Integrand_Two(x,E_b,m_N,m_T,V,q):
    return (m_T*V)/(2*pi*q)*(x)/(x-E_b)*(x/m_N)**2

###############################################################################
## calculates the square of a number, or the element-wise square of an array ##
############################################################################### 
def sq(A):
    return A*conjugate(A)         

###############################################################################      
## Create a function for rounding to 4 significant figures (for readability) ##
###############################################################################   
def round_sig(x, sig=4):
   return round(x, sig-int(floor(log10(abs(x))))-1)

############################################################################   
## make a function to convert 1D array into a 2D array of dimension (x,y) ##
############################################################################
def make2d(x,y,vector):
    a = len(x)
    b = len(y)
    vector2 = array([[0.0 for j in range(b)] for i in range(a)]) 
    for i in range(a):
        for j in range(b):
            vector2[i][j] = vector[i*b+j]         
    return vector2

####################################################################    
## Create a function to calculate point-wise and total xi squared ##
####################################################################   
def calc_chi_squared(Theo,Exp,Exp_err):
    chi_squared_individual =  sq((Theo-Exp)/Exp_err) 
    chi_squared = sum(chi_squared_individual)
    return chi_squared_individual,chi_squared
    
##################################################################################
## Create a function that performs a weighted sum over the last axis of 'vector'##
##################################################################################
def weight_sum_3d(vector,weight):
    a = vector.shape
    x = a[0]
    y = a[1]
    z = a[2]
    output = array([[0.0 for i in range(y)] for j in range(x)])
    for i in range(x):
        for j in range(y):
            Int = 0
            for k in range(z-1):
                Int = Int + 0.5*(weight[i][j][k]*vector[i][j][k] + weight[i][j][k+1]*vector[i][j][k+1])
            output[i][j] = Int
    return output
    
def weight_sum_2d(vector,weight):
    a = vector.shape
    x = a[0]
    y = a[1]
    output = array([0.0 for i in range(x)])
    for j in range(x):
        Int = 0
        for k in range(y-1):
            Int = Int + 0.5*(weight[j][k]*vector[j][k] + weight[j][k+1]*vector[j][k+1] )
        output[j] = Int
    return output
 
   
## Interpolate the flux function that we are  integrating over to calculate a better ddxs  ##  
def flux_interpolate(num_flux,M_A):
    
    m_mu = .1057
    
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
    E_nu_max = 20.   
    num_neuts = 12.
    
    num_nucs = 12.    
    E_nu_Flux = linspace(0.,E_nu_max,len(Flux))     
    Func = interp1d(E_nu_Flux,Flux,kind='cubic')
    E_nu_new = linspace(0.,E_nu_max,num_flux)
    Flux_new = Func(E_nu_new)

    Total_Flux = 0
    for i in range(len(Flux_new)):
        Total_Flux = Total_Flux + E_nu_max/num_flux*(Flux_new[i])   
       
    ## Define the weight functions needed to integrate over the flux ##
    weight = []
    for i in range(len(Flux_new)):
        weight.append( (E_nu_max/num_flux)*Flux_new[i]/Total_Flux)  
    
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
    

    weight = broadcast_to(weight,(len(p_T_1D),len(p_P_1D),len(Flux_new))) 
        
    ######################################################
    ## make double diff from fit to total cross section ##  
    ######################################################
    double_diff,M_A = make_double_diff_dipole(E_mu_3D,E_nu_3D,P_mu_3D,cos_mu_3D,num_nucs,M_A,1)  
    
    #######################################################################
    ## find the bounds on the indices where the cross section is nonzero ##
    #######################################################################
        
    a = empty([len(p_P_1D),len(p_T_1D)], dtype=int)
    b = empty([len(p_P_1D),len(p_T_1D)],dtype=int)
    for i in range(len(p_P_1D)):
        for j in range(len(p_T_1D)):
            A = 0
            B = num_flux-1
            while double_diff[i][j][A] == 0.:
                A += 1
                if A ==  num_flux:
                    break
            while  double_diff[i][j][B] == 0. :
                B -= 1
                if B == 0:
                    break
            a[i][j] = A-1
            b[i][j] = B+1
    
    b = where( b == num_flux, num_flux-1, b)
    
    
    #for i in range(len(p_T_1D)):
    #    for j in range(len(p_P_1D)):        
    #        print ( "(%s,%s)" % (a[i][j],b[i][j]))
    
    ## Find new ranges of flux for each combo of p_T and p_|| ##
    even_newer_E_nu = empty([len(p_P_1D),len(p_T_1D),num_flux])
    for i in range(len(p_P_1D)):
        for j in range(len(p_T_1D)):
            temp_flux = linspace(E_nu_new[a[i][j]],E_nu_new[b[i][j]],num_flux)
            for k in range(num_flux):
                even_newer_E_nu[i][j][k] = temp_flux[k]
    
    ## Create new weight funcntions for each combo of p_T and p_|| ##
    newer_flux_new = Func(even_newer_E_nu)
    new_weights = empty([len(p_P_1D),len(p_T_1D),num_flux])
    for i in range(len(p_P_1D)):
        for j in range(len(p_T_1D)):
            temp_max = amax(even_newer_E_nu[i][j])
            temp_min = amin(even_newer_E_nu[i][j])
            for k in range(num_flux):
                new_weights[i][j][k] = ((temp_max-temp_min)/num_flux)*(newer_flux_new[i][j][k]/Total_Flux)
                
    double_diff,M_A = make_double_diff_dipole(E_mu_3D,even_newer_E_nu,P_mu_3D,cos_mu_3D,num_nucs,M_A,1) 
    double_diff = weight_sum_3d(double_diff.real,new_weights)/num_neuts
    double_diff  = double_diff*Jac
    
    double_diff = double_diff.ravel()
    
    return double_diff

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
    
    T_mu = broadcast_to(T_mu,(int(2*N_cos/100),N_T))
    T_mu = swapaxes(T_mu,0,1)
    E_mu = broadcast_to(E_mu,(int(2*N_cos/100),N_T))
    E_mu = swapaxes(E_mu,0,1)
    P_mu = broadcast_to(P_mu,(int(2*N_cos/100),N_T))
    P_mu = swapaxes(P_mu,0,1)
                   
    return T_mu,E_mu,P_mu,E_nu,cos_mu,DELTA_cos_mu,DELTA_T_mu
    
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

#######################################################    
## Create a function to make the kinematic variables ##
#######################################################
def make_variables_constrained(N_T,N_cos,E_nu):
    m_N = (0.9389)                                            # mass of the Nucleon
    m_mu = (0.1057)                                           # mass of Muon GeV
    p_F = (0.220)                                             # Fermi Momentum   
    
    ## make the original cos_mu vector from [-1,1]
    cos_mu = linspace(-1.0,1.0,2*N_cos)
    cos_mu = broadcast_to(cos_mu,(N_T,2*N_cos))
        
    
    ## 1D Create arrays for neutron momentum and energy ##
    p_n = linspace(-p_F,p_F,N_T)
    p_n = broadcast_to(p_n,(2*N_cos,N_T))
    p_n = swapaxes(p_n,0,1)
    E_n = sqrt(sq(p_n) + sq(m_N))
    
    ## define quantities for solving the quadratic equation for T_mu ##
    e = -2.0*(E_nu+E_n)*(E_nu*(E_n-m_mu-p_n)- m_mu*E_n + sq(m_mu)/2.)
    a = sq(E_nu+E_n) - sq(cos_mu)*sq(E_nu+p_n)
    b = e - 2.0*sq(cos_mu)*sq(E_nu+p_n)
    c = sq(e)/(sq(E_nu+E_n))
    
    cos_bounds = where(sq(b)-4.0*a*c > 0.0 , cos_mu,10)
    cos_bound = nanmin(cos_bounds,axis=1)
    cos_bound = where(cos_bound < 1.0, cos_bound, 1.0)
    
    cos_mu = array([linspace(-cos_bound[i],cos_bound[i],2*N_cos,endpoint=False) for i in range(N_T)])

    a = sq(E_nu+E_n) - sq(cos_mu)*sq(E_nu+p_n)
    b = e - 2.0*sq(cos_mu)*sq(E_nu+p_n)
    c = sq(e)/(sq(E_nu+E_n))
    d = sq(b)-4.0*a*c 
    d = nanmax(d)

    T_mu_max = abs(nanmax(1.0/(2.0*a)*(-b+sqrt(d))))
    T_mu_min = abs(nanmin(1.0/(2.0*a)*(-b-sqrt(d))))
    DELTA_T_mu = abs(T_mu_max-T_mu_min)/N_T
    T_mu = linspace(T_mu_min,T_mu_max,N_T,endpoint=False)
    E_mu = T_mu + m_mu
    P_mu = sqrt(sq(E_mu) - sq(m_mu))
    
    cos_max = E_mu/P_mu - sq(m_mu)/(2.0*E_nu*P_mu)
    cos_max = where(cos_max < 1.0 ,cos_max,1.0) 
    
    for i in range(len(cos_bound)):
        cos_max = where(cos_bound > cos_max, cos_bound, cos_max)
    
    cos_mu = array([linspace(-cos_max[i],cos_max[i],2*N_cos,endpoint=False) for i in range(N_T)])
    #cos_mu = np.where((-0.24 < cos_mu) & (cos_mu < 0.24) , 0.25, cos_mu)
    
    DELTA_cos_mu = array([0.0 for i in range(N_T)])
    for i in range(N_T):
        DELTA_cos_mu[i] = abs(cos_mu[i][1] - cos_mu[i][0])
    
    T_mu = broadcast_to(T_mu,(int(2*N_cos/500),N_T))
    T_mu = swapaxes(T_mu,0,1)
    E_mu = broadcast_to(E_mu,(int(2*N_cos/500),N_T))
    E_mu = swapaxes(E_mu,0,1)
    P_mu = broadcast_to(P_mu,(int(2*N_cos/500),N_T))
    P_mu = swapaxes(P_mu,0,1)
    
    return T_mu,E_mu,P_mu,E_nu,cos_mu,DELTA_cos_mu,DELTA_T_mu
 
###########################   
## Create the a elements ##
###########################
def make_a_elements(Q2,q,w,w_eff):
    A = (12)                                                  # number of target neutrons
    m_N = (0.9389)                                            # mass of the Nucleon
    p_F = (0.220)                                             # Fermi Momentum
    E_hi = sqrt(m_N**2 + p_F**2)                       # Upper limit of neutron energy integration
    E_b = 0.025                                             # binding energy GeV
    m_T = A*(m_N-E_b)                                       # mass of target Nucleus
    V = (3*pi**2*A)/(2.*p_F**3)                         # Volume of the tagret
    
    ## fill in the values for some useful coefficients ##
    c = -w_eff/q                  
    d = -(sq(w_eff) - sq(q))/(2.*q*m_N ) 
    alpha = 1.0 - sq(c) + sq(d)
    alpha = where( alpha > 0., alpha, 0)
    
    #i,j,k = where(alpha != 0.0)
    #for l in range(len(i)):
    #    print ("( %s,%s,%s ) -> cos_mu = %s" % (i[l],j[l],k[l],alpha[i[l]][j[l]][k[l]])) 
    kappa = 1. - 0.8*exp(-20.*Q2)
    kappa = 1.
    E_lo = maximum( kappa*(E_hi - w_eff), m_N*( c*d + sqrt(alpha) )/( 1. - sq(c) ) )
    E_lo = where(E_lo < E_hi, E_lo, E_hi)   
    E_lo = where(E_lo <= E_b, E_hi,E_lo)
                        
    ## Expressions for the b elements analytically ##
    #b_0 = quad(Integrand_Zero, E_lo, E_hi, args=(E_b,m_N,m_T,V,q))
    #b_1 = quad(Integrand_One, E_lo, E_hi, args=(E_b,m_N,m_T,V,q))
    #b_2 = quad(Integrand_Two, E_lo, E_hi, args=(E_b,m_N,m_T,V,q))
    b_0 = (m_T*V)/(2.0*pi*q)*((E_hi - E_lo) + E_b*log((E_hi - E_b)/(E_lo - E_b)))
    b_1 = (m_T*V)/(2.0*pi*m_N*q)*(.5*(sq(E_hi) - sq(E_lo)) + E_b*(E_hi - E_lo) + (sq(E_b))*log((E_hi - E_b)/(E_lo - E_b)))
    b_2 = (m_T*V)/(2.0*pi*sq(m_N)*q)*(1./3.*(E_hi**3 - power(E_lo,3)) + E_b*(0.5*(sq(E_hi) - sq(E_lo)) + E_b*(E_hi - E_lo) + sq(E_b)*log((E_hi - E_b)/(E_lo - E_b))))
    ## using the b elements as well as c and d to calculate the a elements ##
    a_1 = b_0
    a_2 = b_2 - b_0
    a_3 = sq(c)*b_2 + 2.0*(c*d*b_1) + sq(d)*b_0
    a_4 = b_2 - 2.0*E_b*b_1/m_N + sq(E_b)/sq(m_N)*b_0
    a_5 = -c*b_2 + (E_b*c/m_N - d)*b_1 + (E_b*d/m_N)*b_0
    a_6 = -c*b_1 - d*b_0
    a_7 = b_1 - E_b*b_0/m_N
    
    #i,j,k = where(alpha <= 0.0)
    #for l in range(len(i)):
    #    print ("( %s,%s ) -> a_1 = %s" % (i[l],j[l],alpha[i[l]][j[l]][0]))    
    #print len(i)
    
    return a_1,a_2,a_3,a_4,a_5,a_6,a_7,E_lo,E_hi

#######################
## Make form factors ##
#######################
def make_form_factors(Q2,Form_factor):
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
    g_A = -1.2723                                            # F_A(q^2 = 0)
    M_pi = 0.1396                                           # mass of the pion
    M_rho = 0.775
    Q_low = 1.3                                             # Energies for determining the form of Dr Friedland's F_A parametrization
    Q_high = 1000.0
    m_a_1 = 1.230                                           # Masses for defining the Masjuan double pole parametrization for F_A
    m_a_2 = 1.647
    
    GEp = where(Q2 < 6.,1./(1 + a2[0]*Q2 + a4[0]*sq(Q2) + a6[0]*power(Q2,3) + a8[0]*power(Q2,4) + a10[0]*power(Q2,5) + a12[0]*power(Q2,6)),(mu_p/(1+a2[1]*Q2+a4[1]*sq(Q2)+a6[1]*power(Q2,3)+a8[1]*power(Q2,4)+a10[1]*power(Q2,5)+a12[1]*power(Q2,6))) * (.5/(1.+a2[0]*6.+a4[0]*6.**2+a6[0]*6.**3+a8[0]*6.**4+a10[0]*6.**5+a12[0]*6.**6)) / (.5*mu_p/(1.+a2[1]*6.+a4[1]*6.**2+a6[1]*6.**3+a8[1]*6.**4+a10[1]*6.**5+a12[1]*6.**6)))                                                               
    GMp = mu_p/(1+a2[1]*Q2+a4[1]*sq(Q2)+a6[1]*power(Q2,3)+a8[1]*power(Q2,4)+a10[1]*power(Q2,5)+a12[1]*power(Q2,6))
    GMn = mu_n/(1+a2[2]*Q2+a4[2]*sq(Q2)+a6[2]*power(Q2,3)+a8[2]*power(Q2,4)+a10[2]*power(Q2,5)+a12[2]*power(Q2,6))  
    GEn = -mu_n*0.942*(Q2/(4*sq(M))) / (1+(Q2/(4*M**2)*4.61)) * (1/(1 + (sq(Q2)/M_V2**2)))                            
    GEV = (GEp - GEn)
    GMV = (GMp - GMn)                                           
    ## Create the form factors as a function of Q^2 = -q^2 ##
    #F_1 = (GEV + (Q2/(4*M**2)*GMV))/(1 + Q2/(4*sq(M)))
    #F_2 = (GMV - GEV)/((1 + Q2/(4*M**2)))
    F_1 = 1.
    F_2 = 1.
    if Form_factor == 1:
        M_A = 1.03
        F_A = g_A / sq(1 + Q2/sq(M_A))
    elif Form_factor == 2:
        M_A = 0.7
        F_A = g_A / sq(1 + Q2/sq(M_A))
    elif Form_factor == 3:
        ## create Dr. Friedland's Q^2 axial mass dependence ##
        M_A = "Varied"
        M_A_Varied = 1.35 + (sq(Q2)/Q_low**4)/(1.0 + (sq(Q2)/Q_low**4 + (power(Q2,5)/Q_high**10)))*(1.03-1.65)
        F_A = g_A / (1 + Q2/M_A_Varied**2)**2
    elif Form_factor == 4:
        M_A = "Double pole"
        F_A = g_A*(m_a_1**2)*(m_a_2**2)/((m_a_1**2 + Q2)*(m_a_2**2 + Q2))
    elif Form_factor == 5:
        M_A = 0.856
        GAMMA = 0.600
        F_A = where(Q2<sq(M_rho+M_pi),g_A*sq(M_A)/(sq(M_A) + Q2 - (1j*M_A*GAMMA*(4.1*power(Q2-9*sq(M_pi),3)*(1-3.3*(Q2-9*sq(M_pi))+5.8*sq(Q2-9*sq(M_pi))))/(4.1*(sq(M_A)-9*sq(M_pi))*(1-3.3*(sq(M_A)-9*sq(M_pi))+5.8*sq(sq(M_A)-9*sq(M_pi)))))),g_A*sq(M_A)/(sq(M_A) + Q2 - (1j*M_A*GAMMA*(Q2*(1.623+10.38/Q2-9.32/sq(Q2)+0.65/power(Q2,3)))/(sq(M_A)*(1.623+10.38/Q2-9.32/sq(sq(M_A))+0.65/power(sq(M_A),3))))))
    else:
        exit()    
    F_P = (2.0*m_N**2)*F_A/(M_pi**2 + Q2)
    
    return F_1,F_2,F_A,F_P,M_A

#############################
## Make FFs for dipole fit ##
#############################    
def make_form_factors_dipole(Q2,M_A):
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
    F_A = g_A / sq(1. + Q2/sq(M_A))
    #F_P = 2.0*sq(m_N)*F_A/(M_pi**2 + Q2)
    F_P = 2.*sq(m_N)/(-Q2)*(g_A/(1+Q2/sq(M_pi)) - F_A)       # from 1972 paper
    
    return F_1,F_2,F_A,F_P,M_A    

## Create the W elements which are used for the double differential cross sextion ##            
def make_double_diff_miniboone((T_mu,cos_mu,E_nu),M_A):
    ## parameters ##
    A = 12                                                  # number of Nucleons  
    m_N = 0.9389                                            # mass of the Nucleon
    E_b = 0.025                                             # binding energy GeV
    m_T = A*(m_N-E_b)                                       # mass of target Nucleus
    m_mu = 0.1057                                           # mass of Muon GeV
    V_ud = 0.9742                                           # Mixing element for up and down quarks
    GeV_To_Cm = 5.06773076*10**(13)                         # Conversion factor for GeV to cm
    G_F = 1.166*10**(-5)                                    # Fermi Constant
    POT = 5.58*10**(20)                                     # Integrated number of protons on target
    
    ## Create an array of the neutrino flux ##    
    Flux = array([45.4,171,222,267,332,364,389,
        409,432,448,456,458,455,451,443,
        431,416,398,379,358,335,312,288,
        264,239,214,190,167,146,126,108,
        92,78,65.7,55.2,46.2,38.6,32.3,
        27.1,22.8,19.2,16.3,13.9,11.9,
        10.3,8.96,7.87,7,6.3,5.73,5.23,
        4.82,4.55,4.22,3.99,3.84,3.63,
        3.45,3.33,3.20])*10**(-12) 
         
          
    T_mu_1d = linspace(0.25,1.95,18,endpoint=True) 
    cos_mu_1d = linspace(-.95,.95,20,endpoint=True) 
    E_nu_1d = linspace(0.05, (len(Flux))/20.0,len(Flux),endpoint=True)  
    
    NT = len(T_mu_1d)
    NC = len(cos_mu_1d)
    
    ## fill in the Q^2 = -q^2 values ##
    E_nu = linspace(0., 3.,len(Flux),endpoint=True)
    Func = interp1d(E_nu,Flux,kind='cubic')

    num_flux = 10000
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
    Q2 = 2.0*E_mu*E_nu_new - 2.0*E_nu_new*P_mu*cos_mu 
    ## fill in values of the W Boson Energy ##
    w = (E_nu_new - E_mu)           
    ## fill in the values of the W boson energy ##
    w_eff = (w - E_b)     
    ## fill in the values for the 3-momentum of the Boson ##
    q = sqrt(Q2 + sq(w))   
    
    ## calculate the a elements ##
    a_1,a_2,a_3,a_4,a_5,a_6,a_7,E_lo,E_hi = make_a_elements(Q2,q,w,w_eff)  
                                 
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
    
    
    double_diff = (sq(G_F)*P_mu*V_ud**2)/(16.0*sq(pi)*m_T*(GeV_To_Cm**2))*( 2.*(E_mu-P_mu*cos_mu)*W_1 + (E_mu+P_mu*cos_mu)*W_2 + (1/m_T)*((E_mu-P_mu*cos_mu)*(E_nu_new+E_mu) - sq(m_mu))*W_3 + sq(m_mu/m_T)*(E_mu-P_mu*cos_mu)*W_4 - (sq(m_mu)/m_T)*W_5)                            
    double_diff = weight_sum_3d(double_diff.real,weight).ravel()
        
    return double_diff

##############################################
## Create double differential cross section ##  
### opt = 1 neutrino, opt = 2 antineutrino ###
##############################################
def make_double_diff_dipole(E_mu,E_nu,P_mu,cos_mu,num_nucs,M_A,opt):
    ## parameters ## 
    A = num_nucs
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
    a_1,a_2,a_3,a_4,a_5,a_6,a_7,E_lo,E_hi = make_a_elements(Q2,q,w,w_eff)                                
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
    
    if opt == 1:
        double_diff = (sq(G_F)*P_mu*V_ud**2)/(16.0*sq(pi)*m_T*(GeV_To_Cm**2))*( 2.0*(E_mu-P_mu*cos_mu)*W_1 + (E_mu+P_mu*cos_mu)*W_2 + (1/m_T)*((E_mu-P_mu*cos_mu)*(E_nu+E_mu) - sq(m_mu))*W_3 + sq(m_mu/m_T)*(E_mu-P_mu*cos_mu)*W_4 - (sq(m_mu)/m_T)*W_5)                            
    elif opt == 2:
        double_diff = (sq(G_F)*P_mu*V_ud**2)/(16.0*sq(pi)*m_T*(GeV_To_Cm**2))*( 2.0*(E_mu-P_mu*cos_mu)*W_1 + (E_mu+P_mu*cos_mu)*W_2 - (1/m_T)*((E_mu-P_mu*cos_mu)*(E_nu+E_mu) - sq(m_mu))*W_3 + sq(m_mu/m_T)*(E_mu-P_mu*cos_mu)*W_4 - (sq(m_mu)/m_T)*W_5)
    else:
        print("Please Enter 1 for neutrino or 2 for antineutrino in the last argument of make_double_diff_dipole")
        exit()
    #double_diff = where(Q2 > 1., 0., double_diff)     
    return double_diff,M_A  


##################################################################################
## Calculate the total cross section by integrating over T_mu and cos_mu values ##
##################################################################################   
def calc_cross_section(E_nu,N_T,N_cos,DELTA_cos_mu,DELTA_T_mu,double_diff):
    A = 12                                                  # number of Nucleons  
    new_E_nu = E_nu
    S = double_diff.shape
    if S[0] != N_T:
        print "Error: T_mu dimension is incorrect."
        print ("(%s,(%s,%s))" % (S,N_T,2*N_cos)) 
        exit()
    elif S[1] !=  2*N_cos:
        print "Error: cos_mu dimension is incorrect."
        print ("(%s,(%s,%s))" % (S,N_T,2*N_cos))
        exit()
         
    ## Integrate double_diff over cos_mu and T_mu to get the total cross section ##
    diff = zeros(N_T)
    
    for i in range(N_T):
        for j in range(2*N_cos-1):
            diff[i] = diff[i] + 0.5*DELTA_cos_mu[i]*(double_diff[i][j] + double_diff[i][j+1]) 
                 
    INT = 0.0
    for i in range(N_T-1):
        INT = INT + 0.5*DELTA_T_mu*(diff[i]+diff[i+1])/(A/2.0)
    
    SIGMA = INT
    
    return SIGMA,new_E_nu
    

    
###################################
## function to do the dipole fit ##
###################################
def make_total_xs_dipole(E_nu,M_A):  
    
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
            double_diff,M_A = make_double_diff_dipole(E_mu,E_nu,P_mu,cos_bin,M_A,1) 
            
            ## apply the angle cut of Minerva ##
            double_diff = where((arccos(cos_bin)*180/pi <= 20) & (arccos(cos_bin)*180/pi >= -20), double_diff, 0.)
            
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
    F_1,F_2,F_A,F_P,M_A = make_form_factors_dipole(Q2,M_A) 
                    
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
    
            
                    
                    
    
    
## file  containing variable  creation functions ##
from numpy import array,linspace,sqrt,broadcast_to,swapaxes,nanmin,nanmax,where
from misc_fns import sq

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
    
#########################################################################
## Create a function to make kinematic variables where E_nu is an array##
#########################################################################
def make_variables_3D(N_T,N_cos,E_nu):
    
    N_E = len(E_nu)
    m_N = (0.9389)                                            # mass of the Nucleon
    m_mu = (0.1057)                                           # mass of Muon GeV
    p_F = (0.220)                                             # Fermi Momentum
    E_hi = sqrt(sq(m_N) + sq(p_F))                            # Upper limit of neutron energy integration

    ## constraints on T_mu  ##
    T_mu_max = E_nu + E_hi - m_mu - m_N
    T_mu_max = where(T_mu_max > 0.05, T_mu_max, 0.06)
    T_mu_min = 0.05
    
    T_mu = array([linspace(T_mu_min,T_mu_max[k],N_T,endpoint=False) for k in range(N_E)] )
    T_mu = swapaxes(T_mu,0,1)
    E_mu = T_mu + m_mu
    P_mu = sqrt(sq(E_mu) - sq(m_mu))

    ## Restrict cos values to those satisfying Q2 > 0 ##
    cos_max = where(P_mu > 0.,E_mu/P_mu - m_mu**2/(2.0*E_nu*P_mu),0.)
    cos_max = where(cos_max < 1.0, cos_max, 1.0)

    cos_mu = array([[linspace(0.,cos_max[i,j],2*N_cos,endpoint=False) for j in range(N_E)] for i in range(N_T)])
    cos_mu = swapaxes(cos_mu,2,1)

    T_mu = array([[[T_mu[i,j] for j in range(N_E)] for k in range(2*N_cos)] for i in range(N_T)])
    E_mu = T_mu + m_mu
    P_mu = sqrt(sq(E_mu) - sq(m_mu))
    
    p_P = P_mu*cos_mu
    p_T = P_mu*sqrt(1.-cos_mu**2)

    return T_mu,E_mu,P_mu,cos_mu,p_P,p_T

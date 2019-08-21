## create tables of Q^2(E_nu,p_P) for each p_T ##

from numpy import array,linspace,sqrt,meshgrid

m_mu  = .1057

## Lower edges of the bins ##
p_T_1D_low = array([0.,.075,.15,.25,.325,.4,.475,.55,.7,.85,1.,1.25])
p_P_1D_low = array([1.5,2.,2.5,3.,3.5,4.,4.5,5.,6.,8.,10.,15.])
## higher edges of the bins ##
p_T_1D_high = array([.075,.15,.25,.325,.4,.475,.55,.7,.85,1.,1.25,1.5])
p_P_1D_high = array([2.,2.5,3.,3.5,4.,4.5,5.,6.,8.,10.,15.,20.])
## middle of each bin ##
p_T_1D = array((p_T_1D_low + p_T_1D_high)/2.)
p_P_1D = array((p_P_1D_low + p_P_1D_high)/2.)

Flux_FHC = array([2.57,6.53,17.,25.1,33.1,40.7,42.8,34.2,20.4,11.1,6.79,4.87,3.95,3.34,2.91,2.55,2.29,2.05,1.85,1.7,1.54,1.41,1.28,1.18,1.07,
    .989,.906,.842,.761,.695,.619,.579,.532,.476,.44,.403,.371,.34,.317,.291])*3.34*10**(14)
Flux = Flux_FHC 

E_nu_flux = linspace(0.,20.,len(Flux))

for k in range(len(p_T_1D)):
    
    p_P,E_nu =  meshgrid(p_P_1D,E_nu_flux,indexing='ij')
    
    p_mu = sqrt(p_P**2 + p_T_1D[k]**2)
    cos_mu = p_P/p_mu
    E_mu = sqrt(p_mu**2 +  m_mu**2)
    
    Q2 = 2.*E_nu*E_mu - 2.*p_mu*cos_mu - m_mu**2
    

    g=open("Desktop/Research/Axial FF/txt files/Q2 values/Q2_vals_pt=%s.txt" % p_T_1D[k],"w+")
    g.write("     Q2            E_nu            p_P             \n\n")
    for i in range(len(p_P_1D)):
        for j in range(len(p_T_1D)):
            g.write("    %s            %s            %s \n" % (Q2[i,j],E_nu[i,j],p_P[i,j]))
    g.close()    
    
    











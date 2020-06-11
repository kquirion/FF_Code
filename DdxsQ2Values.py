## create tables of Q^2(E_nu,p_P) for each p_T ##

from numpy import array,linspace,sqrt,meshgrid

E_nu_flux = linspace(0.,20.,len(Flux))

for k in range(len(p_T_1D)):
    
    p_P,E_nu =  meshgrid(p_P_1D,E_nu_flux,indexing='ij')
    
    p_mu = sqrt(p_P**2 + p_T_1D[k]**2)
    cos_mu = p_P/p_mu
    E_mu = sqrt(p_mu**2 +  m_mu**2)
    
    Q2 = 2.*E_nu*E_mu - 2.*E_nu*p_mu*cos_mu - m_mu**2
    

    g=open("Desktop/Research/Axial FF/txt files/Q2 values/Q2_vals_pt=%s.txt" % p_T_1D[k],"w+")
    g.write("     Q2            E_nu            p_P             \n\n")
    for i in range(len(p_P_1D)):
        for j in range(len(E_nu_flux)):
            g.write("    %s            %s            %s \n" % (Q2[i,j],E_nu[i,j],p_P[i,j]))
    g.close()    
    
    











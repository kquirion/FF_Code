from numpy import meshgrid,linspace,array,sqrt,pi,arccos,where
from os import getcwd
from math import floor,log10

   
m_mu = 0.1057                                           # mass of Muon GeV

## Lower edges of the bins ##
p_T_1D_low = array([0.,.075,.15,.25,.325,.4,.475,.55,.7,.85,1.,1.25,1.5])
p_P_1D_low = array([1.5,2.,2.5,3.,3.5,4.,4.5,5.,6.,8.,10.,15.])

## higher edges of the bins ##
p_T_1D_high = array([.075,.15,.25,.325,.4,.475,.55,.7,.85,1.,1.25,1.5,2.5])
p_P_1D_high = array([2.,2.5,3.,3.5,4.,4.5,5.,6.,8.,10.,15.,20.])

## middle of each bin ##
p_T_1D = (p_T_1D_low + p_T_1D_high)/2.
p_P_1D = (p_P_1D_low + p_P_1D_high)/2.

E_nu_Flux = linspace(0.25,19.75,40,endpoint=True)
 
p_T,p_P = meshgrid(p_T_1D,p_P_1D,indexing='ij')
theta = arccos(p_P/sqrt((p_P)**2 + (p_T)**2))*180./pi
T_mu = sqrt(p_T**2 + p_P**2 + m_mu**2) - m_mu

Jac = (p_T/sqrt(p_T**2+p_P**2)/(T_mu+m_mu))**(-1)

#################################################
## Define the info needed for flux integration ##
#################################################
Minerva_ddxs = array([
    [7.5,4.24,8.83,9.38,5.69,1.86,1.98,1.09,.84,.624,.387,.25],
    [22.7,32.,40.4,34.8,22.4,13.4,9.08,5.34,3.38,2.41,1.22,.503],
    [87.5,97.,114.,94.4,58.1,35.,20.5,14.5,8.36,4.79,2.29,.961],
    [154.,166.,185.,174.,113.,58.6,34.2,20.8,13.4,7.63,4.34,1.71],
    [194.,234.,254.,199.,132.,74.5,43.2,28.7,17.9,10.3,6.13,2.5],
    [226.,269.,285.,251.,147.,85.1,48.,35.2,18.8,13.4,7.7,3.27],
    [256.,294.,304.,255.,150.,81.6,49.7,30.6,18.8,11.,7.2,3.14],
    [140.,258.,260.,202.,120.,66.3,42.8,28.4,16.7,9.46,5.48,2.48],
    [5.78,143.,172.,132.,77.1,45.8,25.8,19.4,14.3,7.99,4.06,2.14],
    [0.,11.3,94.5,74.,43.,25.,17.6,12.9,7.82,6.06,2.81,1.6],
    [0.,0.,7.07,19.,14.4,9.09,7.38,5.67,3.15,2.49,1.62,.815],
    [0.,0.,0.,0.,1.33,1.41,2.08,1.72,.932,.851,.485,.25],
    [0.,0.,0.,0.,0.,.00685,.131,.108,.069,.0494,.0387,.0283]
    ])*10**(-41)

Minerva_cos_T = Minerva_ddxs*Jac


f=open("Desktop/Research/Axial FF/FF PDFs/Minerva_angles.txt","w+")  
f.write("p_T    p_||    theta     T_mu     momentum_ddxs   cos_ddxs \n\n")
for i in range(len(p_T_1D)):
    for j in range(len(p_P_1D)):
        f.write( "%s   %s   %s   %s   %s   %s \n\n" % (p_P[i][j],p_T[i][j],round(theta[i][j],4),round(T_mu[i][j],4),Minerva_ddxs[i][j],Minerva_cos_T[i][j]))
f.close()
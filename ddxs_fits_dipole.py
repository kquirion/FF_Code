# -*- coding: utf-8 -*-
## This file fits to Minerva and miniboone, calculates chi_sq, and plots the fit ##

import time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from pandas import DataFrame
from scipy.optimize import curve_fit,minimize
from scipy.interpolate import interp1d
from xs_functions_dipole import sq,weight_sum_3d,make2d,calc_chi_squared,make_form_factors_dipole,flux_interpolate,round_sig,make_double_diff_miniboone,flux_interpolate_unc
from numpy import (array,inf,where,linspace,power,broadcast_to,swapaxes,set_printoptions,sqrt,zeros,asarray,zeros_like,diag,
    meshgrid,nanmax,nanmin,cos,arccos,amin,amax,empty,transpose,concatenate,sum,append,set_printoptions )
from numpy.linalg import inv
from math import pi
from misc_fns import *

set_printoptions(precision=3)

N = 150
num_flux = 200

popt,pcov = curve_fit(flux_interpolate,(N,num_flux),Minerva_ddxs_true.ravel(),1.)

T_mu,cos_mu,E_nu = meshgrid(T_mu_1D,cos_mu_1D,E_nu_1D,indexing='ij')

print(" M_A_minerva = %s" % popt[len(popt)-1])

M_A = 1.35
M_A_minerva = popt[0]
col  = 'cyan'
if M_A == 1.35:
    col = 'red'
elif M_A == 1.05:
    col = 'green'
elif M_A == 1.45:
    col = 'cyan'


#M_A_minerva = 1.45
col = 'cyan'

#double_diff_miniboone = make_double_diff_miniboone(res_miniboone.x[0])
#double_diff_minerva = flux_interpolate(res_minerva.x[0])

double_diff_minerva = flux_interpolate((N,num_flux),popt[0])
minerva_unc = flux_interpolate_unc((N,num_flux),M_A_minerva)
minerva_chi_sq,minerva_tot_chi_sq =  calc_chi_squared(double_diff_minerva, Minerva_ddxs_true, Minerva_Error)

#double_diff_minerva = make2d(p_P_1D,p_T_1D,double_diff_minerva)
#minerva_chi_sq = make2d(p_P_1D,p_T_1D,minerva_chi_sq)
#minerva_unc = make2d(p_P_1D,p_T_1D,minerva_unc)

length_minerva  =  len(Minerva_ddxs_true.ravel()) - 1
length_miniboone  =  137 - 1

#print ("Uncertainty  for minerva is:  %s " % minerva_unc)
#print(minerva_chi_sq)
print ('Minerva chi^2/(d.o.f.) =  %s'  %  round_sig(minerva_tot_chi_sq/length_minerva))
print  minerva_tot_chi_sq

E_nu_Flux = linspace(0.,20.,40,endpoint=True)
E_nu_new = linspace(0.,20.,200,endpoint=True)

## recreate the cross section with new E_nu values from interpolation ##
p_P_2D,p_T_2D = meshgrid(p_P_1D,p_T_1D,indexing='ij')
cos_mu_2D = p_P_2D/sqrt(sq(p_P_2D) + sq(p_T_2D))
T_mu_2D = sqrt(sq(p_P_2D) + sq(p_T_2D) + sq(m_mu)) - m_mu
Jac = p_T_2D/(T_mu_2D+m_mu)/sqrt(sq(p_P_2D) + sq(p_T_2D))

p_P_3D,p_T_3D,E_nu_3D = meshgrid(p_P_1D,p_T_1D,E_nu_new,indexing = 'ij')
T_mu_3D = sqrt(sq(p_P_3D) + sq(p_T_3D) + sq(m_mu)) - m_mu
cos_mu_3D = p_P_3D/sqrt(sq(p_T_3D) + sq(p_P_3D))
E_mu_3D = T_mu_3D + m_mu
P_mu_3D = sqrt(sq(p_T_3D) + sq(p_P_3D))

#miniboone_chi_sq,miniboone_tot_chi_sq =  calc_chi_squared(double_diff_miniboone, Miniboone_XS, Miniboone_Error)
#print ('MiniBooNE chi^2/(d.o.f.) =  %s'  %  round_sig(miniboone_tot_chi_sq/length_miniboone))

## Create the plot for the double differential cross section ##
figax1 = plt.figure()
#figax2 = plt.figure()

ax1 = figax1.gca(projection='3d')
ax1.set_xlabel(r"$p_P$ (GeV)")
ax1.set_ylabel('$p_T$ (GeV)')
ax1.set_zlabel(r"$\frac{d\sigma}{dP_{||} \, dp_T} $")
ax1.set_title(r'Double Differential Cross Section $(cm^2/GeV^2)$')

#ax2 = figax2.gca(projection='3d')
#ax2.set_xlabel(r"$T_{\mu}$ (GeV)")
#ax2.set_ylabel('$cos\theta_\mu$ ')
#ax2.set_zlabel(r"$\frac{d\sigma}{dT_\mu \, dcos\theta_\mu} $")
#ax2.set_title(r'Double Differential Cross Section $(cm^2/GeV^2)$')

x,y = meshgrid(p_P_1D,p_T_1D,indexing='ij')
ax1.scatter(x,y,double_diff_minerva,color=col,marker='s',label="RFG Model: M_A = %s GeV" % round_sig(M_A_minerva))
ax1.scatter(x,y,Minerva_ddxs_true,color='black',marker='s',label="Minerva Neutrino Data",depthshade=False)
ax1.legend(loc=(0.35,0.7))

figax1.savefig("Desktop/Research/Axial FF/Plots/Minerva_ddxs_%s.pdf" % round_sig(M_A_minerva) )

#x2,y2 = meshgrid(T_mu_1D,cos_mu_1D,indexing='ij')
#ax2.scatter(x2,y2,double_diff_miniboone,color=col,marker='s',label="RFG Model: M_A = %s GeV" % round_sig(M_A_miniboone))
#ax2.scatter(x2,y2,Miniboone_XS,color='black',marker='s',label="Miniboone Neutrino Data",depthshade=False)

#ax2.legend(loc=(0.52,0.65))
#f=open("Desktop/Research/xial FF/txt files/miniboone_chisq_table.txt","w+")
#f.write(" data     model     error      chisq     \n")
#for i in range(len(T_mu_1D)):
#    for j in range(len(cos_mu_1D)):
#        f.write(" %s     %s     %s    %s   \n" % (Miniboone_XS[i,j], (double_diff_miniboone[i,j]),  Miniboone_Error[i,j], (miniboone_chi_sq[i,j])))
#.f.close()
g=open("Desktop/Research/Axial FF/txt files/minerva_chisq_table_%s_%s_bins.txt" % (round_sig(M_A_minerva),N),"w+")
g.write("\n\n Total chi-squared = %s \n chi-squared/d.o.f.  = %s \n M_A = %s  \n\n" % (minerva_tot_chi_sq,minerva_tot_chi_sq/length_minerva,round_sig(M_A_minerva)))
g.write("p_T       p_||       data       model       error       chisq     \n")
for i in range(len(p_P_1D)):
    for j in range(len(p_T_1D)):
        g.write("%s      %s      %s      %s      %s      %s \n" % (p_T_2D[i,j],p_P_2D[i,j],Minerva_ddxs[i,j],round_sig(double_diff_minerva[i*len(p_T_1D)+j]),Minerva_Error[i,j],round_sig(minerva_chi_sq[i*len(p_T_1D)+j])))
g.close()


plt.show()

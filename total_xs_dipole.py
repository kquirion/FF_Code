## In this code, I test out different ways of making the total cross section calculation faster and more memory efficient ##

from scipy.interpolate import interp1d
import os
import psutil
import time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from numpy import array,logspace,longdouble,log,log10
from xs_functions_dipole import *
import datetime
from math import log

start_time = time.time()
print(datetime.datetime.now())

#############################################################################################################
#############################################################################################################

num_SIGMA = 30
E_low  = -1.
E_high = log10(2.)
new_E_nu = logspace(E_low,E_high,num_SIGMA)


## Make Neutrino Cross Section ##
#for i in range(num_SIGMA):
#    SIGMA_TEMP,new_E_nu_temp,M_A = make_total_xs_2(100,2000,new_E_nu[i],5)
#    SIGMA[i] = SIGMA_TEMP

M_A = 1.05
M_A_2 = 1.35
M_A_3 = 1.14
M_A_4 =  1.45
## Make Neutrino Cross Section ##
#for i in range(num_SIGMA):
SIGMA = make_total_xs_dipole(new_E_nu,M_A)
print("--- %s Minutes Until Finishing First Cross Section" % ((time.time() - start_time)/60.0))
SIGMA_2 = make_total_xs_dipole(new_E_nu,M_A_2)
print("--- %s Minutes Until Finishing Second Cross Section" % ((time.time() - start_time)/60.0))
#SIGMA_3 = make_total_xs_dipole(new_E_nu,M_A_3)
#print("--- %s Minutes Until Finishing Third Cross Section" % ((time.time() - start_time)/60.0))
#SIGMA_4 = make_total_xs_dipole(new_E_nu,M_A_4)
#print("--- %s Minutes Until Finishing Fourth Cross Section" % ((time.time() - start_time)/60.0))

Func = interp1d(new_E_nu,SIGMA,kind='cubic')
Func_2 = interp1d(new_E_nu,SIGMA_2,kind='cubic')
#Func_3 = interp1d(new_E_nu,SIGMA_3,kind='cubic')
#Func_4 = interp1d(new_E_nu,SIGMA_4,kind='cubic')
newer_E_nu = logspace(E_low,E_high,200)
SIGMA_new = Func(newer_E_nu)
SIGMA_new_2 = Func_2(newer_E_nu)
#SIGMA_new_3 = Func_3(newer_E_nu)
#SIGMA_new_4 = Func_4(newer_E_nu)

#print(SIGMA)
#print(new_E_nu)

fig_SIGMA_BAR = plt.figure()
SIGMA_graph_BAR = fig_SIGMA_BAR.gca()
SIGMA_graph_BAR.set_xlabel(r'$E_{\nu}$ ($GeV$)')
SIGMA_graph_BAR.set_ylabel(r'$\sigma$ ($cm^2$)')
SIGMA_graph_BAR.semilogx(newer_E_nu,SIGMA_new,linestyle='-',linewidth=2,color='green',label=r'$M_A = %s GeV$' % M_A)
SIGMA_graph_BAR.semilogx(newer_E_nu,SIGMA_new_2,linestyle='-',linewidth=2,color='red',label=r'$M_A = %s GeV$' % M_A_2)
#SIGMA_graph_BAR.semilogx(newer_E_nu,SIGMA_new_3,linestyle='-',linewidth=2,color='orange',label=r'$M_A = %s GeV$' % M_A_3)
#SIGMA_graph_BAR.semilogx(newer_E_nu,SIGMA_new_4,linestyle='-',linewidth=2,color='cyan',label=r'$M_A = %s GeV$' % M_A_4)
#SIGMA_graph_BAR.errorbar(Minerva_XData,Minerva_XS,yerr=Minerva_Error,marker='s',color='m',fmt='o',label='Minerva XS')
SIGMA_graph_BAR.errorbar(Miniboone_XData,Miniboone_XS,yerr=Miniboone_Error,marker='s',color='black',fmt='o',label='Miniboone XS')
SIGMA_graph_BAR.errorbar(Nomad_XData,Nomad_XS,yerr=Nomad_Error,marker='s',color='grey',fmt='o',label='Nomad XS')
SIGMA_graph_BAR.legend(loc=(0.05,0.60))
SIGMA_graph_BAR.set_title(r'Neutrino $^{12}C$ Cross Section')
SIGMA_graph_BAR.set_xlim(0.1,20.0)
SIGMA_graph_BAR.set_ylim(0.0,2.*10**(-38))

plt.show()
print("--- %s Minutes Until Finish" % ((time.time() - start_time)/60.0))
fig_SIGMA_BAR.savefig("Desktop/Research/Axial FF/Plots/total_xs_miniboone_only.pdf" )

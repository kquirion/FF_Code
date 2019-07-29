## In this code, I test out different ways of making the total cross section calculation faster and more memory efficient ##

from scipy.interpolate import interp1d
import os
import psutil
import time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from numpy import array,logspace,longdouble,log,log10
from XSFunctions import *
import datetime
from math import log
from xs_functions_sergi import *

start_time = time.time()
print(datetime.datetime.now())

#############################################################################################################
#############################################################################################################

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

num_SIGMA = 30
E_low  = -1.
E_high = log10(20.)
new_E_nu = logspace(E_low,E_high,num_SIGMA)


## Make Neutrino Cross Section ##
#for i in range(num_SIGMA):
#    SIGMA_TEMP,new_E_nu_temp,M_A = make_total_xs_2(100,2000,new_E_nu[i],5)
#    SIGMA[i] = SIGMA_TEMP

## Make Neutrino Cross Section ##
#for i in range(num_SIGMA):
L = 1.5
SIGMA = make_total_xs_sergi(new_E_nu,L)
print("--- %s Minutes Until Finishing First Cross Section" % ((time.time() - start_time)/60.0))

Func = interp1d(new_E_nu,SIGMA,kind='cubic')
#Func_2 = interp1d(new_E_nu,SIGMA_2,kind='cubic')
#Func_3 = interp1d(new_E_nu,SIGMA_3,kind='cubic')
newer_E_nu = logspace(E_low,E_high,200)
SIGMA_new = Func(newer_E_nu)
#SIGMA_new_2 = Func_2(newer_E_nu)
#SIGMA_new_3 = Func_3(newer_E_nu)

#print(SIGMA)
#print(new_E_nu)

fig_SIGMA_BAR = plt.figure()
SIGMA_graph_BAR = fig_SIGMA_BAR.gca()
SIGMA_graph_BAR.set_xlabel(r'$E_{\nu}$ ($GeV$)')
SIGMA_graph_BAR.set_ylabel(r'$\sigma$ ($cm^2$)')
SIGMA_graph_BAR.semilogx(newer_E_nu,SIGMA_new,linestyle='-',linewidth=2,color='red',label=r'$M_A = 1.05$ GeV, $\Lambda = %s$ GeV' % L)
#SIGMA_graph_BAR.semilogx(newer_E_nu,SIGMA_new_2,linestyle='-',linewidth=2,color='green',label='Single Pole: MA = 1.0 GeV')
#SIGMA_graph_BAR.semilogx(newer_E_nu,SIGMA_new_3,linestyle='-',linewidth=2,color='cyan',label='Single Pole: MA = 0.9 GeV')
SIGMA_graph_BAR.errorbar(Minerva_XData,Minerva_XS,yerr=Minerva_Error,marker='s',color='m',fmt='o',label='Minerva XS')
SIGMA_graph_BAR.errorbar(Miniboone_XData,Miniboone_XS,yerr=Miniboone_Error,marker='s',color='black',fmt='o',label='Miniboone XS')
SIGMA_graph_BAR.errorbar(Nomad_XData,Nomad_XS,yerr=Nomad_Error,marker='s',color='grey',fmt='o',label='Nomad XS')
#SIGMA_graph_BAR.errorbar(Nomad_XData,Nomad_XS,yerr=Nomad_Error,color='black',fmt='o',label='Minerva Neutrino')
SIGMA_graph_BAR.legend(loc=(0.05,0.65))
SIGMA_graph_BAR.set_title(r'Neutrino $^{12}C$ Cross Section')
SIGMA_graph_BAR.set_xlim(0.1,20.0)
SIGMA_graph_BAR.set_ylim(0.0,2.5*10**(-38))

#plt.show()
print("--- %s Minutes Until Finish" % ((time.time() - start_time)/60.0))
fig_SIGMA_BAR.savefig("Desktop/Research/Axial FF/Plots/Sergi Parametrization/total_xs_sergi_%s_1.05.pdf" % L )
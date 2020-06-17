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

NumSigma = 30
Elow  = -1.
Ehigh = log10(2.)
NewEnu = logspace(Elow,Ehigh,NumSigma)

MA = 1.05
MA2 = 1.35
MA3 = 1.14
MA4 =  1.45
## Make Neutrino Cross Section ##
SIGMA = XsDipole(NewEnu,MA)
print("--- %s Minutes Until Finishing First Cross Section" % ((time.time() - start_time)/60.0))
SIGMA_2 = XsDipole(newEnu,MA2)
print("--- %s Minutes Until Finishing Second Cross Section" % ((time.time() - start_time)/60.0))
#SIGMA_3 = XsDipole(NewEnu,MA3)
#print("--- %s Minutes Until Finishing Third Cross Section" % ((time.time() - start_time)/60.0))
#SIGMA_4 = XsDipole(NewEnu,MA4)
#print("--- %s Minutes Until Finishing Fourth Cross Section" % ((time.time() - start_time)/60.0))

Func = interp1d(NewEnu,SIGMA,kind='cubic')
Func2 = interp1d(NewEnu,SIGMA_2,kind='cubic')
#Func_3 = interp1d(NewEnu,SIGMA_3,kind='cubic')
#Func_4 = interp1d(NewEnu,SIGMA_4,kind='cubic')
NewerEnu = logspace(Elow,Ehigh,200)
SigmaNew = Func(NewerEnu)
SigmaNew2 = Func2(NewerEnu)
#SigmaNew3 = Func3(NewerEnu)
#SigmaNew4 = Func4(NewerEnu)

#print(SIGMA)
#print(NewEnu)

FigSigma = plt.figure()
SigmaGraph = FigSigma.gca()
SigmaGraph.set_xlabel(r'$E_{\nu}$ ($GeV$)')
SigmaGraph.set_ylabel(r'$\sigma$ ($cm^2$)')
SigmaGraph.semilogx( NewerEnu, SigmaNew, linestyle='-', linewidth=2, color='green', label=r'$MA = %s GeV$' % MA )
SigmaGraph.semilogx( NewerEnu, SigmaNew2, linestyle='-', linewidth=2, color='red', label=r'$MA = %s GeV$' % MA2 )
#SigmaGraph.semilogx( NewerEnu, SigmaNew3, linestyle='-', linewidth=2, color='orange', label=r'$MA = %s GeV$' % MA3 )
#SigmaGraph.semilogx( NewerEnu, SigmaNew4, linestyle='-', linewidth=2, color='cyan', label=r'$MA = %s GeV$' % MA4)
#SigmaGraph.errorbar( MinervaXData, MinervaXs, yerr=MinervaError, marker='s', color='m', fmt='o', label='Minerva XS' )
SigmaGraph.errorbar( MinibooneXData, MinibooneXs, yerr=MinibooneError, marker='s', color='black', fmt='o', label='Miniboone XS')
SigmaGraph.errorbar( NomadXData, NomadXs, yerr=NomadError, marker='s', color='grey', fmt='o', label='Nomad XS' )
SigmaGraph.legend(loc=(0.05,0.60))
SigmaGraph.set_title(r'Neutrino $^{12}C$ Cross Section')
SigmaGraph.set_xlim(0.1,20.0)
SigmaGraph.set_ylim(0.0,2.*10**(-38))

plt.show()
print("--- %s Minutes Until Finish" % ((time.time() - start_time)/60.0))
FigSigma.savefig("Desktop/Research/Axial FF/Plots/total_xs_miniboone_only.pdf" )

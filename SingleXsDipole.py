## Calculat Single Differential XS ##

import time
import matplotlib.pyplot as plt
from numpy import array,diag,sqrt,linspace
from scipy.optimize import curve_fit
from xs_functions_dipole import make_single_diff,round_sig

start_time = time.time()

#############################
## Find M_A that fits data ##
#############################
popt,pcov = curve_fit(make_single_diff,Miniboone_XData,Miniboone_XS,sigma=Miniboone_Error,absolute_sigma=True)
perr = sqrt(diag(pcov))
print("N = %s pm %s" % (popt[0],perr))


Q2 = linspace(0.001,2.,50)

single_diff = make_single_diff(Q2,1.)
single_diff_2 = make_single_diff(Q2,popt[0])

print single_diff

fig_SIGMA_BAR = plt.figure()
SIGMA_graph_BAR = fig_SIGMA_BAR.gca()
SIGMA_graph_BAR.set_xlabel(r'$Q^2$ ($GeV$)')
SIGMA_graph_BAR.set_ylabel(r'$\frac{d\sigma}{dQ^2}$ ($cm^2$)')
SIGMA_graph_BAR.plot(Q2,single_diff,linestyle='-',linewidth=2,color='red',label='Normalization Factor = 1')
SIGMA_graph_BAR.plot(Q2,single_diff_2,linestyle='-',linewidth=2,color='blue',label='Normalization Factor = %s ' % round_sig(popt[0]))
SIGMA_graph_BAR.errorbar(Miniboone_XData,Miniboone_XS,yerr=Miniboone_Error,color='black',fmt='o',label='MiniBooNE Data')
#SIGMA_graph_BAR.scatter(A_Miniboone_XData,A_Miniboone_XS,marker='s',color='black',label='Minerva Data')
#SIGMA_graph_BAR.errorbar(A_Miniboone_XData,A_Miniboone_XS,yerr=A_Miniboone_Error,color='black',fmt='o',label='MiniBoone Antineutrino')
SIGMA_graph_BAR.legend(loc=(0.52,0.65))
SIGMA_graph_BAR.set_title(r'Neutrino Single Differential Cross Section')
SIGMA_graph_BAR.set_xlim(0.0,2.0)
SIGMA_graph_BAR.set_ylim(0.0,35.0*10**(-39))

fig_SIGMA_BAR.savefig("Desktop/Research/Axial FF/Plots/Single_XS_MA1.35GeV.pdf" )

plt.show()

print("Time taken = %s seconds" % (time.time() - start_time))

## Fit for the total cross section ##

import time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from numpy import array,diag,sqrt,inf,logspace,longdouble 
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from scipy.optimize import curve_fit
from XSFunctions import *     
 
start_time = time.time()   

#############################################################################################################
#############################################################################################################
#############################################################################################################

#########################
## Just MiniBooNE Data ##
#########################
Miniboone_XData = array([.425,.475,.525,.575,.625,.675,.725,.775,.85,.95,1.05,1.2,1.4,1.75])
Miniboone_XS = array([7.985,8.261,8.809,9.530,10.13,10.71,11.11,11.55,12.02,12.30,12.58,12.58,12.78,12.36])*10**(-39)
Miniboone_Error = array([1.997,1.532,1.330,1.209,1.24,1.089,1.065,1.078,1.129,1.217,1.359,1.662,2.116,2.613])*10**(-39)

###########################################
## MiniBooNE with some Nomad Data Points ##
###########################################
Combined_XData = array([.425,.475,.525,.575,.625,.675,.725,.775,.85,.95,1.05,1.2,1.4,1.75,4.7,7.7,10.5,13.5,17.8])
Combined_XS = array([7.985,8.261,8.809,9.530,10.13,10.71,11.11,11.55,12.02,12.30,12.58,12.58,12.78,12.36,9.94,9.42,10.14,8.59,8.43])*10**(-39)
Combined_Error = array([1.997,1.532,1.330,1.209,1.24,1.089,1.065,1.078,1.129,1.217,1.359,1.662,2.116,2.613,1.25,0.72,0.61,0.57,0.40])*10**(-39)

################
## Nomad Data ##
################
NOMAD_xDATA = array([4.7,7.7,10.5,13.5,17.8,23.8,29.8,35.8,45.3,71.7])
NOMAD_DATA = array([9.94,9.42,10.14,8.59,8.43,9.91,8.88,9.70,8.96,9.11])*10**(-39)
NOMAD_ERROR = array([1.25,0.72,0.61,0.57,0.40,0.52,0.64,.86,.70,.73])*10**(-39)

###############################
## Minerva Antineutrino Data ##
###############################
A_Minerva_XData = array([1.75,2.25,2.75,3.25,3.75,4.5,5.5,6.5,7.5,9.0])
A_Minerva_XS = array([2.9773,3.7445,4.4340,4.7043,4.2805,4.1718,4.8057,6.2044,5.8574,5.6274])*10**(-39)
A_Minerva_Error = array([17.21,13.69,11.64,11.14,11.22,11.75,20.93,31.25,34.57,33.25])*10**(-41)

###########################
## Minerva Neutrino Data ##
###########################
Minerva_XData = array([1.75,2.25,2.75,3.25,3.75,4.25,4.75,5.5,7.,9.,12.5,17.5])
Minerva_XS = array([3.72,4.67,5.24,5.36,5.16,5.13,5.63,6.15,6.91,6.56,6.74,7.79])*10**(-39)
Minerva_Error = array([8.88,5.58,5.34,4.66,5.50,7.15,8.15,6.91,7.33,7.56,7.62,10.2])*10**(-40)

##################################
## make E_nu array for graphing ##
##################################
num_SIGMA = 130
E_nu = logspace(-0.9,1.3,num_SIGMA)

##############################################
## Fit to data using dipole parametrization ##
##############################################
#popt,pcov = curve_fit(make_total_xs_dipole,Minerva_XData,Minerva_XS,sigma=Minerva_Error,absolute_sigma=True)
#popt,pcov = curve_fit(make_total_xs, Miniboone_XData, Miniboone_XS,sigma=Miniboone_Error,absolute_sigma=True)
#perr = sqrt(diag(pcov))

#print("M_A = %s" % popt)
#print("Uncertainty is %s" % perr)
#SIGMA = make_total_xs_dipole(E_nu,popt[0])

##########################################
## Fit to data using BW parametrization ##
##########################################
popt,pcov = curve_fit(make_total_xs_BW,Combined_XData,Combined_XS,bounds=([0.,0.],[6.,6.]),sigma=Combined_Error,absolute_sigma=True)
perr = sqrt(diag(pcov))
print('(Gamma,M_A) = (%s,%s)' % (popt[0],popt[1]))
print pcov
print perr
SIGMA = make_total_xs_BW(E_nu,popt[0],popt[1])


###############################################
## Create a SIGMA for calculating xi_squared ##
###############################################
#SIGMA2 = make_total_xs(Minerva_XData,popt2[0],popt2[1])
#SIGMA2 = make_total_xs(Minerva_XData,popt2[0])

##########################
## Calculate xi-squared ##
##########################
#chi_sq_ind,chi_sq = calc_chi_squared(SIGMA ,Minerva_XS,Minerva_Error)

#print("chi-Squared of the individual points are %s " % chi_sq_ind)
#print("Total chi-Squared is %s " % xi_sq)

#SIGMA = make_total_xs(E_nu,2.4,1.03)

######################  
## Create the plots ##
######################
fig_SIGMA = plt.figure()    
SIGMA_graph = fig_SIGMA.gca()
SIGMA_graph.set_xlabel(r'$E_{\nu}$ ($GeV$)') 
SIGMA_graph.set_ylabel(r'$\sigma$ ($cm^2$)')     
SIGMA_graph.semilogx(E_nu,SIGMA,linestyle='-',linewidth=2,color='green',label=r'Dipole Fit')
#SIGMA_graph.scatter(E_nu_MiniBooNE,MiniBooNE_Total,marker='s',color='black',label='MiniBooNE Data')
#SIGMA_graph.scatter(Minerva_XData,Minerva_XS,marker='s',color='black',label='Minerva Antineutrino Data')
SIGMA_graph.errorbar(Combined_XData,Combined_XS,yerr=Combined_Error,color='black',fmt='o',label="Minerva Neutrino Data")
#SIGMA_graph.errorbar(E_nu_MiniBooNE,MiniBooNE_Total,yerr=MiniBooNE_Error,color='black',fmt='o')
#SIGMA_graph.scatter(NOMAD_xDATA,NOMAD_DATA,marker='s',color='gray',label='NOMAD Data')
#SIGMA_graph.errorbar(NOMAD_xDATA,NOMAD_DATA,yerr=NOMAD_ERROR,color='gray',fmt='o')
SIGMA_graph.set_xlim(0.1,10**1.4)
SIGMA_graph.set_ylim(0.0,1.7*10**(-38))
SIGMA_graph.set_title(r'Neutrino Total Cross Section $^{12}C$')
SIGMA_graph.legend(loc=4)
        
plt.show()

print("--- %s Minutes Until Finish" % ((time.time() - start_time)/60.0))

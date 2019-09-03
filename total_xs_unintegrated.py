import time
import matplotlib.pyplot as plt
from numpy import array,logspace,log10
from xs_functions_unintegrated import make_total_xs_unintegrated
import datetime

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


M_A = 1.35
#################################
## Make Neutrino Cross Section ##
#################################
E_nu = array([0.5,1.,2.])
#num_SIGMA = 30
#E_low  = -1.
#E_high = log10(5.)
#E_nu = logspace(E_low,E_high,num_SIGMA)
SIGMA = make_total_xs_unintegrated(E_nu,M_A)
print("--- %s Minutes Until Finishing First Cross Section" % ((time.time() - start_time)/60.0))

plt.show()
print("--- %s Minutes Until Finish" % ((time.time() - start_time)/60.0))

import time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from pandas import DataFrame
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
from XSFunctions import sq,weight_sum_3d,make2d,calc_chi_squared,make_form_factors_dipole,flux_interpolate,round_sig,make_double_diff_miniboone
from numpy import (array,inf,where,linspace,power,broadcast_to,swapaxes,set_printoptions,sqrt,
    meshgrid,nanmax,nanmin,cos,arccos,amin,amax,empty,transpose,concatenate )
from math import pi 

start_time = time.time()

  
#E_nu_Flux = linspace(0.25,19.75/2.,40,endpoint=True)
    
m_N = (0.9389)                                            # mass of the Nucleon
m_mu = (0.1057)                                           # mass of Muon GeV
p_F = (0.220)                                             # Fermi Momentum
A = 12                                                    # number of nucleons


#################################################
## Define the info needed for flux integration ##
##########################################################
## Rows are p_T and columns are p_|| qe-like scattering ##
##########################################################
Minerva_ddxs = array([
    [7.50e-41 ,2.27e-40 ,8.75e-40 ,1.54e-39 ,1.94e-39 ,2.26e-39 ,2.56e-39 ,1.40e-39 ,5.78e-41 ,0.00e+00 ,0.00e+00 ,0.00e+00 ,0.00e+00],
    [4.24e-41 ,3.20e-40 ,9.70e-40 ,1.66e-39 ,2.34e-39 ,2.69e-39 ,2.94e-39 ,2.58e-39 ,1.43e-39 ,1.13e-40 ,0.00e+00 ,0.00e+00 ,0.00e+00],
    [8.83e-41 ,4.04e-40 ,1.14e-39 ,1.85e-39 ,2.54e-39 ,2.85e-39 ,3.04e-39 ,2.60e-39 ,1.72e-39 ,9.45e-40 ,7.07e-41 ,0.00e+00 ,0.00e+00],
    [9.38e-41 ,3.48e-40 ,9.44e-40 ,1.74e-39 ,1.99e-39 ,2.51e-39 ,2.55e-39 ,2.02e-39 ,1.32e-39 ,7.40e-40 ,1.90e-40 ,0.00e+00 ,0.00e+00],
    [5.69e-41 ,2.24e-40 ,5.81e-40 ,1.13e-39 ,1.32e-39 ,1.47e-39 ,1.50e-39 ,1.20e-39 ,7.71e-40 ,4.30e-40 ,1.44e-40 ,1.33e-41 ,0.00e+00],
    [1.86e-41 ,1.34e-40 ,3.50e-40 ,5.86e-40 ,7.45e-40 ,8.51e-40 ,8.16e-40 ,6.63e-40 ,4.58e-40 ,2.50e-40 ,9.09e-41 ,1.41e-41 ,6.85e-44],
    [1.98e-41 ,9.08e-41 ,2.05e-40 ,3.42e-40 ,4.32e-40 ,4.80e-40 ,4.97e-40 ,4.28e-40 ,2.58e-40 ,1.76e-40 ,7.38e-41 ,2.08e-41 ,1.31e-42],
    [1.09e-41 ,5.34e-41 ,1.45e-40 ,2.08e-40 ,2.87e-40 ,3.52e-40 ,3.06e-40 ,2.84e-40 ,1.94e-40 ,1.29e-40 ,5.67e-41 ,1.72e-41 ,1.08e-42],
    [8.40e-42 ,3.38e-41 ,8.36e-41 ,1.34e-40 ,1.79e-40 ,1.88e-40 ,1.88e-40 ,1.67e-40 ,1.43e-40 ,7.82e-41 ,3.15e-41 ,9.32e-42 ,6.90e-43],
    [6.24e-42 ,2.41e-41 ,4.79e-41 ,7.63e-41 ,1.03e-40 ,1.34e-40 ,1.10e-40 ,9.46e-41 ,7.99e-41 ,6.06e-41 ,2.49e-41 ,8.51e-42 ,4.94e-43],
    [3.87e-42 ,1.22e-41 ,2.29e-41 ,4.34e-41 ,6.13e-41 ,7.70e-41 ,7.20e-41 ,5.48e-41 ,4.06e-41 ,2.81e-41 ,1.62e-41 ,4.85e-42 ,3.87e-43],
    [2.50e-42 ,5.03e-42 ,9.61e-42 ,1.71e-41 ,2.50e-41 ,3.27e-41 ,3.14e-41 ,2.48e-41 ,2.14e-41 ,1.60e-41 ,8.15e-42 ,2.50e-42 ,2.83e-43]
    ])
    
Minerva_ddxs_true = array([
    [4.01e-41, 1.34e-40, 6.70e-40, 1.21e-39 ,1.54e-39 ,1.77e-39 ,2.03e-39 ,1.11e-39 ,5.27e-41 ,0.00e+00 ,0.00e+00 ,0.00e+00 ,0.00e+00],
    [1.16e-41 ,2.06e-40 ,7.47e-40 ,1.32e-39 ,1.91e-39 ,2.19e-39 ,2.41e-39 ,2.10e-39 ,1.17e-39 ,1.01e-40 ,0.00e+00 ,0.00e+00 ,0.00e+00],
    [4.79e-41 ,2.87e-40 ,9.16e-40 ,1.51e-39 ,2.17e-39 ,2.42e-39 ,2.59e-39 ,2.20e-39 ,1.46e-39 ,8.48e-40 ,6.95e-41 ,0.00e+00 ,0.00e+00],
    [6.79e-41 ,2.60e-40 ,7.83e-40 ,1.50e-39 ,1.71e-39 ,2.20e-39 ,2.24e-39 ,1.75e-39 ,1.16e-39 ,6.75e-40 ,1.79e-40 ,0.00e+00 ,0.00e+00],
    [3.99e-41 ,1.73e-40 ,4.77e-40 ,9.82e-40 ,1.15e-39 ,1.28e-39 ,1.32e-39 ,1.05e-39 ,6.75e-40 ,3.86e-40 ,1.33e-40 ,1.15e-41 ,0.00e+00],
    [1.08e-41 ,1.01e-40 ,2.93e-40 ,5.07e-40 ,6.51e-40 ,7.51e-40 ,7.17e-40 ,5.72e-40 ,3.97e-40 ,2.18e-40 ,7.97e-41 ,1.10e-41 ,1.01e-42],
    [1.35e-41 ,7.32e-41 ,1.72e-40 ,2.91e-40 ,3.71e-40 ,4.17e-40 ,4.36e-40 ,3.68e-40 ,2.12e-40 ,1.46e-40 ,6.46e-41 ,1.63e-41 ,1.14e-42],
    [6.17e-42 ,4.23e-41 ,1.22e-40 ,1.78e-40 ,2.49e-40 ,3.07e-40 ,2.62e-40 ,2.41e-40 ,1.61e-40 ,1.07e-40 ,4.68e-41 ,1.38e-41 ,6.56e-43],
    [5.77e-42, 2.68e-41 ,6.90e-41 ,1.14e-40 ,1.55e-40 ,1.61e-40 ,1.57e-40 ,1.38e-40 ,1.19e-40 ,6.33e-41 ,2.47e-41 ,6.87e-42 ,3.87e-43],
    [4.05e-42 ,1.93e-41 ,3.92e-41 ,6.33e-41 ,8.79e-41 ,1.17e-40 ,9.08e-41 ,7.67e-41 ,6.46e-41 ,5.10e-41 ,2.01e-41 ,6.23e-42 ,2.33e-43],
    [1.50e-42 ,9.82e-42 ,1.78e-41 ,3.59e-41 ,5.28e-41 ,6.77e-41 ,6.16e-41 ,4.46e-41 ,3.24e-41 ,2.22e-41 ,1.34e-41 ,3.70e-42 ,2.01e-43],
    [8.51e-43 ,3.88e-42 ,7.44e-42 ,1.42e-41 ,2.13e-41 ,2.89e-41 ,2.72e-41 ,2.05e-41 ,1.79e-41 ,1.35e-41 ,6.77e-42 ,1.82e-42 ,1.56e-43]
    ])
    
Minerva_Error = transpose(array([
    [2.43e-41, 1.78e-41, 2.25e-41, 1.97e-41, 1.51e-41, 6.65e-42, 6.86e-42, 4.24e-42, 2.69e-42, 1.89e-42, 1.19e-42, 7.33e-43],
    [6.27e-41, 5.90e-41, 6.02e-41, 4.51e-41, 3.24e-41, 2.15e-41, 1.61e-41, 9.71e-42, 5.77e-42, 4.93e-42, 2.09e-42, 1.00e-42],
    [1.69e-40, 1.27e-40, 1.26e-40, 9.06e-41, 6.11e-41, 4.00e-41, 2.54e-41, 1.71e-41, 1.01e-41, 6.91e-42, 3.12e-42, 1.44e-42],
    [2.62e-40, 2.15e-40, 2.00e-40, 1.61e-40, 1.13e-40, 6.58e-41, 4.02e-41, 2.40e-41, 1.49e-41, 9.92e-42, 5.50e-42, 2.32e-42],
    [3.44e-40, 2.67e-40, 2.76e-40, 1.85e-40, 1.34e-40, 8.89e-41, 4.94e-41, 3.13e-41, 1.91e-41, 1.27e-41, 7.39e-42, 3.31e-42],
    [3.49e-40, 3.04e-40, 2.88e-40, 2.33e-40, 1.53e-40, 9.99e-41, 5.76e-41, 3.71e-41, 2.02e-41, 1.50e-41, 8.12e-42, 3.93e-42],
    [3.82e-40, 3.34e-40, 2.87e-40, 2.43e-40, 1.82e-40, 9.14e-41, 5.72e-41, 3.29e-41, 1.96e-41, 1.24e-41, 7.53e-42, 3.83e-42],
    [1.98e-40, 2.83e-40, 2.43e-40, 1.99e-40, 1.45e-40, 8.21e-41, 4.76e-41, 2.90e-41, 1.71e-41, 1.09e-41, 6.16e-42, 3.03e-42],
    [2.03e-41, 1.60e-40, 1.75e-40, 1.64e-40, 1.05e-40, 6.18e-41, 3.25e-41, 2.20e-41, 1.61e-41, 9.93e-42, 4.88e-42, 2.92e-42],
    [0.00e+00, 2.82e-41, 1.27e-40, 1.21e-40, 7.30e-41, 4.34e-41, 2.81e-41, 1.91e-41, 1.19e-41, 9.14e-42, 4.03e-42, 2.52e-42],
    [0.00e+00, 0.00e+00, 1.89e-41, 4.27e-41, 3.34e-41, 2.01e-41, 1.48e-41, 1.22e-41, 6.56e-42, 5.23e-42, 3.00e-42, 2.01e-42],
    [0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 5.33e-42, 5.47e-42, 7.22e-42, 5.82e-42, 3.46e-42, 3.22e-42, 1.72e-42, 1.13e-42],
    [0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 1.63e-43, 7.02e-43, 5.31e-43, 3.21e-43, 2.67e-43, 2.16e-43, 2.05e-43]
    ]))
    
Minerva_Error = where(Minerva_Error == 0., inf, Minerva_Error)

## create the 2d array of miniBooNE data for the double fifferential cross section ##
Miniboone_XS = array([[289.2,348.7,418.3,497.6,600.2,692.3,778.1,557.5,891.8,919.3,1003.0,1007.0,992.3,910.2,871.9,765.6,681.9,553.6,401.9,190.0],
    [15.18,25.82,44.84,85.80,135.2,202.2,292.1,401.6,503.3,686.6,813.1,970.2,1148.0,1157.0,1279.0,1233.0,1222.0,981.1,780.6,326.5],
    [0.0,0.0,0.0,0.164,3.624,17.42,33.69,79.10,134.7,272.3,404.9,547.9,850.0,1054.0,1301.0,1495.0,1546.0,1501.0,1258.0,539.2],
    [0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.947,10.92,40.63,92.93,201.5,394.4,628.8,989.9,1289.0,1738.0,1884.0,1714.0,901.8],
    [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.071,2.176,11.63,36.51,105.0,231.0,469.1,872.2,1365.0,1847.0,2084.0,1288.0],
    [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.844,16.96,57.95,147.4,392.3,909.6,1629.0,2100.0,1633.0],
    [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,10.93,10.69,45.02,157.5,526.7,1203.0,2035.0,1857.0],
    [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,12.44,49.23,222.8,723.8,1620.0,1874.0],
    [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.012,9.241,81.65,359.8,1118.0,1803.0],
    [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.229,35.61,156.2,783.6,1636.0],
    [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,4.162,11.36,66.90,451.9,1354.0],
    [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.131,26.87,239.4,1047.0],
    [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.527,116.4,794.0],
    [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,19.50,73.07,687.9],
    [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,41.67,494.3],
    [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,36.55,372.5],
    [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,278.3],
    [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,227.4]])*10**(-41)
       
Miniboone_Error = array([
    [349.8,339.7,408.8,450.2,520.8,596.9,698.0,922.7,917.7,980.4,1090.,1351.,1293.,1267.,1477.,1380.,1435.,1134.,905.0,684.3],
    [63.32,107.6,184.4,236.6,360.7,482.6,553.3,586.4,746.9,783.6,1078.,1246.,1105.,1154.,1273.,1372.,1455.,1557.,1352.,1071.],
    [0.,0.,0.,31.22,34.63,57.73,135.3,215.6,337.5,515.7,695.5,1048.,1041.,1155.,1365.,1434.,1581.,1781.,1754.,1778.],
    [0.,0.,0.,0.,0.,0.,0.,55.88,50.92,114.6,238.2,415.1,742.5,965.3,1369.,1370.,1648.,1845.,2009.,1664.],
    [0.,0.,0.,0.,0.,0.,0.,0.,3.422,20.92,45.96,114.3,250.6,574.7,1021.,1201.,1791.,1769.,2222.,1883.],
    [0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,41.02,77.66,149.2,475.5,870.2,1513.,1823.,2334.,2193.],
    [0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,110.3,53.26,161.6,432.3,1068.,1873.,2711.,2558.],
    [0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,55.58,162.2,598.2,1464.,2870.,3037.],
    [0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,16.32,71.88,267.2,963.8,2454.,3390.],
    [0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,49.10,155.1,601.6,1880.,3320.],
    [0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,54.01,69.28,339.6,1391.,3037.],
    [0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,89.01,184.1,1036.,3110.],
    [0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,170.1,758.7,2942.],
    [0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,230.6,544.3,2424.],
    [0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,505.5,2586.],
    [0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,359.6,2653.],
    [0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,3254.],
    [0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,3838.]])*10**(-42)
                        
Miniboone_Error = where(Miniboone_Error == 0, inf, Miniboone_Error)
  
## create arrays for  the combined XS and XS error ##
total_ddxs = concatenate((Minerva_ddxs_true.ravel(),Miniboone_XS.ravel()))
total_error = concatenate((Minerva_Error.ravel(),Miniboone_Error.ravel()))
       
## Lower edges of the bins ##
p_T_1D_low = array([0.,.075,.15,.25,.325,.4,.475,.55,.7,.85,1.,1.25,1.5])
p_P_1D_low = array([1.5,2.,2.5,3.,3.5,4.,4.5,5.,6.,8.,10.,15.])
## higher edges of the bins ##  
p_T_1D_high = array([.075,.15,.25,.325,.4,.475,.55,.7,.85,1.,1.25,1.5,2.5])
p_P_1D_high = array([2.,2.5,3.,3.5,4.,4.5,5.,6.,8.,10.,15.,20.])
## middle of each bin ##
p_T_1D = (p_T_1D_low + p_T_1D_high)/2.
p_P_1D = (p_P_1D_low + p_P_1D_high)/2.
   
## Miniboone Kinematic variables ##
T_mu_1D = linspace(0.25,1.95,18,endpoint=True) 
cos_mu_1D = linspace(-.95,.95,20,endpoint=True)  
E_nu_1D = linspace(0.05, (60)/20.0,60,endpoint=True) 
T_mu,cos_mu,E_nu = meshgrid(T_mu_1D,cos_mu_1D,E_nu_1D,indexing='ij')
         
num_flux = 500
popt_miniboone,pcov_miniboone = curve_fit(make_double_diff_miniboone,(T_mu,cos_mu,E_nu), Miniboone_XS.ravel(),sigma=(Miniboone_Error.ravel()), bounds=(0.,inf))
print("Finished Calculating First M_A: Time taken = %s seconds" % (time.time() - start_time)) 
num_flux = 500
popt_minerva,pcov_minerva =  curve_fit(flux_interpolate,num_flux,Minerva_ddxs_true.ravel(),sigma=Minerva_Error.ravel(),bounds=(0.,inf))
print("Finished Calculating Second M_A: Time taken = %s seconds" % (time.time() - start_time)) 

double_diff_miniboone = flux_interpolate(num_flux,popt_miniboone[0])
double_diff_miniboone = make2d(T_mu_1D,cos_mu_1D,double_diff_miniboone)
print("Finished Calculating First cross section: Time taken = %s seconds" % (time.time() - start_time))  
double_diff_minerva = flux_interpolate((num_flux,'minerva'),popt_minerva[0])
double_diff_minerva = make2d(p_P_1D,p_T_1D,double_diff_minerva)
print("Finished Calculating Second cross section: Time taken = %s seconds" % (time.time() - start_time))  
 
print(" M_A_miniboone = %s" % popt_miniboone[0])
print(" M_A_minerva = %s" % popt_minerva[0])

M_A = 1.35
M_A_minerva = popt_minerva[0]
M_A_miniboone = popt_miniboone[0]
if M_A == 1.35:
    col = 'red'
else:
    col = 'green' 
    
E_nu_Flux = linspace(0.,20.,40,endpoint=True)     
E_nu_new = linspace(0.,20.,num_flux,endpoint=True)  
    
            
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


double_diff_minerva = where(cos_mu_2D < cos(20.*pi/180), 0., double_diff_minerva)     


total_ddxs_calc = concatenate((double_diff_minerva.ravel(),double_diff_miniboone.ravel()))
  
length_minerva  =  len(Minerva_ddxs_true.ravel())
length_miniboone  =  len(Miniboone_XS.ravel())
length_total = len(total_ddxs_calc)

minerva_chi_sq,minerva_tot_chi_sq =  calc_chi_squared(double_diff_minerva, Minerva_ddxs_true, Minerva_Error)
print ('Minerva chi^2/(d.o.f.) =  %s'  %  round_sig(minerva_tot_chi_sq/length_minerva))    

miniboone_chi_sq,miniboone_tot_chi_sq =  calc_chi_squared(double_diff_miniboone, Miniboone_XS, Miniboone_Error)
print ('MiniBooNE chi^2/(d.o.f.) =  %s'  %  round_sig(miniboone_tot_chi_sq/length_miniboone))

total_chi_sq,total_tot_chi_sq =  calc_chi_squared(total_ddxs_calc, total_ddxs, total_error)
print ('Combined chi^2/(d.o.f.) =  %s'  %  round_sig(total_tot_chi_sq/length_total))
              


#popt,pcov = curve_fit(make_double_diff,(T_mu,cos_mu), Miniboone_XS.ravel(), bounds=([0.,0.],[inf,inf]),sigma=Miniboone_Error.ravel(),absolute_sigma=True)
#print(popt)
#print(pcov)
#perr = sqrt(diag(pcov))
#print(perr)
#double_diff_fit = make_double_diff((T_mu,cos_mu),popt[0],popt[1])
#double_diff_fit = make2d(T_mu,cos_mu,double_diff_fit)
#xi_sq_ind,xi_sq = calc_xi_squared(double_diff_fit,Miniboone_XS,Miniboone_Error)
#chi_sq_ind2,chi_sq2 = calc_chi_squared(double_diff,Miniboone_XS,Miniboone_Error)

#print (chi_sq/(18.*20.-2.))
#print (chi_sq2/(18.*20.-2.))


## Create the plot for the double differential cross section ##    
figax1 = plt.figure()
figax2 = plt.figure()

ax1 = figax1.gca(projection='3d')  
ax1.set_xlabel(r"$p_P$ (GeV)")
ax1.set_ylabel('$p_T$ (GeV)')
ax1.set_zlabel(r"$\frac{d\sigma}{dP_{||} \, dp_T} $")
ax1.set_title(r'Double Differential Cross Section $(cm^2/GeV^2)$')

ax2 = figax2.gca(projection='3d')  
ax2.set_xlabel(r"$T_{\mu}$ (GeV)")
ax2.set_ylabel('$cos\theta_\mu$ ')
ax2.set_zlabel(r"$\frac{d\sigma}{dT_\mu \, dcos\theta_\mu} $")
ax2.set_title(r'Double Differential Cross Section $(cm^2/GeV^2)$')

x,y = meshgrid(p_P_1D,p_T_1D,indexing='ij')
ax1.scatter(x,y,double_diff_minerva,color=col,marker='s',label="RFG Model: M_A = %s GeV" % M_A_minerva)
ax1.scatter(x,y,Minerva_ddxs_true,color='black',marker='s',label="Minerva Neutrino Data",depthshade=False)

x2,y2 = meshgrid(T_mu_1D,cos_mu_1D,indexing='ij')
ax2.scatter(x2,y2,double_diff_miniboone,color=col,marker='s',label="RFG Model: M_A = %s GeV" % M_A_miniboone)
ax2.scatter(x2,y2,Miniboone_XS,color='black',marker='s',label="Miniboone Neutrino Data",depthshade=False)


## Plot the Cross Section ##
#for i in range(len(p_T_1D)-1):
#    for j in range(len(p_P_1D)-1):        
#        xs = p_P_1D[j]
#        ys = p_T_1D[i]
        #zs = double_diff_fit[i][j]
#        zs2 = Minerva_ddxs_true[j][i] 
#        zs3 = double_diff_momenta_3[i][j]           
        #ax1.scatter(ys,xs,zs,color='green',marker='s')
#        ax1.scatter(xs,ys,zs2,color='black',marker='o')
#        ax1.scatter(xs,ys,zs3,color='red',marker='o')
        
ax1.legend(loc=(0.52,0.65))
ax2.legend(loc=(0.52,0.65))

plt.show()

## make an array of strings of the data to put in the table ##
pt = []
pp = []
th = []
Tm = []
EE = []
Minerva_mom = []
Minerva_cos = []
Minerva_mom_RFG = []
Minerva_cos_RFG = []

a=open("Desktop/Research/Axial FF/txt files/ddxs_chi_sq.txt","w+")
a.write("\n chi^2/(d.o.f.):   MiniBooNE = %s,  Minerva = %s, Total = %s    \n \n" % (round_sig(miniboone_tot_chi_sq/length_miniboone),round_sig(minerva_tot_chi_sq/length_minerva),round_sig(total_tot_chi_sq/length_total)))
a.write(" MiniBooNE chi_sq = \n %s \n \n" % miniboone_chi_sq)
a.write(" Minerva chi_sq = \n %s \n" % minerva_chi_sq)

#f=open("Desktop/Research/Axial FF/txt files/cross_section_values_%s.txt" % num_flux_1 ,"w+")
#g=open("Desktop/Research/Axial FF/txt files/cross_section_values_%s.txt" % num_flux_2 ,"w+")
#h=open("Desktop/Research/Axial FF/txt files/cross_section_values_%s.txt" % num_flux_3 ,"w+")
#f.write("   p_T      p_||  true-minerva_ddxs_%s    \n" % num_flux_1)
#g.write("   p_T      p_||  true-minerva_ddxs_%s    \n" % num_flux_2)
#h.write("   p_T      p_||  true-minerva_ddxs_%s    \n" % num_flux_3)
#for i in range(len(p_P_1D)):
#    for j in range(len(p_T_1D)):
#        f.write("   %s   %s   %s   \n" % (p_T_2D[i][j],p_P_2D[i][j],Minerva_ddxs_true[i][j]-double_diff_1[i][j]))
#        g.write("   %s   %s   %s   \n" % (p_T_2D[i][j],p_P_2D[i][j],Minerva_ddxs_true[i][j]-double_diff_2[i][j]))
#        h.write("   %s   %s   %s   \n" % (p_T_2D[i][j],p_P_2D[i][j],Minerva_ddxs_true[i][j]-double_diff_3[i][j]))

#figax1.savefig("Desktop/Research/Axial FF/Plots/Minerva_DDXS_MA%sGeV.pdf" % M_A ) 
print("Time taken = %s seconds" % (time.time() - start_time))        

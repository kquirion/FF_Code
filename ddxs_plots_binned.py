## here  we will show the Q^2 binned  plot of the  ddxs ##

from numpy import array,linspace,transpose,amin,amax,zeros,inf,where,nonzero
import matplotlib.pyplot as plt
from xs_functions_binned import flux_interpolate_binned_mv
from numpy.linalg import inv
from misc_fns import round_sig

len_pp = len(p_P_1D)
len_pt = len(p_T_1D)

## some  parameters ##
N = 200
num_flux = 5000
M_A = 1.32
p_P_1D_smooth =  linspace(0.1,20.,N)
E_nu_flux = linspace(0.,20.,40,endpoint=True)
E_nu_new = linspace(0.,20.,num_flux,endpoint=True)

Q2_bins = array([0.0,0.2,0.4,0.6,0.9,1.2,1.5,2.0,3.,7.5,100.])
y_labels = []
y = zeros((len(Q2_bins)-1,N,len_pt))

## create an array for a ddxs for each Q2 range ##
for i in range(len(Q2_bins)-1):
    y_labels.append(r"%s < $Q^2$ < %s GeV$^2$" % (Q2_bins[i],Q2_bins[i+1]))
    Q2,double_diff,double_diff_3D = flux_interpolate_binned_mv((N,num_flux),M_A,Q2_bins[i],Q2_bins[i+1])
    print "bin %s/%s complete" % (i+1,len(Q2_bins)-1) 
    for j in  range(N):
        for k in range(len_pt):
            y[i,j,k] = double_diff[j,k]

#a,b,c = nonzero(double_diff_3D)
#print a,b,c
#g=open("Desktop/Research/Axial FF/txt files/Q2 values/Q2_values.txt" % p_T_1D[k],"w+")
#g.write("         Q2            E_nu            p_P            p_T            ddxs \n\n")
#for i in range(len(a)):
#    g.write("    %s            %s            %s            %s            %s \n" % (Q2[a[i],b[i],c[i]],E_nu_new[c[i]],p_P_1D_smooth[a[i]],p_T_1D[b[i]],double_diff_3D[a[i],b[i],c[i]]))
#g.close()
## plot ddxs for each value of transverse momentum ##
for k in range(len_pt):         
    temp_str = "%s" % k
    temp_str = plt.figure()
    ax1 = temp_str.gca()
    ax1.set_xlabel(r"$p_{||}$ (GeV)")
    ax1.set_ylabel(r"$\frac{d\sigma}{dP_{||} \, dp_T} \,\, (cm^2/GeV^2) $")
    ax1.set_title(r'Double Differential Cross Section $(cm^2/GeV^2)$')
    ax1.errorbar(p_P_1D,Minerva_ddxs_true[:,k],Minerva_Error[:,k],linestyle='None',marker='s',markersize=3.,color='black',label=r'$p_T=%s$ GeV'  % p_T_1D[k] )
    ax1.set_ylim(amin(Minerva_ddxs_true),3.0*amax(Minerva_ddxs_true[:,k]))
    plt.stackplot(p_P_1D_smooth,y[:,:,k],labels=y_labels,alpha=0.7)
    ax1.legend(loc='best')
    temp_str.savefig("Desktop/Research/Axial FF/Plots/Stacked DDXS/Minerva_ddxs_pt_%s_test.pdf" % p_T_1D[k] )
    plt.close()

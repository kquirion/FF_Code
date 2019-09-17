## here  we will show the Q^2 binned  plot of the  ddxs ##
from numpy import sqrt,meshgrid,array,linspace,transpose,amin,amax,zeros,inf,where,nonzero
import matplotlib.pyplot as plt
from xs_functions_binned import flux_interpolate_binned_mb
from numpy.linalg import inv
from misc_fns import round_sig

#matplotlib inline
 
from IPython.display import set_matplotlib_formats
set_matplotlib_formats('png', 'pdf')

## create the 2d array of miniBooNE data for the double fifferential cross section ##
Miniboone_ddxs = array([[289.2,348.7,418.3,497.6,600.2,692.3,778.1,557.5,891.8,919.3,1003.0,1007.0,992.3,910.2,871.9,765.6,681.9,553.6,401.9,190.0],
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
    [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,227.4]])*10**(-41)*10**(39)

Miniboone_error = array([
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
    [0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,3838.]])*10**(-42)*10**(39)

Miniboone_Error = where(Miniboone_error == 0, inf, Miniboone_error)

T_mu_1D = linspace(0.25,1.95,18,endpoint=True)
cos_mu_1D = linspace(-.95,.95,20,endpoint=True)
E_nu_1D = linspace(0.05, 3.,60,endpoint=True)

len_t = len(T_mu_1D)
len_c = len(cos_mu_1D)

## some  parameters ##
m_mu = .1057
N = 200
num_flux = 360
M_A = 1.05
cos_mu_1D_smooth = linspace(-1.,1.,N)
v = 6 ## version number ##

cos_mu_2D_smooth,T_mu_2D_smooth = meshgrid(cos_mu_1D_smooth,T_mu_1D,indexing='ij')
E_mu_2D = T_mu_2D_smooth + m_mu
P_mu_2D = sqrt(E_mu_2D**2 - m_mu**2)
p_T_2D_smooth = P_mu_2D*(1.-cos_mu_2D_smooth**2)
p_P_2D_smooth = P_mu_2D*cos_mu_2D_smooth
Jac = p_T_2D_smooth/E_mu_2D/P_mu_2D


Q2_bins = array([0.0,0.2,0.4,0.6,0.9,1.2,1.5,100.])
y_labels = []
y = zeros((len(Q2_bins)-1,N,len_t))
y_p = zeros((len(Q2_bins)-1,N,len_t))


## create an array for a ddxs for each Q2 range ##
for i in range(len(Q2_bins)-1):
    y_labels.append(r"%s < $Q^2$ < %s GeV$^2$" % (Q2_bins[i],Q2_bins[i+1]))
    Q2,double_diff,double_diff_3D = flux_interpolate_binned_mb((N,num_flux),M_A,Q2_bins[i],Q2_bins[i+1])
    double_diff_p = double_diff*Jac
    print "bin %s/%s complete" % (i+1,len(Q2_bins)-1) 
    for j in  range(N):
        for k in range(len_t):
            y[i,j,k] = double_diff[j,k]
            y_p[i,j,k] = double_diff_p[j,k]
            
y = y*10.**(39)
y_p = y_p*10.**(39)

#a,b,c = nonzero(double_diff_3D)
#print a,b,c
#g=open("Desktop/Research/Axial FF/txt files/Q2 values/Q2_values.txt" % p_T_1D[k],"w+")
#g.write("         Q2            E_nu            p_P            p_T            ddxs \n\n")
#for i in range(len(a)):
#    g.write("    %s            %s            %s            %s            %s \n" % (Q2[a[i],b[i],c[i]],E_nu_new[c[i]],p_P_1D_smooth[a[i]],p_T_1D[b[i]],double_diff_3D[a[i],b[i],c[i]]))
#g.close()
## plot ddxs for each value of transverse momentum ##

######################################
######################################

fig,axs = plt.subplots(2,3,figsize=(15,6),sharex=True)
ax = fig.add_subplot(111, frameon=False)
ax.spines['top'].set_color('none')
ax.spines['bottom'].set_color('none')
ax.spines['left'].set_color('none')
ax.spines['right'].set_color('none')
ax.tick_params(labelbottom=False, labeltop=False, labelleft=False, labelright=False, top='off', bottom='off', left='off', right='off')
ax = plt.gca()
ax.xaxis.set_label_coords(0.5,-.06)
ax.yaxis.set_label_coords(-.05,0.5)
plt.ylabel(r"$\frac{d\sigma}{dT_{\mu} \, dcos\theta_\mu}(10^{-39} \,\, cm^2/GeV) $", fontsize=14)
plt.xlabel(r"$cos\theta_\mu$ ",fontsize=14)

fig.suptitle(r'Double Differential Cross Section,$\,\,\,M_A = %s \,GeV$' % M_A,y=0.94,fontsize=16)

fig.subplots_adjust(hspace=0.,wspace=0.)
## turn all axis labels off ##
plt.setp([a.get_xticklabels() for a in fig.axes[:]], visible=False)
plt.setp([a.get_yticklabels() for a in fig.axes[:]], visible=False)
## turn select axis  labels on ##
plt.setp([a.get_xticklabels() for a in fig.axes[:]], visible=True)
plt.setp([a.get_yticklabels() for a in fig.axes[::3]], visible=True)
for k in range(len(axs.ravel())):  
    axs.ravel()[k].errorbar(cos_mu_1D,Miniboone_ddxs[k,:],Miniboone_error[k,:],linestyle='None',marker='s',markersize=3.,color='black',label='MiniBooNE Data' )
    axs.ravel()[k].set_ylim(0,25)
    axs.ravel()[k].set_xlim(-1.0,1.0)
    axs.ravel()[k].tick_params(direction='in',length=5)
    plt.setp(axs.ravel()[k].get_yticklabels()[0], visible=False)    
    plt.setp(axs.ravel()[k].get_yticklabels()[-1], visible=False)
    plt.setp(axs.ravel()[k].get_xticklabels()[0], visible=False)    
    plt.setp(axs.ravel()[k].get_xticklabels()[-1], visible=False)
    axs.ravel()[k].stackplot(cos_mu_1D_smooth,y[:,:,k],labels=y_labels,alpha=0.7)
    axs.ravel()[k].legend(loc=(0.,.58),ncol=2,frameon=False,fontsize='x-small',title= r' $T_{\mu}=%s$ GeV' % T_mu_1D[k])
plt.savefig("Desktop/Research/Axial FF/Plots/Stacked DDXS/Mb Stacked DDXS/MB_ddxs_1_ma%s_v%s.pdf" % (M_A,v)  )
plt.close()

######################################
######################################
fig,axs = plt.subplots(2,3,figsize=(15,6),sharex=True)
ax = fig.add_subplot(111, frameon=False)
ax.spines['top'].set_color('none')
ax.spines['bottom'].set_color('none')
ax.spines['left'].set_color('none')
ax.spines['right'].set_color('none')
ax.tick_params(labelbottom=False, labeltop=False, labelleft=False, labelright=False, top='off', bottom='off', left='off', right='off')
ax = plt.gca()
ax.xaxis.set_label_coords(0.5,-.06)
ax.yaxis.set_label_coords(-.05,0.5)
plt.ylabel(r"$\frac{d\sigma}{dT_{\mu} \, dcos\theta_\mu}(10^{-39} \,\, cm^2/GeV) $",fontsize=14)
plt.xlabel(r"$cos\theta_\mu$ ",fontsize=14)
fig.suptitle(r'Double Differential Cross Section, $M_A = %s\,GeV $' % M_A,y=0.94,fontsize=16)

fig.subplots_adjust(hspace=0.,wspace=0.)
## turn all axis labels off ##
plt.setp([a.get_xticklabels() for a in fig.axes[:]], visible=False)
plt.setp([a.get_yticklabels() for a in fig.axes[:]], visible=False)
## turn select axis  labels on ##
plt.setp([a.get_xticklabels() for a in fig.axes[:]], visible=True)
plt.setp([a.get_yticklabels() for a in fig.axes[::3]], visible=True)
for k in range(len(axs.ravel())):  
    l = k+6
    if k < 3:
        axs.ravel()[k].set_ylim(0,30)
        axs.ravel()[k].set_xlim(-.5,1.1)
    else:
        axs.ravel()[k].set_ylim(0,25) 
        axs.ravel()[k].set_xlim(-.5,1.1)
    axs.ravel()[k].tick_params(direction='in',length=5)
    plt.setp(axs.ravel()[k].get_yticklabels()[0], visible=False)    
    plt.setp(axs.ravel()[k].get_yticklabels()[-1], visible=False)
    plt.setp(axs.ravel()[k].get_xticklabels()[0], visible=False)    
    plt.setp(axs.ravel()[k].get_xticklabels()[-1], visible=False)
    axs.ravel()[k].errorbar(cos_mu_1D,Miniboone_ddxs[l,:],Miniboone_error[l,:],linestyle='None',marker='s',markersize=3.,color='black',label='MiniBooNE Data' )
    axs.ravel()[k].stackplot(cos_mu_1D_smooth,y[:,:,l],labels=y_labels,alpha=0.7)
    axs.ravel()[k].legend(loc=(0.,.58),ncol=2,frameon=False,fontsize='x-small',title= r' $T_{\mu}=%s$ GeV' % T_mu_1D[l])
plt.savefig("Desktop/Research/Axial FF/Plots/Stacked DDXS/Mb Stacked DDXS/MB_ddxs_2_ma%s_v%s.pdf" % (M_A,v)  )
plt.close()

######################################
######################################
fig,axs = plt.subplots(2,3,figsize=(15,6),sharex=True)
ax = fig.add_subplot(111, frameon=False)
ax.spines['top'].set_color('none')
ax.spines['bottom'].set_color('none')
ax.spines['left'].set_color('none')
ax.spines['right'].set_color('none')
ax.tick_params(labelbottom=False, labeltop=False, labelleft=False, labelright=False, top='off', bottom='off', left='off', right='off')
ax = plt.gca()
ax.xaxis.set_label_coords(0.5,-.06)
ax.yaxis.set_label_coords(-.05,0.5)
plt.ylabel(r"$\frac{d\sigma}{dT_{\mu} \, dcos\theta_\mu}(10^{-39} \,\, cm^2/GeV) $",fontsize=14)
plt.xlabel(r"$cos\theta_\mu$ ",fontsize=14)
fig.suptitle(r'Double Differential Cross Section, $M_A = %s \,GeV$' % M_A,y=0.94,fontsize=16)

fig.subplots_adjust(hspace=0.,wspace=0.)
## turn all axis labels off ##
plt.setp([a.get_xticklabels() for a in fig.axes[:]], visible=False)
plt.setp([a.get_yticklabels() for a in fig.axes[:]], visible=False)
## turn select axis  labels on ##
plt.setp([a.get_xticklabels() for a in fig.axes[:]], visible=True)
plt.setp([a.get_yticklabels() for a in fig.axes[::3]], visible=True)
for k in range(len(axs.ravel())):  
    l = k+12
    if k < 3:
        axs.ravel()[k].set_ylim(0,15)
        axs.ravel()[k].set_xlim(0.2,1.1)
    else:
        axs.ravel()[k].set_ylim(0,9) 
        axs.ravel()[k].set_xlim(0.2,1.1)
    axs.ravel()[k].tick_params(direction='in',length=5)
    plt.setp(axs.ravel()[k].get_yticklabels()[0], visible=False)    
    plt.setp(axs.ravel()[k].get_yticklabels()[-1], visible=False)
    plt.setp(axs.ravel()[k].get_xticklabels()[0], visible=False)    
    plt.setp(axs.ravel()[k].get_xticklabels()[-1], visible=False)
    axs.ravel()[k].errorbar(cos_mu_1D,Miniboone_ddxs[l,:],Miniboone_error[l,:],linestyle='None',marker='s',markersize=3.,color='black',label='MiniBooNE Data' )
    axs.ravel()[k].stackplot(cos_mu_1D_smooth,y[:,:,l],labels=y_labels,alpha=0.7)
    axs.ravel()[k].legend(loc=(0.,.58),ncol=2,frameon=False,fontsize='x-small',title= r' $T_{\mu}=%s$ GeV' % T_mu_1D[l])
plt.savefig("Desktop/Research/Axial FF/Plots/Stacked DDXS/Mb Stacked DDXS/MB_ddxs_3_ma%s_v%s.pdf" % (M_A,v)  )
plt.close()

#########################################################
#########################################################
## Samenow but  for the one multiplied by the jacobian ##
#########################################################
#########################################################
T_mu_2D,cos_mu_2D = meshgrid(T_mu_1D,cos_mu_1D,indexing='ij')
E_mu_2D = T_mu_2D + m_mu
P_mu_2D = sqrt(E_mu_2D**2 - m_mu**2)
p_T_2D = P_mu_2D*(1.-cos_mu_2D**2)
p_P_2D = P_mu_2D*cos_mu_2D
Jac2 = p_T_2D/E_mu_2D/P_mu_2D

Miniboone_ddxs_p = Miniboone_ddxs*Jac2
Miniboone_error_p =  Miniboone_error*Jac2

fig,axs = plt.subplots(2,3,figsize=(15,6),sharex=True,sharey=False)
ax = fig.add_subplot(111, frameon=False)
ax.spines['top'].set_color('none')
ax.spines['bottom'].set_color('none')
ax.spines['left'].set_color('none')
ax.spines['right'].set_color('none')
ax.tick_params(labelbottom=False, labeltop=False, labelleft=False, labelright=False, top='off', bottom='off', left='off', right='off')
ax = plt.gca()
ax.xaxis.set_label_coords(0.5,-.06)
ax.yaxis.set_label_coords(-.05,0.5)
plt.ylabel(r"$\frac{d\sigma}{dp_T \, dp_{||}}(10^{-39} \,\, cm^2/GeV^2) $",fontsize=14)
plt.xlabel(r"$p_{||}$ (GeV) ",fontsize=14)
fig.suptitle(r'Double Differential Cross Section, $M_A = %s \,GeV$' % M_A,y=0.94,fontsize=16)

fig.subplots_adjust(hspace=0.,wspace=0.)
## turn all axis labels off ##
plt.setp([a.get_xticklabels() for a in fig.axes[:]], visible=False)
plt.setp([a.get_yticklabels() for a in fig.axes[:]], visible=False)
## turn select axis  labels on ##
plt.setp([a.get_xticklabels() for a in fig.axes[:]], visible=True)
plt.setp([a.get_yticklabels() for a in fig.axes[::3]], visible=True)
for k in range(len(axs.ravel())): 
    if k < 3:
        axs.ravel()[k].set_ylim(0,45)
    else:
        axs.ravel()[k].set_ylim(0,25) 
    axs.ravel()[k].tick_params(direction='in',length=5)
    plt.setp(axs.ravel()[k].get_yticklabels()[0], visible=False)    
    plt.setp(axs.ravel()[k].get_yticklabels()[-1], visible=False)
    plt.setp(axs.ravel()[k].get_xticklabels()[0], visible=False)    
    plt.setp(axs.ravel()[k].get_xticklabels()[-1], visible=False)
    axs.ravel()[k].errorbar(p_P_2D[0,:],Miniboone_ddxs_p[k,:],linestyle='None',marker='s',markersize=3.,color='black',label='MiniBooNE Data' )
    axs.ravel()[k].stackplot(p_P_2D_smooth[:,0],y_p[:,:,k],labels=y_labels,alpha=0.7)
    axs.ravel()[k].legend(loc=(0.,.58),ncol=2,frameon=False,fontsize='x-small',title= r' $p_T=%s$ GeV' % round_sig(p_T_2D[k,0]))
plt.savefig("Desktop/Research/Axial FF/Plots/Stacked DDXS/Mb Stacked DDXS/MB_momentum_ddxs_1_ma%s_v%s.pdf" % (M_A,v)  )
plt.close()

#########################################################
#########################################################
## Samenow but  for the one multiplied by the jacobian ##
#########################################################
#########################################################
fig,axs = plt.subplots(2,3,figsize=(15,6),sharex=True)
ax = fig.add_subplot(111, frameon=False)
ax.spines['top'].set_color('none')
ax.spines['bottom'].set_color('none')
ax.spines['left'].set_color('none')
ax.spines['right'].set_color('none')
plt.ylabel(r"$\frac{d\sigma}{dp_T \, dp_{||}}(10^{-39} \,\, cm^2/GeV^2) $",fontsize=14)
plt.xlabel(r"$p_{||}$ (GeV)",fontsize=14)
ax.tick_params(labelbottom=False, labeltop=False, labelleft=False, labelright=False, top='off', bottom='off', left='off', right='off')
ax = plt.gca()
ax.xaxis.set_label_coords(0.5,-.06)
ax.yaxis.set_label_coords(-.05,0.5)
fig.suptitle(r'Double Differential Cross Section, $M_A = %s \,GeV $' % M_A,y=0.94,fontsize=16)

fig.subplots_adjust(hspace=0.,wspace=0.)
## turn all axis labels off ##
plt.setp([a.get_xticklabels() for a in fig.axes[:]], visible=False)
plt.setp([a.get_yticklabels() for a in fig.axes[:]], visible=False)
## turn select axis  labels on ##
plt.setp([a.get_xticklabels() for a in fig.axes[:]], visible=True)
plt.setp([a.get_yticklabels() for a in fig.axes[::3]], visible=True)
for k in range(len(axs.ravel())):  
    l = k+6
    if k < 3:
        axs.ravel()[k].set_ylim(0,11)
        axs.ravel()[k].set_xlim(.1,.35)
    else:
        axs.ravel()[k].set_ylim(0,2.5)
        axs.ravel()[k].set_xlim(.1,.35)
    axs.ravel()[k].tick_params(direction='in',length=5)
    plt.setp(axs.ravel()[k].get_yticklabels()[0], visible=False)    
    plt.setp(axs.ravel()[k].get_yticklabels()[-1], visible=False)
    plt.setp(axs.ravel()[k].get_xticklabels()[0], visible=False)    
    plt.setp(axs.ravel()[k].get_xticklabels()[-1], visible=False)
    axs.ravel()[k].errorbar(p_P_2D[0,:],Miniboone_ddxs_p[l,:],linestyle='None',marker='s',markersize=3.,color='black',label='MiniBooNE Data' )
    axs.ravel()[k].stackplot(p_P_2D_smooth[:,0],y_p[:,:,l],labels=y_labels,alpha=0.7)
    axs.ravel()[k].legend(loc=(0.,.58),ncol=2,frameon=False,fontsize='x-small',title= r' $p_T=%s$ GeV' % round_sig(p_T_2D[l,0]))
plt.savefig("Desktop/Research/Axial FF/Plots/Stacked DDXS/Mb Stacked DDXS/MB_momentum_ddxs_2_ma%s_v%s.pdf" % (M_A,v)  )
plt.close()

#########################################################
#########################################################
## Samenow but  for the one multiplied by the jacobian ##
#########################################################
#########################################################
fig,axs = plt.subplots(2,3,figsize=(15,6),sharex=True)
ax = fig.add_subplot(111, frameon=False)
ax.spines['top'].set_color('none')
ax.spines['bottom'].set_color('none')
ax.spines['left'].set_color('none')
ax.spines['right'].set_color('none')
ax.tick_params(labelbottom=False, labeltop=False, labelleft=False, labelright=False, top='off', bottom='off', left='off', right='off')
ax = plt.gca()
ax.xaxis.set_label_coords(0.5,-.06)
ax.yaxis.set_label_coords(-.05,0.5)
plt.ylabel(r"$\frac{d\sigma}{dp_T \, dp_{||}}(10^{-39} \,\, cm^2/GeV^2)$",fontsize=14)
plt.xlabel(r"$p_{||}$ (GeV)",fontsize=14)
fig.suptitle(r'Double Differential Cross Section, $M_A = %s \,GeV$' % M_A,y=0.94,fontsize=16)

## turn all axis labels off ##
plt.setp([a.get_xticklabels() for a in fig.axes[:]], visible=False)
plt.setp([a.get_yticklabels() for a in fig.axes[:]], visible=False)
## turn select axis  labels on ##
plt.setp([a.get_xticklabels() for a in fig.axes[:]], visible=True)
plt.setp([a.get_yticklabels() for a in fig.axes[::3]], visible=True)

fig.subplots_adjust(hspace=0.,wspace=0.)
for k in range(len(axs.ravel())):  
    l = k+12
    if k < 3:
        axs.ravel()[k].set_ylim(0.,.7)
        axs.ravel()[k].set_xlim(0.,.35)
    else:
        axs.ravel()[k].set_ylim(0,.25)
        axs.ravel()[k].set_xlim(0.,.35)
    axs.ravel()[k].tick_params(direction='in',length=5)
    plt.setp(axs.ravel()[k].get_yticklabels()[0], visible=False)    
    plt.setp(axs.ravel()[k].get_yticklabels()[-1], visible=False)
    plt.setp(axs.ravel()[k].get_xticklabels()[0], visible=False)    
    plt.setp(axs.ravel()[k].get_xticklabels()[-1], visible=False)
    axs.ravel()[k].errorbar(p_P_2D[0,:],Miniboone_ddxs_p[l,:],linestyle='None',marker='s',markersize=3.,color='black',label='MiniBooNE Data' )
    axs.ravel()[k].stackplot(p_P_2D_smooth[:,0],y_p[:,:,l],labels=y_labels,alpha=0.7)
    axs.ravel()[k].legend(loc=(0.,.58),ncol=2,frameon=False,fontsize='x-small',title= r' $p_T=%s$ GeV' % round_sig(p_T_2D[l,0]))
plt.savefig("Desktop/Research/Axial FF/Plots/Stacked DDXS/Mb Stacked DDXS/MB_momentum_ddxs_3_ma%s_v%s.pdf" % (M_A,v)  )
plt.close()



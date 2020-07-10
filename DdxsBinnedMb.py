## here  we will show the Q^2 binned  plot of the  ddxs ##
from numpy import sqrt,meshgrid,array,linspace,transpose,amin,amax,zeros,inf,where,nonzero
import matplotlib.pyplot as plt
from XsFunctionsBinned import DdxsBinnedMb
from numpy.linalg import inv
from MiscFns import roundSig

#matplotlib inline
 
from IPython.display import set_matplotlib_formats
set_matplotlib_formats('png', 'pdf')

len_t = len(T_mu_1D)
len_c = len(cos_mu_1D)

## some  parameters ##
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
    print ("bin %s/%s complete" % (i+1,len(Q2_bins)-1) )
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



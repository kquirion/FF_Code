## here  we will show the Q^2 binned  plot of the  ddxs ##
from numpy import sqrt,meshgrid,array,linspace,transpose,amin,amax,zeros,inf,where,nonzero
import matplotlib.pyplot as plt
from XsFunctionsBinned import DdxsBinnedSmoothMb
from numpy.linalg import inv
from MiscFns import roundSig,WeightSum3D
from DataFile import *
from pylab import subplot

#matplotlib inline
 
from IPython.display import set_matplotlib_formats
set_matplotlib_formats('png', 'pdf')

len_t = len(Tmu1DMb)
len_c = len(CosMu1DMb)

## some  parameters ##
N = 10
numFlux = 360
MA = 1.35
CosMu1DSmooth = linspace(-1.,1.,N)
v = 1 ## version number ##

CosMu2DSmooth,Tmu2DSmooth = meshgrid(CosMu1DSmooth,Tmu1DMb,indexing='ij')
Emu2D = Tmu2DSmooth + mMu
Pmu2D = sqrt(Emu2D**2 - mMu**2)
pT2DSmooth = Pmu2D*(1.-CosMu2DSmooth**2)
pP2DSmooth = Pmu2D*CosMu2DSmooth
Jac = pT2DSmooth/Emu2D/Pmu2D

Q2Bins = array([0.0,0.2,0.4,0.6,0.9,1.2,1.5,100.])
yLabels = []
y = zeros((len(Q2Bins)-1,N,len_t))
## create an array for a ddxs for each Q2 range ##
<<<<<<< Updated upstream
for i in range(len(Q2_bins)-1):
    y_labels.append(r"%s < $Q^2$ < %s GeV$^2$" % (Q2_bins[i],Q2_bins[i+1]))
    Q2,double_diff,double_diff_3D = flux_interpolate_binned_mb((N,num_flux),M_A,Q2_bins[i],Q2_bins[i+1])
    double_diff_p = double_diff*Jac
    print ("bin %s/%s complete" % (i+1,len(Q2_bins)-1) )
=======
for i in range(len(Q2Bins)-1):
    yLabels.append(r"%s $\leq$ $Q^2$ < %s GeV$^2$" % (Q2Bins[i],Q2Bins[i+1]))
    Q2,DoubleDiff,DoubleDiff3D = DdxsBinnedSmoothMb(N,MA,Q2Bins[i],Q2Bins[i+1])
    print("bin %s/%s complete" % (i+1,len(Q2Bins)-1) )
>>>>>>> Stashed changes
    for j in  range(N):
        for k in range(len_t):
            y[i,j,k] = DoubleDiff[j,k]          
y = y*10.**(39)
y = np.swapaxes(y, 1, 2 )
y = np.swapaxes(y, 0, 1 )
MinibooneDdxs *= 1e39
######################################
######################################
NumRows = 2
NumCols = 3

"""
plt.subplots_adjust(hspace=.5)
for i in range( NumRows * NumCols ):
    ax1 = subplot(NumRows, NumCols, i+1, snap=False)
    ax1.scatter( CosMu1DMb, MinibooneDdxs[i], color = 'k', label='MiniBooNE Data')
    ax1.stackplot( CosMu1DSmooth, y[i], alpha=0.7)
    ax1.set_ylim(0., 1.5 * np.nanmax(MinibooneDdxs[i]) )
    ax1.legend(title=r'$T_\mu =$ %s GeV' % roundSig(Tmu1DMb[i]))
ax1.set_xlabel(r'$\cos\theta_\mu$', position=(-.695,0), fontsize='xx-large')
ax1.set_title(r'$\frac{d^2\sigma}{dT_\mu c\cos\theta_\mu}$', position=(-2.53,.95), rotation=90, fontsize='xx-large' )
plt.show()
"""
f1 = plt.figure(figsize=(4,4.5))
f1.suptitle("Legend")
ax1 = f1.add_subplot(111)
ax1.stackplot( CosMu1DSmooth, y[0], alpha=0.7, labels=yLabels)
h,l = ax1.get_legend_handles_labels()
f1.clear()
ax1 = f1.add_subplot(111)
ax1.set_axis_off()
ax1.legend(handles=h, labels=l, title=r"$Q^2$ Color Coding", title_fontsize='xx-large', fontsize='large' )



f = plt.figure(figsize=(12,5))
f.suptitle(r"Double Differential Cross Section With $M_A = %s$ GeV " % MA, fontsize='xx-large')
for i in range( NumRows * NumCols ):
    ax = f.add_subplot( NumRows, NumCols, i+1 )
    ax.scatter( CosMu1DMb, MinibooneDdxs[i], color = 'k')
    ax.stackplot( CosMu1DSmooth, y[i], alpha=0.7)
    ax.set_ylim(0., 1.5 * np.nanmax(MinibooneDdxs[i]) )
    ax.legend(title=r'$T_\mu =$ %s GeV' % roundSig(Tmu1DMb[i]), loc='upper left', frameon=False)
ax.set_xlabel(r'$\cos\theta_\mu$', position=(-.695,-0.1), fontsize='x-large')
ax.set_title(r'$\frac{d^2\sigma}{dT_\mu c\cos\theta_\mu}$', position=(-2.65,.95), rotation=90, fontsize='xx-large' )
plt.show()

"""
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

fig.suptitle(r'Double Differential Cross Section,$\,\,\,MA = %s \,GeV$' % MA,y=0.94,fontsize=16)

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
    axs.ravel()[k].stackplot(CosMu1DSmooth,y[:,:,k],labels=yLabels,alpha=0.7)
    axs.ravel()[k].legend(loc=(0.,.58),ncol=2,frameon=False,fontsize='x-small',title= r' $T_{\mu}=%s$ GeV' % Tmu1DMb[k])
plt.savefig("Desktop/Research/Axial FF/Plots/Stacked DDXS/Mb Stacked DDXS/MB_ddxs_1_ma%s_v%s.pdf" % (MA,v)  )
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
fig.suptitle(r'Double Differential Cross Section, $MA = %s\,GeV $' % MA,y=0.94,fontsize=16)

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
    axs.ravel()[k].stackplot(CosMu1DSmooth,y[:,:,l],labels=yLabels,alpha=0.7)
    axs.ravel()[k].legend(loc=(0.,.58),ncol=2,frameon=False,fontsize='x-small',title= r' $T_{\mu}=%s$ GeV' % Tmu1DMb[l])
plt.savefig("Desktop/Research/Axial FF/Plots/Stacked DDXS/Mb Stacked DDXS/MB_ddxs_2_ma%s_v%s.pdf" % (MA,v)  )
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
fig.suptitle(r'Double Differential Cross Section, $MA = %s \,GeV$' % MA,y=0.94,fontsize=16)

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
    axs.ravel()[k].stackplot(CosMu1DSmooth,y[:,:,l],labels=yLabels,alpha=0.7)
    axs.ravel()[k].legend(loc=(0.,.58),ncol=2,frameon=False,fontsize='x-small',title= r' $T_{\mu}=%s$ GeV' % Tmu1DMb[l])
plt.savefig("Desktop/Research/Axial FF/Plots/Stacked DDXS/Mb Stacked DDXS/MB_ddxs_3_ma%s_v%s.pdf" % (MA,v)  )
plt.close()

"""





## here  we will show the Q^2 binned  plot of the  ddxs ##

from numpy import array,linspace,transpose,amin,amax,zeros,inf,where,nonzero
import matplotlib.pyplot as plt
from XsFunctionsBinned import DdxsBinnedMv
from numpy.linalg import inv
from MiscFns import roundSig
from DataFile import *

len_pp = len(pP1D)
len_pt = len(pT1D)

## some  parameters ##
N = 200
numFlux = 5000
MA = 1.32
pP1DSmooth =  linspace(0.1,20.,N)
EnuFlux = linspace(0.,20.,40,endpoint=True)
EnuNew = linspace(0.,20.,numFlux,endpoint=True)

Q2Bins = array([0.0,0.2,0.4,0.6,0.9,1.2,1.5,2.0,3.,7.5,100.])
yLabels = []
y = zeros((len(Q2Bins)-1,N,len_pt))

## create an array for a ddxs for each Q2 range ##
for i in range(len(Q2Bins)-1):
    yLabels.append(r"%s < $Q^2$ < %s GeV$^2$" % (Q2Bins[i],Q2Bins[i+1]))
    Q2,DoubleDiff,DoubleDiff3D = DdxsBinnedMv(N,numFlux,MA,Q2Bins[i],Q2Bins[i+1])
    print("bin %s/%s complete" % (i+1,len(Q2Bins)-1) )
    for j in  range(N):
        for k in range(len_pt):
            y[i,j,k] = DoubleDiff[j,k]

## plot ddxs for each value of transverse momentum ##
for k in range(len_pt):         
    TempStr = "%s" % k
    TempStr = plt.figure()
    ax1 = TempStr.gca()
    ax1.set_xlabel(r"$p_{||}$ (GeV)")
    ax1.set_ylabel(r"$\frac{d\sigma}{dP_{||} \, dp_T} \,\, (cm^2/GeV^2) $")
    ax1.set_title(r'Double Differential Cross Section $(cm^2/GeV^2)$')
    ax1.errorbar(pP1D,MinervaDdxsTrue[:,k],MinervaError[:,k],linestyle='None',marker='s',markersize=3.,color='black',label=r'$p_T=%s$ GeV'  % pT1D[k] )
    ax1.set_ylim( amin( MinervaDdxsTrue ), 3. * amax( MinervaDdxsTrue[:,k] ) )
    plt.stackplot( pP1DSmooth,y[:,:,k], labels=yLabels , alpha=0.7 )
    ax1.legend(loc='best')
    TempStr.savefig("Desktop/Research/Axial FF/Plots/Stacked DDXS/Minerva_ddxs_pt_%s_test.pdf" % pT1D[k] )
    plt.close()

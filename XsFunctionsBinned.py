## XS_functions_binned  ##
## Here we have the functions for checking how total cross section depends on the range of Q^2 being used ##
from math import pi
import numpy as np 
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from XsFunctionsDipole import (aElements,FormFactorsDipole,
                               CalcCrossSection,DdxsDipole)
from MiscFns import WeightSum3D,roundSig
from VariableFns import Variables,VariablesUnbinned,Variables3D
from DataFile import *

## Interpolate the flux function that we are  integrating over to calculate a better ddxs  ##
def DdxsBinnedSmoothMv(N,MA,lower,upper):
    numFlux=100
    pP1d  = np.linspace(0.05,20.,N)
    ## Interpolate the flux data ##
    EnuFlux = np.linspace(0.,20.,len(FluxMv))
    Func = interp1d(EnuFlux,Flux,kind='cubic')
    EnuNew = np.linspace(0.1,20.,numFlux)
    FluxNew = Func(EnuNew)    
    
    Enu3D = np.broadcast_to(EnuNew,(N,2*N,numFlux))

    TotalFlux = 0.
    TotalFlux = 20.*sum(FluxNew)/numFlux

    ## Define the weight functions needed to integrate over the flux ##
    weight = []
    [weight.append( (20./numFlux)*flux/TotalFlux/12.) for flux in FluxNew]
    
    ###############################################
    ## define the kinematic inputs for each case ##
    ###############################################    
    pP3D,pT3D,Enu3D = np.meshgrid(pP1d,pT1D,EnuNew,indexing='ij')
    Pmu3D = np.sqrt(pP3D**2 + pT3D**2)
    Emu3D = np.sqrt(Pmu3D**2 + mMu**2)
    CosMu3D = pP3D/Pmu3D
    
    Jac = pT3D/Emu3D/Pmu3D
    
    Q2 = 2.*Enu3D*(Emu3D - Pmu3D*CosMu3D) - mMu**2
    
    weight = np.broadcast_to(weight,(N,2*N,len(FluxNew)))
    
    ######################################################
    ## make double diff from fit to total cross section ##
    ######################################################
    DoubleDiff3D = DdxsBinned(Emu3D,Enu3D,Pmu3D,CosMu3D,MA,lower,upper)
    DoubleDiff3D  = DoubleDiff3D*Jac
    DoubleDiff3DTemp = np.where(CosMu3D <= np.cos(angleCut*pi/180.), 0., DoubleDiff3D)
    
    EmuMax = Enu3D + Ehi - mN
    CosMax = Emu3D/Pmu3D - mMu**2/(2.0*Enu3D*Pmu3D)
    
    DoubleDiff3DTemp = np.where(Emu3D <= EmuMax, DoubleDiff3DTemp,0.)
    DoubleDiff3DTemp = np.where(CosMu3D <= CosMax, DoubleDiff3DTemp,0.)  
    DoubleDiff2D = WeightSum3D(DoubleDiff3DTemp,weight)
    
    return Q2,DoubleDiff2D,DoubleDiff3D
    
## Interpolate the flux function that we are  integrating over to calculate a better ddxs  ##
def DdxsBinnedSmoothMb(N,MA,lower,upper):
    numFlux=200
    CosMu1d  = np.linspace(-1.,1.,N)                           
    ## Interpolate the flux data ##
    Func = interp1d(EnuFluxMb,FluxMb,kind='cubic')
    EnuNew = np.linspace(0.05,3.,numFlux)
    FluxNew = Func(EnuNew)    
    Enu3D = np.broadcast_to(EnuNew,(N,2*N,numFlux))
    TotalFlux = 0.
    TotalFlux = 3.*sum(FluxNew)/numFlux
    ## Define the weight functions needed to integrate over the flux ##
    weight = []
    [weight.append( (3./numFlux)*flux/TotalFlux/6.) for flux in FluxNew]
    ###############################################
    ## define the kinematic inputs for each case ##
    ###############################################   
    CosMu2D,Tmu2D = np.meshgrid(CosMu1d,Tmu1DMb,indexing='ij')
    Emu2D = Tmu2D + mMu
    Pmu2D = np.sqrt(Emu2D**2 - mMu**2)
    pT2D = Pmu2D*CosMu2D
    pP2D = Pmu2D*np.sqrt(1.-CosMu2D**2)
    Jac = pT2D/Emu2D/Pmu2D
    CosMu3D,Tmu3D,Enu3D = np.meshgrid(CosMu1d,Tmu2D,EnuNew,indexing='ij')
    Emu3D = Tmu3D + mMu
    Pmu3D = np.sqrt(Emu3D**2 - mMu**2)
    Q2 = 2.*Enu3D*(Emu3D - Pmu3D*CosMu3D) - mMu**2
    weight = np.broadcast_to(weight,(N,N*len(Tmu1DMb),len(FluxNew)))
    ######################################################
    ## make double diff from fit to total cross section ##
    ######################################################
    DoubleDiff3D = DdxsBinned(Emu3D,Enu3D,Pmu3D,CosMu3D,MA,lower,upper)
    #DoubleDiff3DTemp = where(CosMu3D <= cos(angleCut*pi/180.), 0., DoubleDiff3D)
    EmuMax = Enu3D + Ehi - mN
    CosMax = Emu3D/Pmu3D - mMu**2/(2.0*Enu3D*Pmu3D)
    DoubleDiff3DTemp = np.where(Emu3D <= EmuMax, DoubleDiff3D,0.)
    DoubleDiff3DTemp = np.where(CosMu3D <= CosMax, DoubleDiff3DTemp,0.)  
    DoubleDiff2D = WeightSum3D(DoubleDiff3DTemp,weight)
    #DoubleDiff2D = DoubleDiff2D*Jac
    
    return Q2,DoubleDiff2D,DoubleDiff3D
    
##############################################
## Create double differential cross section ##
##############################################
def DdxsBinned(Emu,Enu,Pmu,CosMu,MA,lower,upper):
    ## parameters ##
    A = 12                                                  # number of Nucleons
    mT = A*(mN-eB)                                       # mass of target Nucleus
    ## fill in the Q^2 = -q^2 values ##
    Q2 = 2.0*Emu*Enu - 2.0*Enu*Pmu*CosMu - mMu**2
    ## fill in values of the W Boson Energy ##
    w = Enu - Emu
    ## fill in the values of the W boson energy ##
    wEff = w - eB
    ## fill in the values for the 3-momentum of the Boson ##
    q = np.sqrt( Q2 + w**2 )
    ## calculate the a elements ##
    a1,a2,a3,a4,a5,a6,a7 = aElements(Q2,q,w,wEff)
    ## calculate the form factors ##
    F1,F2,FA,FP = FormFactorsDipole(Q2,MA)
    ## Use the Form Factors to Define the H Elements ##
    H1 = 8. * mN**2* FA**2 + 2. * Q2 * ( ( F1 + F2 )**2 + FA**2 )
    H2 = 8. * mN**2 * ( F1**2 + FA**2 ) + 2. * Q2 * F2**2
    H3 = -16. * mN**2 * FA * ( F1 + F2 )
    H4 = ( Q2 / 2. * (  F2**2 + 4. * FP**2 ) - 2. * mN**2 * F2**2 
          - 4. * mN**2 * ( F1 * F2 + 2. * FA * FP ) )
    H5 = 8. * mN**2 * ( F1**2 + FA**2 ) + 2. * Q2 * F2**2
    ## Use the a and H values to determine the W values ##
    W1 = a1 * H1 + .5 * ( a2 - a3 ) * H2
    W2 = ( ( a4 + ( w / q )**2 * a3 - 2. * ( w / q ) * a5 
            + .5 * ( ( 1 - ( w / q )**2 ) * ( a2 - a3) ) ) * H2 )
    W3 = mT / mN * ( a7 - ( w / q ) * a6 ) * H3
    W4 = ( ( mT / mN )**2 * ( a1 * H4 + mN * a6 * H5 / q 
            + ( mN**2 / 2. ) * ( ( 3. * a3 - a2 ) * H2 ) / q**2 ) )
    W5 = ( mT / mN * ( a7 - w / q * a6 ) * H5 
          + mT * ( 2. * a5 + w / q * ( a2 - 3. * a3 ) ) * ( H2 / q ) )
    DoubleDiff = ( ( GF**2 * Pmu * Vud**2 ) / ( 16. * pi**2 * mT * ( GeVToCm**2 ) )
                  * ( 2. * ( Emu - Pmu * CosMu ) * W1 + ( Emu + Pmu * CosMu ) * W2 
        + 1. / mT * ( ( Emu - Pmu * CosMu ) * ( Enu + Emu ) - mMu**2 ) * W3 
        + ( mMu / mT )**2 * ( Emu - Pmu * CosMu ) * W4 
        - ( mMu**2 / mT ) * W5 ) )
    DoubleDiff = np.where(lower <= Q2, DoubleDiff, 0.)
    DoubleDiff = np.where(upper >= Q2, DoubleDiff, 0.)
    #DoubleDiff = np.where(CosMu < cos(20.*pi/180), 0., DoubleDiff)
    return DoubleDiff

###################################
## function to do the dipole fit ##
###################################
def XsBinned(Enu,MA):
    N = len(Enu)
    NCosMax = 200 + int(amax(Enu)+1)*800
    NTMax = 100+20*int(amax(Enu))
    Tmu,Emu,Pmu,Enu,CosMu,DELTACosMu,DELTATmu = VariablesUnbinned(NTMax,NCosMax,Enu[N-1])
    Q2 = 2. * Emu * Enu - 2. * Enu * Pmu * CosMu - mMu**2
    print(amin(Q2))
    BinEdges = np.array([0.0,0.2,0.4,0.6,0.9,1.2,1.5,2.0,3.,7.5,roundSig(amax(Q2))])
    NumQ2 = len(BinEdges)
    Elow  = -1
    Ehigh = np.log10(20.)
    SigmaTot = np.zeros(200)
    y = []
    YLabels = []
    k = 0
    while k < NumQ2-1:
        Sigma = np.zeros(N)
        for m  in range(N):
            print(" %s%% complete" % (1.*(k*len(Enu) + m)/(1.*(NumQ2-1)*len(Enu))*100.))
            NCos = 100 + int( Enu[m] + 1 ) * 800
            NT = 50 + 30 * int( Enu[m] )
            BinSize = int( 2 * NCos / 100 )
            NumBins = int( 2 * NCos / BinSize )
            CosBin =np.array( [ [ 0.0 for j in range(BinSize) ] for i in range(NT)])
            ################################
            ## Create Kinematic Variables ##
            ################################
            Tmu,Emu,Pmu,Enu,CosMu,DELTACosMu,DELTATmu = Variables(NT,NCos,Enu[m])
            for l in range(NumBins):
                for i in range(NT):
                    for j in range(BinSize):
                        CosBin[i][j] = CosMu[i][j+l*BinSize]
                #####################################################################
                ## Create RFG Variables #############################################
                ## For the last entry, put 1 for neutrinos, or 2 for antineutrinos ##
                #####################################################################
                DoubleDiff = DdxsBinned(Emu,Enu,Pmu,CosBin,MA,BinEdges[k],BinEdges[k+1])
                ## apply the angle cut of Minerva ##
                #DoubleDiff = where((arccos(CosBin)*180/pi <= 20) & (arccos(CosBin)*180/pi >= -20), DoubleDiff, 0.)
                #############################
                ## Calculate Cross Section ##
                #############################
                SigmaTemp,NewEnu = CalcCrossSection(Enu,NT,int(NCos/100),DELTACosMu,DELTATmu,DoubleDiff)
                #################################
                ## Add to total value of Sigma ##
                #################################
                Sigma[m] += SigmaTemp
        ###############################################
        ## plot the contribution of  each Q^2  range ##
        ###############################################
        Func = interp1d(Enu,Sigma,kind='cubic')
        NewerEnu = np.logspace(Elow,Ehigh,200)
        SigmaNew = Func(NewerEnu)
        ## uncomment these below if  you want to  get a rough idea  what the graph should look like, with lower number of T_mu, CosMu points ##
        #for j in range(100):
        #    for i in range(len(NewerEnu)):
        #        if NewerEnu[i] >= 9.:
        #            SigmaNew[i] = SigmaNew[i-1]
        y.append(SigmaNew)
        YLabels.append(r"%s < $Q^2$ < %s " % (BinEdges[k],BinEdges[k+1]))
        SigmaTot  += SigmaNew
        k += 1
    #Sigma_old = make_total_xs_dipole(Enu,MA)
    fig = plt.figure()
    SigmaGraph = fig.gca()
    SigmaGraph.set_xlabel(r'$E_{\nu}$ ($GeV$)')
    SigmaGraph.set_ylabel(r'$\sigma$ ($cm^2$)')
    SigmaGraph.set_title(r'Neutrino Cross Section: $MA = %s GeV$ ' % MA, y=1.05)
    SigmaGraph.set_xlim(0.1,20.0)
    SigmaGraph.set_ylim(0.0,2.0*10**(-38))
    SigmaGraph.set_xscale('log')
    if MA == 1.05:
        col = 'green'
    elif MA == 1.35:
        col = 'red'
    elif MA ==  1.45:
        col =  'cyan'
    SigmaGraph.stackplot(NewerEnu,y,linestyle='-',linewidth=2,labels=YLabels)
    SigmaGraph.plot(NewerEnu,SigmaTot,color=col,linestyle='-')
    SigmaGraph.errorbar(MinibooneXData,MinibooneXs,yerr=MinibooneError,marker='s',color='black',fmt='o',label='Miniboone XS')
    #SigmaGraph.errorbar(MinervaXData,MinervaXS,yerr=MinervaError,marker='s',color='m',fmt='o',label='Minerva XS')
    SigmaGraph.errorbar(NomadXData,NomadXs,yerr=NomadError,marker='s',color='grey',fmt='o',label='Nomad XS')
    chartBox = SigmaGraph.get_position()
    SigmaGraph.set_position([chartBox.x0, chartBox.y0, chartBox.width*0.9, chartBox.height])
    SigmaGraph.legend(loc='upper center', title = r"$Q^2$ in $GeV^2$", bbox_to_anchor=(1.12, 1.), shadow=True, ncol=1, prop={'size': 6})
    fig.savefig("Desktop/Research/Axial FF/Plots/Q2 Conts 2./Q2_Stacks_%s_v6..pdf" % MA )


    return SigmaTot

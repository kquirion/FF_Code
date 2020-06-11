## This file contains many functions for running cross section calculations ##
import numpy as np
from math import pi
from sys import exit
from scipy.interpolate import interp1d
from MiscFns import WeightSum3d,RoundSig
from VariableFns import MakeVariables
from DataFile import *

np.set_printoptions(precision=3)

## Interpolate the flux function that we are  integrating over to calculate a better ddxs  ##
def DdxsDipoleSmooth(N,numFlux,MA):

    len_pt = len(pT1D)
    len_pp = len(pP1D)
    
    ## create upper and lower bounds on bins within each experimental bin ##
    pT1d = np.array([np.linspace(pT1DLow[i],pT1DHigh[i],N) for i in range(len_pt)])
    pP1d = np.array([np.linspace(pP1DLow[i],pP1DHigh[i],N) for i in range(len_pp)])
    
    ## Interpolate the flux data ##
    EnuFlux = np.linspace(0.,20.,len(FluxMv))
    Func = interp1d(EnuFlux,FluxMv,kind='cubic')
    EnuNew = np.linspace(0.,20.,numFlux)
    FluxNew = Func(EnuNew)

    TotalFlux = 0
    TotalFlux = 20 * sum(FluxNew) / numFlux

    ## Define the weight functions needed to integrate over the flux ##
    weight = []
    [ weight.append( 20 * flux / numFlux  / TotalFlux ) for flux in FluxNew ]

    ## an array to hold the different ddxs arrays to be averaged over ##
    ddxsHolder = np.zeros((N,len_pp*len_pt))
    
    ###############################################
    ######## calculation for eacch sub-bin ########
    ###############################################
    for o in range(N):
        pP2D,pT2D = np.meshgrid(pP1d[:,o],pT1d[:,o],indexing='ij')
        CosMu2D = pP2D /  np.sqrt( pP2D**2 + pT2D**2 )
        Tmu2D = np.sqrt( pP2D**2 + pT2D**2 + mMu**2 ) - mMu
        Jac = pT2D / ( Tmu2D + mMu ) / np.sqrt( pP2D**2 + pT2D**2 )
    
        pP3D,pT3D,Enu3D = np.meshgrid(pP1d[:,o],pT1d[:,o],EnuNew,indexing = 'ij')
        Tmu3D = np.sqrt( pP3D**2 + pT3D**2 + mMu**2 ) - mMu
        CosMu3D = pP3D / np.sqrt( pT3D**2 + pP3D**2 )
        Emu3D = Tmu3D + mMu
        Pmu3D = np.sqrt( pT3D**2 + pP3D**2 )
    
        weight = np.broadcast_to(weight,(len_pp,len_pt,len(FluxNew)))
    
        ######################################################
        ## make double diff from fit to total cross section ##
        ######################################################
        doubleDiff = DdxsDipole(Emu3D,Enu3D,Pmu3D,CosMu3D,MA,1)
    
        #######################################################################
        ## find the bounds on the indices where the cross section is nonzero ##
        #######################################################################
        a = np.empty([len_pp,len_pt], dtype=int)
        b = np.empty([len_pp,len_pt],dtype=int)
        for i in range(len_pp):
            for j in range(len_pt):
                A = 0
                B = numFlux-1
                while doubleDiff[i,j,A] == 0.:
                    A += 1
                    if A ==  numFlux:
                        break
                while  doubleDiff[i,j,B] == 0. :
                    B -= 1
                    if B == 0:
                        break
                a[i,j] = A-1
                b[i,j] = B+1
        b = np.where( b == numFlux, numFlux-1, b)
    
        ## Find new ranges of flux for each combo of p_T and p_|| ##
        EvenNewerEnu = np.empty([len_pp,len_pt,numFlux])
        for i in range(len_pp):
            for j in range(len_pt):
                tempFlux = np.linspace(EnuNew[a[i,j]],EnuNew[b[i,j]],numFlux)
                for k in range(numFlux):
                    EvenNewerEnu[i,j,k] = tempFlux[k]
        ## Create new weight funcntions for each combo of p_T and p_|| ##
        NewerFluxNew = Func(EvenNewerEnu)
        NewWeights = np.empty([len_pp,len_pt,numFlux])
        for i in range(len_pp):
            for j in range(len_pt):
                TempMax = np.amax(EvenNewerEnu[i,j])
                TempMin = np.amin(EvenNewerEnu[i,j])
                for k in range(numFlux):
                    NewWeights[i,j,k] = ( (TempMax-TempMin) / numFlux ) * ( NewerFluxNew[i,j,k] / TotalFlux )
    
        doubleDiff = DdxsDipole(Emu3D,EvenNewerEnu,Pmu3D,CosMu3D,MA,1)
        doubleDiffTemp = WeightSum3d(doubleDiff.real,NewWeights)/12.
        doubleDiffTemp = np.where( CosMu2D < np.cos( angleCut * pi / 180), 0., doubleDiffTemp)
        doubleDiffTemp  = doubleDiffTemp * Jac
        doubleDiffTemp = doubleDiffTemp.ravel()
        for i in range(len_pp):
            for j in range(len_pt):
                ddxsHolder[o][i*len_pt+j] = doubleDiffTemp[i*len_pt+j]
    
    ddxsAvg = np.array(sum(ddxsHolder,0)/N)
    
    return ddxsAvg

## Interpolate the flux function that we are  integrating over to calculate a better ddxs  ##
def DdxsDipoleSmoothUnc(N,numFlux,MA):
    
    len_pt = len(pT1D)
    len_pp = len(pP1D)
    ## Number of bins to average  the ddxs over ##
    pT1d = np.array([np.linspace(pT1DLow[i],pT1DHigh[i],N) for i in range(len_pt)])
    pP1d = np.array([np.linspace(pP1DLow[i],pP1DHigh[i],N) for i in range(len_pp)])
    ## Interpolate the flux data ##
    EnuFlux = np.linspace(0.,20.,len(FluxMv))
    Func = interp1d(EnuFlux,FluxMv,kind='cubic')
    EnuNew = np.linspace(0.,20.,numFlux)
    FluxNew = Func(EnuNew)
    TotalFlux = 20 * sum( FluxNew ) / numFlux 
    ## Define the weight functions needed to integrate over the flux ##
    weight = []
    [ weight.append( 20 * flux / numFlux / TotalFlux ) for flux in FluxNew ]
    ## an array to hold the different ddxs arrays to be averaged over ##
    ddxsHolder = np.zeros((N,len_pp*len_pt))
    ###############################################
    ## define the kinematic inputs for each case ##
    ###############################################
    for o in range(N):
        pP2D,pT2D = np.meshgrid(pP1d[:,o],pT1d[:,o],indexing='ij')
        CosMu2D = pP2D / np.sqrt( pP2D**2 + pT2D**2 )
        Tmu2D = np.sqrt( pP2D**2 + pT2D**2 + mMu**2 ) - mMu
        Jac = pT2D / ( Tmu2D + mMu ) / np.sqrt( pP2D**2 + pT2D**2)
        pP3D,pT3D,Enu3D = np.meshgrid(pP1d[:,o],pT1d[:,o],EnuNew,indexing = 'ij')
        Tmu3D = np.sqrt( pP3D**2 + pT3D**2 + mMu**2 ) - mMu
        CosMu3D = pP3D / np.sqrt( pT3D**2 + pP3D**2 )
        Emu3D = Tmu3D + mMu
        Pmu3D = np.sqrt( pT3D**2 + pP3D**2 )
        weight = np.broadcast_to(weight,(len_pp,len_pt,len(FluxNew)))
        ######################################################
        ## make double diff from fit to total cross section ##
        ######################################################
        doubleDiff = DdxsDipole(Emu3D,Enu3D,Pmu3D,CosMu3D,MA,1)
        #######################################################################
        ## find the bounds on the indices where the cross section is nonzero ##
        #######################################################################
        a = np.empty([len_pp,len_pt], dtype=int)
        b = np.empty([len_pp,len_pt],dtype=int)
        for i in range(len_pp):
            for j in range(len_pt):
                A = 0
                B = numFlux-1
                while doubleDiff[i,j,A] == 0.:
                    A += 1
                    if A ==  numFlux:
                        break
                while  doubleDiff[i,j,B] == 0. :
                    B -= 1
                    if B == 0:
                        break
                a[i,j] = A-1
                b[i,j] = B+1
        b = np.where( b == numFlux, numFlux-1, b)
        ## Find new ranges of flux for each combo of p_T and p_|| ##
        EvenNewerEnu = np.empty([len_pp,len_pt,numFlux])
        for i in range(len_pp):
            for j in range(len_pt):
                tempFlux = np.linspace(EnuNew[a[i,j]],EnuNew[b[i,j]],numFlux)
                for k in range(numFlux):
                    EvenNewerEnu[i,j,k] = tempFlux[k]
        ## Create new weight funcntions for each combo of p_T and p_|| ##
        NewerFluxNew = Func(EvenNewerEnu)
        NewWeights = np.empty([len_pp,len_pt,numFlux])
        for i in range(len_pp):
            for j in range(len_pt):
                TempMax = np.amax(EvenNewerEnu[i,j])
                TempMin = np.amin(EvenNewerEnu[i,j])
                for k in range(numFlux):
                    NewWeights[i][j][k] = ( ( TempMax - TempMin ) / numFlux) * ( NewerFluxNew[i,j,k] / TotalFlux )
        doubleDiff = DdxsDipole(Emu3D,EvenNewerEnu,Pmu3D,CosMu3D,MA,1)
        doubleDiffTemp = WeightSum3d(doubleDiff.real,NewWeights)/12.
        doubleDiffTemp = np.where(CosMu2D < np.cos( angleCut * pi / 180 ), 0., doubleDiffTemp)
        doubleDiffTemp  = doubleDiffTemp * Jac
        doubleDiffTemp = doubleDiffTemp.ravel()
        for i in range(len_pp):
            for j in range(len_pt):
                ddxsHolder[o][i*len_pt+j] = doubleDiffTemp[i*len_pt+j]
    ddxsAvg = sum(ddxsHolder,0)/N
    resids = np.zeros((N,len_pp*len_pt))
    for i in range(N):
        for j in range(len_pp*len_pt):
            resids[i,j] = ddxsHolder[i,j] - ddxsAvg[j]
    ddxsUnc = np.sqrt( (1./N)* sum( resids**2 ,0 ) )
    return ddxsUnc
    
###########################
## Create the a elements ##
###########################
def aElements(Q2,q,w,wEff):
    A = 12                                             # number of target neutrons                                          # binding energy GeV
    mT = A*(mN-eB)                                    # mass of target Nucleus
    V = (3*pi**2*A)/(2.*pF**3)                         # Volume of the tagret

    ## fill in the values for some useful coefficients ##
    c = -wEff/q
    d = -( wEff**2 - q**2 )/( 2*q*m_N )
    alpha = 1. - c**2 + d**2
    alpha = np.where( alpha > 0., alpha, 0)

    kappa = 1. - 0.8*exp(-20.*Q2)
    kappa = 0.993
    Elo = np.maximum( kappa * ( Ehi - wEff ), mN*( c*d + sqrt(alpha) ) / ( 1. - c**2 ) )
    Elo = np.where(Elo < Ehi, Elo, Ehi)
    Elo = np.where(Elo <= eB, Ehi,Elo)

    ## Expressions for the b elements analytically ##
    b0 = ( (mT * V ) / ( 2 * pi * q ) * ( ( Ehi - Elo ) + 
                        eB * np.log( ( Ehi - eB ) / ( Elo - eB ) ) ) )
    b1 = ( ( mT * V ) / (2 * pi * mN * q ) * ( .5 * ( Ehi**2 - Elo**2 ) 
            + eB *( Ehi - Elo ) 
            + ( eB**2 ) * np.log( ( Ehi - eB ) / ( Elo - eB ) ) ) )
    b2 = ( ( mT * V ) / ( 2 * pi * mN**2 * q ) * ( 1./3 * ( Ehi**3 - Elo**3 ) 
            + eB * ( .5 * ( Ehi**2 - Elo**2) + eB * ( Ehi - Elo ) 
            + eB**2 * np.log( ( Ehi - eB ) / ( Elo - eB ) ) ) ) )
    
    ## using the b elements as well as c and d to calculate the a elements ##
    a1 = b0
    a2 = b2 - b0
    a3 = c**2 * b2 + 2 * c * d * b1  +  d**2 * b0
    a4 = b2 - 2 * eB * b1 / mN + eB**2 / mN**2 * b0
    a5 = -c * b2 + ( eB * c / mN - d ) * b1 + eB * d / mN * b0
    a6 = -c * b1 - d * b0
    a7 = b1 - eB * b0 / mN

    return a1,a2,a3,a4,a5,a6,a7


################################
## Make FFs for dipole ansatz ##
################################
def FormFactorsDipole(Q2,MA):
    a2 = [3.253,3.104,3.043,3.226,3.188]                    # coefficient for form factor denominator (see Arrington Table 2)
    a4 = [1.422,1.428,0.8548,1.508,1.354]
    a6 = [0.08582,0.1112,0.6806,-0.3773,0.1511]
    a8 = [0.3318,-0.006981,-0.1287,0.6109,-0.01135]
    a10 = [-0.09371,0.0003705,0.008912,-0.1853,0.0005330]
    a12 = [0.01076,-0.7063*10**(-5),0.0,0.01596,-0.9005*10**(-5)]                                           # Vector mass parameter
    MVNuance = 0.84
    M = mN
    GEp = ( np.where(Q2 < 6.,
                   1./(1. + a2[0] * Q2 + a4[0] * Q2**2 
                       + a6[0] * Q2**3 + a8[0] * Q2**4 
                       + a10[0] * Q2**5 + a12[0] * Q2**6),
                   ( muP / ( 1. + a2[1] * Q2
                          + a4[1] * Q2**2 + a6[1] * Q2**3
                          +a8[1] * Q2**4 + a10[1] * Q2**5
                          +a12[1] * Q2**6 ) ) 
                   * ( .5 / ( 1. + a2[0] * 6. + a4[0] * 6.**2
                          + a6[0] * 6.**3 + a8[0] * 6.**4
                          + a10[0] * 6.**5 + a12[0] * 6.**6 ) ) 
                   / ( .5 * muP / ( 1. + a2[1] * 6. + a4[1] * 6.**2
                               + a6[1] * 6.**3 + a8[1] * 6.**4
                               + a10[1] * 6.**5 + a12[1] *6.**6 ) ) ) )
    GMp = ( muP / ( 1. + a2[1] * Q2 + a4[1] * Q2**2 + a6[1] * Q2**3 
                   + a8[1] * Q2**4 + a10[1] * Q2**5 + a12[1] * Q2**6 ) )
    GMn = ( muN / (1. + a2[2] * Q2 + a4[2] * Q2**2 + a6[2] * Q2**3 
                   + a8[2] * Q2**4 + a10[2] * Q2**5 + a12[2] * Q2**6 ) )
    GEn = ( - muN * .942 * Q2 / ( 4 * M**2 )  / ( 1. + ( Q2 / 
          ( 4 * M**2 ) * 4.61 ) ) * ( 1. / (1. +  Q2**2 / MVNuance**2  ) ) )
    GEV = ( GEp - GEn )
    GMV = ( GMp - GMn )
    ## Create the form factors as a function of Q^2 = -q^2 ##
    F1 = ( GEV + ( Q2 / ( 4. * M**2) * GMV ) ) / ( 1. + Q2 / ( 4. * M**2 ) )
    F2 = ( GMV - GEV ) / ( ( 1. + Q2 / ( 4. * M**2 ) ) )
    FA = gA / ( 1. + Q2 / MA**2 )**2
    FP = 2. * mN**2 * FA / ( mPi**2 + Q2 )

    return F1,F2,FA,FP

## calculate the Ddsx from dipole at miniboone energies and angles ##
def DoubleDiffMb(MA):
    ## parameters ##
    A = 12                                                  # number of Nucleons
    mT = A*(mN-eB)                                       # mass of target Nucleus    
    NT = len(Tmu1DMb)
    NC = len(CosMu1DMb)
    ## interpolate the flux as a function of energy ##
    Func = interp1d(EnuFluxMb,FluxMb,kind='cubic')
    numFlux = 500
    EnuNew = np.linspace( 0.001, 3., numFlux, endpoint=True )
    FluxNew = Func( EnuNew )
    TotalFlux = 3 * sum( FluxNew ) / numFlux
    ## Define the weight functions needed to integrate over the flux ##
    weight = []
    [ weight.append( ( 3. / numFlux ) * ( flux / TotalFlux ) 
                    / ( A / 2. ) ) for flux in FluxNew]
    ## broadcast everything to the correct dimension for later ##
    weight = np.broadcast_to( weight, ( NT, NC, numFlux ) )
    EnuNew = np.broadcast_to( EnuNew, ( NT, NC, numFlux ) )
    Tmu = np.broadcast_to( Tmu1DMb, ( numFlux, NC, NT ) )
    Tmu = np.swapaxes( Tmu, 0, 2 )
    CosMu = np.broadcast_to( CosMu1DMb, ( NT, numFlux, NC ) )
    CosMu = np.swapaxes(CosMu, 1, 2 )
    ## define kinematic variables ##
    Emu = Tmu + mMu
    Pmu = np.sqrt( Emu**2 - mMu**2 )
    ## fill in the Q^2 = -q^2 values ##
    Q2 = 2. * Emu * EnuNew - 2. * EnuNew * Pmu * CosMu
    w = ( EnuNew - Emu )
    wEff =  w - eB 
    q = np.sqrt( Q2 + w**2 )
    ## calculate the a elements ##
    a1,a2,a3,a4,a5,a6,a7 = aElements( Q2, q, w, wEff )
    ## calculate the form factors ##
    F1,F2,FA,FP = FormFactorsDipole( Q2, MA )
    ## Use the Form Factors to Define the H Elements ##
    H_1 = 8. * mN**2 * FA**2 + 2. * Q2 * ( ( F1 + F2 )**2 + FA**2 )
    H_2 = 8. * mN**2 * ( F1**2 + FA**2 ) + 2. * Q2 * F2**2
    H_3 = -16. * mN**2 * FA * ( F1 + F2 )
    H_4 = ( Q2 / 2. * ( F2**2 + 4. * FP**2 ) - 2. * mN**2 * F2**2 
           - 4. * mN**2 * ( F1 * F2 + 2. * FA * FP ) )
    H_5 = 8. * mN**2 * ( F1**2 + FA**2 ) + 2. * Q2 * F2**2

    ## Use the a and H values to determine the W values ##
    W1 = a1 * H1 + .5 * ( a2 - a3 ) * H2
    W2 = ( (a4 + w**2 / q**2 * a3 - 2. * w / q * a5 
            + .5 * ( ( 1. - w**2 / q**2  ) * ( a2 - a3 ) ) ) * H2 )
    W3 = ( mT / mN ) * ( a7 - w / q * a6 ) * H3
    W4 = ( ( mT / mN )**2 * ( a1 * H4 + mN * ( a6 * H5 ) / q 
            + ( mN**2 / 2. ) * ( ( 3. * a3 - a2 ) * H2 ) / q**2 ) )
    W5 = ( mT / mN  * ( a7 - w / q * a6 ) * H5 
            + mT * ( 2. * a5 + w / q * ( a2 - 3. * a3 ) ) * H2 / q )

    doubleDiff = ( ( GF**2 * Pmu * Vud**2 )
                      / ( 16. * pi**2 * mT * GeVToCm**2 )
                      * ( 2. * ( Emu - Pmu * CosMu ) * W1 
                        + ( Emu + Pmu * CosMu ) * W2 
                        + ( 1. / mT ) * ( ( Emu - Pmu * CosMu ) * ( Enu + Emu ) 
                                   - mMu**2 ) * W3 
                        + (mMu / mT )**2 * ( Emu - Pmu * CosMu ) * W4 
                        -  mMu**2 / mT * W5 ) )
    doubleDiff = WeightSum3d(doubleDiff.real,weight)

    return doubleDiff

##############################################
## Create double differential cross section ##
### opt = 1 neutrino, opt = 2 antineutrino ###
##############################################
def DdxsDipole(Emu,Enu,Pmu,CosMu,MA,opt):
    ## parameters ##
    A = 12.
    mT = A*(mN-eB)                                       # mass of target Nucleus
    
    ## fill in the Q^2 = -q^2 values ##
    Q2 = 2. * Emu * Enu - 2. * Enu * Pmu * CosMu - mMu**2
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
    H1 = 8. * mN**2 * FA**2 + 2. * Q2 * ( ( F1 + F2 )**2 + FA**2 )
    H2 = 8. * mN**2 * ( F1**2 + FA**2 ) + 2. * Q2 * F2**2
    H3 = -16. * mN**2 * FA * ( F1 + F2 )
    H4 = ( Q2 / 2. * ( F2**2 + 4. * FP**2 ) 
           - 2. * mN**2 * F2**2 - 4. * mN**2 * ( F1 * F2 + 2. * FA * FP ) )
    H5 = 8. * mN**2 * ( F1**2 + FA**2 ) + 2. * Q2 * F2**2

    ## Use the a and H values to determine the W values ##
    W1 = a1 * H1 + .5 * ( a2 - a3 ) * H2
    W2 = ( a4 + ( w / q )**2 * a3 - 2. * ( w / q ) * a5 
          + .5*( ( 1. - ( w / q )**2 * ( a2 - a3 ) ) ) * H2 )
    W3 = ( mT / mN ) * ( a7 - ( w / q ) * a6 ) * H3
    W4 = ( ( mT**2 / mN**2 ) * ( a1 * H4 + mN * ( a6 * H5 ) / q 
          + ( mN**2 / 2. ) * ( ( 3. * a3 - a2 ) * H2 ) / q**2 ) )
    W5 = ( ( mT / mN ) * ( a7 - ( w / q ) * a6 ) * H5 
          + mT * ( 2. * a5 + ( w / q ) * ( a2 - 3. * a3 ) ) * H2 / q )

    if opt == 1:
        doubleDiff = ( ( GF**2 * Pmu * Vud**2 )
                      / ( 16. * pi**2 * mT * GeVToCm**2 )
                      * ( 2. * ( Emu - Pmu * CosMu ) * W1 
                        + ( Emu + Pmu * CosMu ) * W2 
                        + ( 1. / mT ) * ( ( Emu - Pmu * CosMu ) * ( Enu + Emu ) 
                                   - mMu**2 ) * W3 
                        + (mMu / mT )**2 * ( Emu - Pmu * CosMu ) * W4 
                        -  mMu**2 / mT * W5 ) )
    elif opt == 2:
       doubleDiff = ( ( GF**2 * Pmu * Vud**2 )
                      / ( 16. * pi**2 * mT * GeVToCm**2 )
                      * ( 2. * ( Emu - Pmu * CosMu ) * W1 
                        + ( Emu + Pmu * CosMu ) * W2 
                        - ( 1. / mT ) * ( ( Emu - Pmu * CosMu ) * ( Enu + Emu ) 
                                   - mMu**2 ) * W3 
                        + (mMu / mT )**2 * ( Emu - Pmu * CosMu ) * W4 
                        -  mMu**2 / mT * W5 ) )
    else:
        print("Please Enter 1 for neutrino or 2 for antineutrino in the last argument of DoubleDiffDipole")
        exit()
    return doubleDiff

##################################################################################
## Calculate the total cross section by integrating over T_mu and cos_mu values ##
##################################################################################
def CalcCrossSection(Enu,NT,Ncos,DELTACosMu,DELTATmu,doubleDiff):
    A = 12                                                  # number of Nucleons
    NewEnu = Enu
    S = doubleDiff.shape
    if S[0] != NT:
        print( "Error: T_mu dimension is incorrect." ) 
        print("(%s,(%s,%s))" % ( S,NT,2 * Ncos ) )
        exit()
    elif S[1] !=  2 * Ncos:
        print("Error: cos_mu dimension is incorrect." )
        print ("(%s,(%s,%s))" % ( S,NT,2 * Ncos ) )
        exit()
    ## Integrate double_diff over cos_mu and T_mu to get the total cross section ##
    diff = np.zeros(NT)
    for i in range(NT):
        for j in range(2 * Ncos):
            diff[i] += DELTACosMu[i] * doubleDiff[i][j] 
    INT = 0.
    for i in range(NT):
        INT += DELTATmu * ( diff[i] ) / ( A / 2. )
    SIGMA = INT
    return SIGMA,NewEnu

###################################
## function to do the dipole fit ##
###################################
def TotalXsDipole(Enu,MA):
    N = len(Enu)
    SIGMA = np.array([0. for i in range(N)])
    for m  in range(N):
        ( print("Starting Calculation for Enu = %s out of Enu = %s" 
                % ( roundSig( Enu[m] ), roundSig( np.nanmax( Enu ) ) ) ) )
        Ncos = 100 + int( Enu[m] + 1 ) * 800
        NT = 100 + 20 * int( Enu[m] )
        binSize = int(2 * Ncos / 100 )
        numBins = int(2 * Ncos / binSize )
        cosBin = np.array([[0. for j in range(binSize) ] for i in range(NT) ] )
        ################################
        ## Create Kinematic Variables ##
        ################################
        Tmu, Emu, Pmu, Enu, CosMu, DELTACosMu, DELTATmu = MakeVariables( NT, Ncos, Enu[m] ) 
        for l in range(numBins):
            for i in range(NT):
                for j in range(binSize):
                    CosBin[i][j] = CosMu[i][j+l*binSize]
            #####################################################################
            ## Create RFG Variables #############################################
            ## For the last entry, put 1 for neutrinos, or 2 for antineutrinos ##
            #####################################################################
            doubleDiff = DdxsDipole( Emu, Enu, Pmu, cosBin, MA, 1 )
            doubleDiff = np.where(cosBin >= np.cos( 20. * pi / 180 ), doubleDiff, 0. ) 
            ## apply the angle cut of Minerva ##
            #double_diff = where((arccos(cos_bin)*180/pi <= 20) & (arccos(cos_bin)*180/pi >= -20), double_diff, 0.)
            #############################
            ## Calculate Cross Section ##
            #############################
            SIGMATemp,NewEnu = CalcCrossSection( Enu, NT, int( Ncos / 100 ) ,DELTACosMu,DELTATmu,doubleDiff)
            #################################
            ## Add to total value of SIGMA ##
            #################################
            SIGMA[m] +=SIGMATemp
    ## uncomment these below if  you want to  get a rough idea  what the graph should look like, with lower number of T_mu, cos_mu points ##.
    for j in range(100):
        for i in range(len(Enu)):
            if Enu[i] >= 12.:
                SIGMA[i] = SIGMA[i-1]
    print(SIGMA)
    return SIGMA



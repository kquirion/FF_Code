## file  containing variable  creation functions ##
from numpy import array,linspace,sqrt,broadcast_to,swapaxes,nanmin,nanmax,where
from DataFile import *

########################################################################
## Create a function to make the less constrained kinematic variables ##
########################################################################
def Variables(NT,Ncos,Enu):
    TmuMax = Enu + Ehi - mMu - mN
    TmuMin = 0.05
    Tmu = linspace(TmuMin,TmuMax,NT,endpoint=False)
    DeltaTmu = (TmuMax-TmuMin)/NT
    Emu = Tmu + mMu
    Pmu = sqrt( Emu**2 - mMu**2 )
    ## Restrict cos values to those satisfying Q2 > 0 ##
    CosMax = Emu / Pmu - mMu**2 / ( 2. * Enu * Pmu )
    CosMax = where(CosMax < 1.0, CosMax, 1.0)
    #CosMax = 20.*pi/180.
    CosMu = array([linspace(-CosMax[i],CosMax[i],2*Ncos,endpoint=False) for i in range(NT)])
    #CosMu = array([linspace(-CosMax,CosMax,2*Ncos,endpoint=False) for i in range(NT)])
    DeltaCosMu = array([0.0  for i in range(NT)])
    for i in range(NT):
        DeltaCosMu[i] = abs(CosMu[i][1] - CosMu[i][0])
    Tmu = broadcast_to(Tmu,(int(2*Ncos/100),NT))
    Tmu = swapaxes(Tmu,0,1)
    Emu = broadcast_to(Emu,(int(2*Ncos/100),NT))
    Emu = swapaxes(Emu,0,1)
    Pmu = broadcast_to(Pmu,(int(2*Ncos/100),NT))
    Pmu = swapaxes(Pmu,0,1)
    return Tmu,Emu,Pmu,Enu,CosMu,DeltaCosMu,DeltaTmu

########################################################################
## Create a function to make the less constrained kinematic variables ##
########################################################################
def VariablesUnbinned(NT,Ncos,Enu):
    TmuMax = Enu + Ehi - mMu - mN
    TmuMin = 0.05
    Tmu = linspace(TmuMin,TmuMax,NT,endpoint=False)
    DeltaTmu = (TmuMax-TmuMin)/NT
    Emu = Tmu + mMu
    Pmu = sqrt( Emu**2 - mMu**2 )
    ## Restrict cos values to those satisfying Q2 > 0 ##
    CosMax = Emu / Pmu - mMu**2 / ( 2. * Enu * Pmu )
    CosMax = where(CosMax < 1.0, CosMax, 1.0)
    #CosMax = 20.*pi/180.
    CosMu = array([linspace(-CosMax[i],CosMax[i],2*Ncos,endpoint=False) for i in range(NT)])
    #CosMu = array([linspace(-CosMax,CosMax,2*Ncos,endpoint=False) for i in range(NT)])
    DeltaCosMu = array([0.0  for i in range(NT)])
    for i in range(NT):
        DeltaCosMu[i] = abs(CosMu[i][1] - CosMu[i][0])
    Tmu = broadcast_to(Tmu,(2*Ncos,NT))
    Tmu = swapaxes(Tmu,0,1)
    Emu = broadcast_to(Emu,(2*Ncos,NT))
    Emu = swapaxes(Emu,0,1)
    Pmu = broadcast_to(Pmu,(2*Ncos,NT))
    Pmu = swapaxes(Pmu,0,1)
    return Tmu,Emu,Pmu,Enu,CosMu,DeltaCosMu,DeltaTmu

#######################################################
## Create a function to make the kinematic variables ##
#######################################################
def VariablesConstrained(NT,Ncos,Enu):
    CosMu = linspace(-1.0,1.0,2*Ncos)
    CosMu = broadcast_to(CosMu,(NT,2*Ncos))
    ## 1D Create arrays for neutron momentum and energy ##
    pn = linspace(-pF,pF,NT)
    pn = broadcast_to(pn,(2*Ncos,NT))
    pn = swapaxes(pn,0,1)
    En = sqrt( pn**2 + mN**2 )
    ## define quantities for solving the quadratic equation for Tmu ##
    e = -2. * ( Enu + En ) * ( Enu * ( En - mMu - pn ) - mMu * En + mMu**2 / 2. )
    a = ( Enu + En )**2 - CosMu**2 * ( Enu + pn )**2
    b = e - 2. * ( CosMu )**2 * ( Enu + pn )**2
    c = e**2 / ( ( Enu + En )**2 )
    CosBounds = where( b**2 - 4. * a * c > 0.0 , CosMu,10)
    CosBound = nanmin(CosBounds,axis=1)
    CosBound = where(CosBound < 1.0, CosBound, 1.0)
    CosMu = array([linspace(-CosBound[i],CosBound[i],2*Ncos,endpoint=False) for i in range(NT)])
    a = ( Enu + En ) - ( CosMu )**2 * ( Enu + pn )**2
    b = e - 2. * ( CosMu )**2 * ( Enu + pn )**2
    c = e**2 / ( ( Enu + En )**2 )
    d = b**2 - 4. * a * c
    d = nanmax(d)
    TmuMax = abs( nanmax( 1. / ( 2. * a ) * ( - b + sqrt( d ) ) ) )
    TmuMin = abs( nanmin( 1. / ( 2. * a ) * ( - b - sqrt( d ) ) ) )
    DeltaTmu = abs( TmuMax - TmuMin ) / NT
    Tmu = linspace(TmuMin,TmuMax,NT,endpoint=False)
    Emu = Tmu + mMu
    Pmu = sqrt( Emu**2 - mMu**2 )
    CosMax = Emu / Pmu - mMu**2 / ( 2. * Enu * Pmu )
    CosMax = where(CosMax < 1.0 ,CosMax,1.0)
    for i in range(len(CosBound)):
        CosMax = where(CosBound > CosMax, CosBound, CosMax)
    CosMu = array([linspace(-CosMax[i],CosMax[i],2*Ncos,endpoint=False) for i in range(NT)])
    #CosMu = np.where((-0.24 < CosMu) & (CosMu < 0.24) , 0.25, CosMu)
    DeltaCosMu = array([0.0 for i in range(NT)])
    for i in range(NT):
        DeltaCosMu[i] = abs(CosMu[i][1] - CosMu[i][0])
    Tmu = broadcast_to(Tmu,(int(2*Ncos/500),NT))
    Tmu = swapaxes(Tmu,0,1)
    Emu = broadcast_to(Emu,(int(2*Ncos/500),NT))
    Emu = swapaxes(Emu,0,1)
    Pmu = broadcast_to(Pmu,(int(2*Ncos/500),NT))
    Pmu = swapaxes(Pmu,0,1)
    return Tmu,Emu,Pmu,Enu,CosMu,DeltaCosMu,DeltaTmu
    
#########################################################################
## Create a function to make kinematic variables where Enu is an array##
#########################################################################
def Variables3D(NT,Ncos,Enu):
    NE = len(Enu)
    ## constraints on Tmu  ##
    TmuMax = Enu + Ehi - mMu - mN
    TmuMax = where(TmuMax > 0.05, TmuMax, 0.06)
    TmuMin = 0.05
    Tmu = array([linspace(TmuMin,TmuMax[k],NT,endpoint=False) for k in range(NE)] )
    Tmu = swapaxes(Tmu,0,1)
    Emu = Tmu + mMu
    Pmu = sqrt( Emu**2 - mMu**2 )
    ## Restrict cos values to those satisfying Q2 > 0 ##
    CosMax = where(Pmu > 0.,Emu/Pmu - mMu**2/(2.0*Enu*Pmu),0.)
    CosMax = where(CosMax < 1.0, CosMax, 1.0)
    CosMu = array([[linspace(0.,CosMax[i,j],2*Ncos,endpoint=False) for j in range(NE)] for i in range(NT)])
    CosMu = swapaxes(CosMu,2,1)
    Tmu = array([[[Tmu[i,j] for j in range(NE)] for k in range(2*Ncos)] for i in range(NT)])
    Emu = Tmu + mMu
    Pmu = sqrt( Emu**2 - mMu**2 )
    pP = Pmu*CosMu
    pT = Pmu*sqrt(1.-CosMu**2)
    return Tmu,Emu,Pmu,CosMu,pP,pT

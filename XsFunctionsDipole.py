## This file contains many functions for running cross section calculations ##
import numpy as np
from math import pi
from sys import exit
from scipy.interpolate import interp1d
from misc_fns import sq,weight_sum_3d,round_sig
from variable_fns import make_variables
from DataFile import *

np.set_printoptions(precision=3)

## Interpolate the flux function that we are  integrating over to calculate a better ddxs  ##
def DdxsDipoleSmooth(N,numFlux,MA):

    len_pt = len(pT1D)
    len_pp = len(pP1D)
    
    ## create upper and lower bounds on bins within each experimental bin ##
    pT1d = np.array([linspace(pT1DLow[i],pT1DHigh[i],N) for i in range(len_pt)])
    pP1d = np.array([linspace(pP1DLow[i],pP1DHigh[i],N) for i in range(len_pp)])
    
    ## Interpolate the flux data ##
    EnuFlux = np.linspace(0.,20.,len(FluxMv))
    Func = interp1d(EnuFlux,FluxMv,kind='cubic')
    EnuNew = np.linspace(0.,20.,num_flux)
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
                while  doubleDiffiff[i,j,B] == 0. :
                    B -= 1
                    if B == 0:
                        break
                a[i,j] = A-1
                b[i,j] = B+1
        b = np.where( b == numFlux, numFlux-1, b)
    
        ## Find new ranges of flux for each combo of p_T and p_|| ##
        EvenNewerEnu = empty([len_pp,len_pt,numFlux])
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
    
    ddxsAvg = array(sum(ddxsHolder,0)/N)
    
    return ddxsAvg

## Interpolate the flux function that we are  integrating over to calculate a better ddxs  ##
def DdxsDipoleSmoothUnc(N,numFlux,MA):
    
    len_pt = len(pT1D)
    len_pp = len(pP1D)

    ## Number of bins to average  the ddxs over ##
    pT1d = np.array([np.linspace(pT1DLow[i],pT1DHigh[i],N) for i in range(len_pt)])
    pP1d = np.array([np.linspace(pP1DLow[i],pP1DHigh[i],N) for i in range(len_pp)])
    
    ## Interpolate the flux data ##
    EnuFlux = linspace(0.,20.,len(FluxMv))
    Func = interp1d(EnuFlux,FluxMv,kind='cubic')
    EnuNew = np.linspace(0.,20.,numFlux)
    FluxNew = Func(EnuNew)

    TotalFlux = 0
    Total_Flux = 20 * sum( FluxNew) / numFlux 

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
    
        weight = broadcast_to(weight,(len_pp,len_pt,len(Flux_new)))
    
        ######################################################
        ## make double diff from fit to total cross section ##
        ######################################################
        double_diff = make_double_diff_dipole(E_mu_3D,E_nu_3D,P_mu_3D,cos_mu_3D,M_A,1)
    
        #######################################################################
        ## find the bounds on the indices where the cross section is nonzero ##
        #######################################################################
        a = empty([len_pp,len_pt], dtype=int)
        b = empty([len_pp,len_pt],dtype=int)
        for i in range(len_pp):
            for j in range(len_pt):
                A = 0
                B = num_flux-1
                while double_diff[i,j,A] == 0.:
                    A += 1
                    if A ==  num_flux:
                        break
                while  double_diff[i,j,B] == 0. :
                    B -= 1
                    if B == 0:
                        break
                a[i,j] = A-1
                b[i,j] = B+1
        b = where( b == num_flux, num_flux-1, b)
    
        ## Find new ranges of flux for each combo of p_T and p_|| ##
        even_newer_E_nu = empty([len_pp,len_pt,num_flux])
        for i in range(len_pp):
            for j in range(len_pt):
                temp_flux = linspace(E_nu_new[a[i,j]],E_nu_new[b[i,j]],num_flux)
                for k in range(num_flux):
                    even_newer_E_nu[i,j,k] = temp_flux[k]
        ## Create new weight funcntions for each combo of p_T and p_|| ##
        newer_flux_new = Func(even_newer_E_nu)
        new_weights = empty([len_pp,len_pt,num_flux])
        for i in range(len_pp):
            for j in range(len_pt):
                temp_max = amax(even_newer_E_nu[i,j])
                temp_min = amin(even_newer_E_nu[i,j])
                for k in range(num_flux):
                    new_weights[i][j][k] = ((temp_max-temp_min)/num_flux)*(newer_flux_new[i,j,k]/Total_Flux)
    
        double_diff = make_double_diff_dipole(E_mu_3D,even_newer_E_nu,P_mu_3D,cos_mu_3D,M_A,1)
        double_diff_temp = weight_sum_3d(double_diff.real,new_weights)/12.
        double_diff_temp = where(cos_mu_2D < cos(angle_cut*pi/180), 0., double_diff_temp)
        double_diff_temp  = double_diff_temp*Jac
        double_diff_temp = double_diff_temp.ravel()
        for i in range(len_pp):
            for j in range(len_pt):
                ddxs_holder[o][i*len_pt+j] = double_diff_temp[i*len_pt+j]
    
    ddxs_avg = sum(ddxs_holder,0)/N
    resids = zeros((N,len_pp*len_pt))
    for i in range(N):
        for j in range(len_pp*len_pt):
            resids[i,j] = ddxs_holder[i,j] - ddxs_avg[j]
    ddxs_unc = sqrt( (1./N)* sum(sq( resids ),0 ) )
    
    return ddxs_unc
    
###########################
## Create the a elements ##
###########################
def make_a_elements(Q2,q,w,w_eff):
    A = 12                                             # number of target neutrons
    E_hi = sqrt(mN**2 + pF**2)                         # Upper limit of neutron energy integration                                           # binding energy GeV
    m_T = A*(mN-eB)                                    # mass of target Nucleus
    V = (3*pi**2*A)/(2.*pF**3)                         # Volume of the tagret

    ## fill in the values for some useful coefficients ##
    c = -w_eff/q
    d = -(sq(w_eff) - sq(q))/(2.*q*m_N )
    alpha = 1.0 - sq(c) + sq(d)
    alpha = where( alpha > 0., alpha, 0)

    #i,j,k = where(alpha != 0.0)
    #for l in range(len(i)):
    #    print ("( %s,%s,%s ) -> cos_mu = %s" % (i[l],j[l],k[l],alpha[i[l]][j[l]][k[l]]))
    kappa = 1. - 0.8*exp(-20.*Q2)
    kappa = 0.993
    E_lo = maximum( kappa*(E_hi - w_eff), m_N*( c*d + sqrt(alpha) )/( 1. - sq(c) ) )
    E_lo = where(E_lo < E_hi, E_lo, E_hi)
    E_lo = where(E_lo <= E_b, E_hi,E_lo)

    ## Expressions for the b elements analytically ##
    #b_0 = quad(Integrand_Zero, E_lo, E_hi, args=(E_b,m_N,m_T,V,q))
    #b_1 = quad(Integrand_One, E_lo, E_hi, args=(E_b,m_N,m_T,V,q))
    #b_2 = quad(Integrand_Two, E_lo, E_hi, args=(E_b,m_N,m_T,V,q))
    b_0 = (m_T*V)/(2.0*pi*q)*((E_hi - E_lo) + E_b*log((E_hi - E_b)/(E_lo - E_b)))
    b_1 = (m_T*V)/(2.0*pi*m_N*q)*(.5*(sq(E_hi) - sq(E_lo)) + E_b*(E_hi - E_lo) + (sq(E_b))*log((E_hi - E_b)/(E_lo - E_b)))
    b_2 = (m_T*V)/(2.0*pi*sq(m_N)*q)*(1./3.*(E_hi**3 - power(E_lo,3)) + E_b*(0.5*(sq(E_hi) - sq(E_lo)) + E_b*(E_hi - E_lo) + sq(E_b)*log((E_hi - E_b)/(E_lo - E_b))))
    ## using the b elements as well as c and d to calculate the a elements ##
    a_1 = b_0
    a_2 = b_2 - b_0
    a_3 = sq(c)*b_2 + 2.0*(c*d*b_1) + sq(d)*b_0
    a_4 = b_2 - 2.0*E_b*b_1/m_N + sq(E_b)/sq(m_N)*b_0
    a_5 = -c*b_2 + (E_b*c/m_N - d)*b_1 + (E_b*d/m_N)*b_0
    a_6 = -c*b_1 - d*b_0
    a_7 = b_1 - E_b*b_0/m_N

    #i,j,k = where(alpha <= 0.0)
    #for l in range(len(i)):
    #    print ("( %s,%s ) -> a_1 = %s" % (i[l],j[l],alpha[i[l]][j[l]][0]))
    #print len(i)

    return a_1,a_2,a_3,a_4,a_5,a_6,a_7

#######################
## Make form factors ##
#######################
def make_form_factors(Q2,M_A,Form_factor):

    a2 = [3.253,3.104,3.043,3.226,3.188]                    # coefficient for form factor denominator (see Arrington Table 2)
    a4 = [1.422,1.428,0.8548,1.508,1.354]
    a6 = [0.08582,0.1112,0.6806,-0.3773,0.1511]
    a8 = [0.3318,-0.006981,-0.1287,0.6109,-0.01135]
    a10 = [-0.09371,0.0003705,0.008912,-0.1853,0.0005330]
    a12 = [0.01076,-0.7063*10**(-5),0.0,0.01596,-0.9005*10**(-5)]
    M_V2 = 0.71                                             # Vector mass parameter
    Q_low = 1.3                                             # Energies for determining the form of Dr Friedland's F_A parametrization
    Q_high = 1000.0
    m_a_1 = 1.230                                           # Masses for defining the Masjuan double pole parametrization for F_A
    m_a_2 = 1.647

    GEp = where(Q2 < 6.,1./(1 + a2[0]*Q2 + a4[0]*sq(Q2) + a6[0]*power(Q2,3) + a8[0]*power(Q2,4) + a10[0]*power(Q2,5) + a12[0]*power(Q2,6)),(mu_p/(1+a2[1]*Q2+a4[1]*sq(Q2)+a6[1]*power(Q2,3)+a8[1]*power(Q2,4)+a10[1]*power(Q2,5)+a12[1]*power(Q2,6))) * (.5/(1.+a2[0]*6.+a4[0]*6.**2+a6[0]*6.**3+a8[0]*6.**4+a10[0]*6.**5+a12[0]*6.**6)) / (.5*mu_p/(1.+a2[1]*6.+a4[1]*6.**2+a6[1]*6.**3+a8[1]*6.**4+a10[1]*6.**5+a12[1]*6.**6)))
    GMp = mu_p/(1+a2[1]*Q2+a4[1]*sq(Q2)+a6[1]*power(Q2,3)+a8[1]*power(Q2,4)+a10[1]*power(Q2,5)+a12[1]*power(Q2,6))
    GMn = mu_n/(1+a2[2]*Q2+a4[2]*sq(Q2)+a6[2]*power(Q2,3)+a8[2]*power(Q2,4)+a10[2]*power(Q2,5)+a12[2]*power(Q2,6))
    GEn = -mu_n*0.942*(Q2/(4*sq(M))) / (1+(Q2/(4*M**2)*4.61)) * (1/(1 + (sq(Q2)/M_V2**2)))
    GEV = (GEp - GEn)
    GMV = (GMp - GMn)
    ## Create the form factors as a function of Q^2 = -q^2 ##
    #F_1 = (GEV + (Q2/(4*M**2)*GMV))/(1 + Q2/(4*sq(M)))
    #F_2 = (GMV - GEV)/((1 + Q2/(4*M**2)))
    F_1 = (GEV + (Q2/(4.*M**2)*GMV))/(1. + Q2/(4.*sq(M)))
    F_2 = (GMV - GEV)/((1. + Q2/(4.*M**2)))
    if Form_factor == 1:
        M_A = 1.03
        F_A = g_A / sq(1 + Q2/sq(M_A))
    elif Form_factor == 2:
        M_A = 0.7
        F_A = g_A / sq(1 + Q2/sq(M_A))
    elif Form_factor == 3:
        ## create Dr. Friedland's Q^2 axial mass dependence ##
        M_A = "Varied"
        M_A_Varied = 1.35 + (sq(Q2)/Q_low**4)/(1.0 + (sq(Q2)/Q_low**4 + (power(Q2,5)/Q_high**10)))*(1.03-1.65)
        F_A = g_A / (1 + Q2/M_A_Varied**2)**2
    elif Form_factor == 4:
        M_A = "Double pole"
        F_A = g_A*(m_a_1**2)*(m_a_2**2)/((m_a_1**2 + Q2)*(m_a_2**2 + Q2))
    elif Form_factor == 5:
        M_A = 0.856
        GAMMA = 0.600
        F_A = where(Q2<sq(M_rho+M_pi),g_A*sq(M_A)/(sq(M_A) + Q2 - (1j*M_A*GAMMA*(4.1*power(Q2-9*sq(M_pi),3)*(1-3.3*(Q2-9*sq(M_pi))+5.8*sq(Q2-9*sq(M_pi))))/(4.1*(sq(M_A)-9*sq(M_pi))*(1-3.3*(sq(M_A)-9*sq(M_pi))+5.8*sq(sq(M_A)-9*sq(M_pi)))))),g_A*sq(M_A)/(sq(M_A) + Q2 - (1j*M_A*GAMMA*(Q2*(1.623+10.38/Q2-9.32/sq(Q2)+0.65/power(Q2,3)))/(sq(M_A)*(1.623+10.38/Q2-9.32/sq(sq(M_A))+0.65/power(sq(M_A),3))))))
    elif Form_factor == 6:
        ## use Sergi's Lambda parametrization  ##
        M_A = 1.35
        L = 1.3
        F_A = g_A / (1.+Q2/sq(M_A)) /(1.+Q2/sq(L))
    else:
        exit()
    F_P = (2.0*m_N**2)*F_A/(M_pi**2 + Q2)

    return F_1,F_2,F_A,F_P,M_A

#############################
## Make FFs for dipole fit ##
#############################
def make_form_factors_dipole(Q2,M_A):
    a2 = [3.253,3.104,3.043,3.226,3.188]                    # coefficient for form factor denominator (see Arrington Table 2)
    a4 = [1.422,1.428,0.8548,1.508,1.354]
    a6 = [0.08582,0.1112,0.6806,-0.3773,0.1511]
    a8 = [0.3318,-0.006981,-0.1287,0.6109,-0.01135]
    a10 = [-0.09371,0.0003705,0.008912,-0.1853,0.0005330]
    a12 = [0.01076,-0.7063*10**(-5),0.0,0.01596,-0.9005*10**(-5)]
    M_V2 = 0.71                                             # Vector mass parameter
    a = 0.942

    GEp = where(Q2 < 6.0,1.0/(1 + a2[0]*Q2 + a4[0]*sq(Q2) + a6[0]*power(Q2,3) + a8[0]*power(Q2,4) + a10[0]*power(Q2,5) + a12[0]*power(Q2,6)),(mu_p/(1+a2[1]*Q2+a4[1]*sq(Q2)+a6[1]*power(Q2,3)+a8[1]*power(Q2,4)+a10[1]*power(Q2,5)+a12[1]*power(Q2,6))) * (0.5/(1+a2[0]*6.0+a4[0]*6.0**2+a6[0]*6.0**3+a8[0]*6.0**4+a10[0]*6.0**5+a12[0]*6.0**6)) / (0.5*mu_p/(1+a2[1]*6.0+a4[1]*6.0**2+a6[1]*6.0**3+a8[1]*6.0**4+a10[1]*6.0**5+a12[1]*6.0**6)))
    GMp = mu_p/(1+a2[1]*Q2+a4[1]*sq(Q2)+a6[1]*power(Q2,3)+a8[1]*power(Q2,4)+a10[1]*power(Q2,5)+a12[1]*power(Q2,6))
    GMn = mu_n/(1+a2[2]*Q2+a4[2]*sq(Q2)+a6[2]*power(Q2,3)+a8[2]*power(Q2,4)+a10[2]*power(Q2,5)+a12[2]*power(Q2,6))
    GEn = -mu_n*0.942*(Q2/(4*sq(M))) / (1.+(Q2/(4*M**2)*4.61)) * (1./(1. + (sq(Q2)/M_V_nuance**2)))
    GEV = (GEp - GEn)
    GMV = (GMp - GMn)
    ## Create the form factors as a function of Q^2 = -q^2 ##
    F_1 = (GEV + (Q2/(4.*M**2)*GMV))/(1. + Q2/(4.*sq(M)))
    F_2 = (GMV - GEV)/((1. + Q2/(4.*M**2)))
    F_A = g_A / sq(1. + Q2/sq(M_A))
    F_P = 2.0*sq(m_N)*F_A/(M_pi**2 + Q2)

    return F_1,F_2,F_A,F_P

## Create the W elements which are used for the double differential cross sextion ##
def make_double_diff_miniboone(M_A):
    ## parameters ##
    A = 12                                                  # number of Nucleons
    m_T = A*(m_N-E_b)                                       # mass of target Nucleus    
    PT = 5.58*10**(20)                                     # Integrated number of protons on target
    NT = len(T_mu_1d)
    NC = len(cos_mu_1d)

    ## fill in the Q^2 = -q^2 values ##
    Func = interp1d(E_nu,Flux,kind='cubic')

    num_flux = 500
    E_nu_new = linspace(0.001,3.,num_flux,endpoint=True)
    Flux_new = Func(E_nu_new)

    Total_Flux = 0
    for i in range(len(Flux_new)):
        Total_Flux = Total_Flux + 3./num_flux*(Flux_new[i])

    ## Define the weight functions needed to integrate over the flux ##
    weight = []
    for i in range(len(Flux_new)):
        weight.append( (3./num_flux)*(Flux_new[i]/Total_Flux)/(A/2))

    weight = broadcast_to(weight,(NT,NC,num_flux))
    E_nu_new = broadcast_to(E_nu_new,(NT,NC,num_flux))

    T_mu = broadcast_to(T_mu_1d,(num_flux,NC,NT))
    T_mu = swapaxes(T_mu,0,2)

    cos_mu = broadcast_to(cos_mu_1d,(NT,num_flux,NC))
    cos_mu = swapaxes(cos_mu,1,2)

    E_mu = T_mu + m_mu
    P_mu = sqrt(sq(E_mu) - sq(m_mu))

    ## fill in the Q^2 = -q^2 values ##
    Q2 = 2.0*E_mu*E_nu_new - 2.0*E_nu_new*P_mu*cos_mu
    ## fill in values of the W Boson Energy ##
    w = (E_nu_new - E_mu)
    ## fill in the values of the W boson energy ##
    w_eff = (w - E_b)
    ## fill in the values for the 3-momentum of the Boson ##
    q = sqrt(Q2 + sq(w))

    ## calculate the a elements ##
    a_1,a_2,a_3,a_4,a_5,a_6,a_7 = make_a_elements(Q2,q,w,w_eff)

    ## calculate the form factors ##
    F_1,F_2,F_A,F_P = make_form_factors_dipole(Q2,M_A)

    ## Use the Form Factors to Define the H Elements ##
    H_1 = 8.0*sq(m_N)*sq(F_A) + 2.0*Q2*(sq(F_1 + F_2) + sq(F_A))
    H_2 = 8.0*m_N**2*(sq(F_1) + sq(F_A)) + 2.0*Q2*sq(F_2)
    H_3 = -16.0*sq(m_N)*F_A*(F_1 + F_2)
    H_4 = Q2/2.0*(sq(F_2) + 4.0*sq(F_P)) - 2.0*sq(m_N)*sq(F_2) - 4.0*sq(m_N)*(F_1*F_2 + 2.0*F_A*F_P)
    H_5 = 8.0*sq(m_N)*(sq(F_1) + sq(F_A)) + 2.0*Q2*sq(F_2)

    ## Use the a and H values to determine the W values ##
    W_1 = a_1*H_1 + 0.5*(a_2 - a_3)*H_2
    W_2 = (a_4 + (sq(w)/sq(q))*a_3 - 2.0*(w/q)*a_5 + 0.5*((1-(sq(w)/sq(q)))*(a_2-a_3)))*H_2
    W_3 = (m_T/m_N)*(a_7 - (w/q)*a_6)*H_3
    W_4 = (sq(m_T)/sq(m_N))*(a_1*H_4 + m_N*(a_6*H_5)/q + (sq(m_N)/2.0)*((3.0*a_3 - a_2)*H_2)/sq(q))
    W_5 = (m_T/m_N)*(a_7 - (w/q)*a_6)*H_5 + m_T*(2.0*a_5 + (w/q)*(a_2 - 3.0*a_3))*(H_2/q)

    double_diff = (sq(G_F)*P_mu*V_ud**2)/(16.0*sq(pi)*m_T*(GeV_To_Cm**2))*( 2.*(E_mu-P_mu*cos_mu)*W_1 + (E_mu+P_mu*cos_mu)*W_2 + (1/m_T)*((E_mu-P_mu*cos_mu)*(E_nu_new+E_mu) - sq(m_mu))*W_3 + sq(m_mu/m_T)*(E_mu-P_mu*cos_mu)*W_4 - (sq(m_mu)/m_T)*W_5)
    double_diff = weight_sum_3d(double_diff.real,weight)

    return double_diff

##############################################
## Create double differential cross section ##
### opt = 1 neutrino, opt = 2 antineutrino ###
##############################################
def DdxsDipole(E_mu,E_nu,P_mu,cos_mu,M_A,opt):
    ## parameters ##
    A = 12.
    m_T = A*(m_N-E_b)                                       # mass of target Nucleus
    
    ## fill in the Q^2 = -q^2 values ##
    Q2 = 2.0*E_mu*E_nu - 2.0*E_nu*P_mu*cos_mu - m_mu**2
    ## fill in values of the W Boson Energy ##
    w = E_nu - E_mu
    ## fill in the values of the W boson energy ##
    w_eff = w - E_b
    ## fill in the values for the 3-momentum of the Boson ##
    q = sqrt(Q2 + sq(w))
    ## calculate the a elements ##
    a_1,a_2,a_3,a_4,a_5,a_6,a_7 = make_a_elements(Q2,q,w,w_eff)
    ## calculate the form factors ##
    F_1,F_2,F_A,F_P = make_form_factors_dipole(Q2,M_A)

    ## Use the Form Factors to Define the H Elements ##
    H_1 = 8.0*sq(m_N)*sq(F_A) + 2.0*Q2*(sq(F_1 + F_2) + sq(F_A))
    H_2 = 8.0*m_N**2*(sq(F_1) + sq(F_A)) + 2.0*Q2*sq(F_2)
    H_3 = -16.0*sq(m_N)*F_A*(F_1 + F_2)
    H_4 = Q2/2.0*(sq(F_2) + 4.0*sq(F_P)) - 2.0*sq(m_N)*sq(F_2) - 4.0*sq(m_N)*(F_1*F_2 + 2.0*F_A*F_P)
    H_5 = 8.0*sq(m_N)*(sq(F_1) + sq(F_A)) + 2.0*Q2*sq(F_2)

    ## Use the a and H values to determine the W values ##
    W_1 = a_1*H_1 + 0.5*(a_2 - a_3)*H_2
    W_2 = (a_4 + (sq(w)/sq(q))*a_3 - 2.0*(w/q)*a_5 + 0.5*((1-(sq(w)/sq(q)))*(a_2-a_3)))*H_2
    W_3 = (m_T/m_N)*(a_7 - (w/q)*a_6)*H_3
    W_4 = (sq(m_T)/sq(m_N))*(a_1*H_4 + m_N*(a_6*H_5)/q + (sq(m_N)/2.0)*((3.0*a_3 - a_2)*H_2)/sq(q))
    W_5 = (m_T/m_N)*(a_7 - (w/q)*a_6)*H_5 + m_T*(2.0*a_5 + (w/q)*(a_2 - 3.0*a_3))*(H_2/q)

    if opt == 1:
        double_diff = (sq(G_F)*P_mu*V_ud**2)/(16.0*sq(pi)*m_T*(GeV_To_Cm**2))*( 2.0*(E_mu-P_mu*cos_mu)*W_1 + (E_mu+P_mu*cos_mu)*W_2 + (1/m_T)*((E_mu-P_mu*cos_mu)*(E_nu+E_mu) - sq(m_mu))*W_3 + sq(m_mu/m_T)*(E_mu-P_mu*cos_mu)*W_4 - (sq(m_mu)/m_T)*W_5)
    elif opt == 2:
        double_diff = (sq(G_F)*P_mu*V_ud**2)/(16.0*sq(pi)*m_T*(GeV_To_Cm**2))*( 2.0*(E_mu-P_mu*cos_mu)*W_1 + (E_mu+P_mu*cos_mu)*W_2 - (1/m_T)*((E_mu-P_mu*cos_mu)*(E_nu+E_mu) - sq(m_mu))*W_3 + sq(m_mu/m_T)*(E_mu-P_mu*cos_mu)*W_4 - (sq(m_mu)/m_T)*W_5)
    else:
        print("Please Enter 1 for neutrino or 2 for antineutrino in the last argument of make_double_diff_dipole")
        exit()
    return double_diff

##################################################################################
## Calculate the total cross section by integrating over T_mu and cos_mu values ##
##################################################################################
def calc_cross_section(E_nu,N_T,N_cos,DELTA_cos_mu,DELTA_T_mu,double_diff):
    A = 12                                                  # number of Nucleons
    new_E_nu = E_nu
    S = double_diff.shape
    if S[0] != N_T:
        print "Error: T_mu dimension is incorrect."
        print ("(%s,(%s,%s))" % (S,N_T,2*N_cos))
        exit()
    elif S[1] !=  2*N_cos:
        print "Error: cos_mu dimension is incorrect."
        print ("(%s,(%s,%s))" % (S,N_T,2*N_cos))
        exit()

    ## Integrate double_diff over cos_mu and T_mu to get the total cross section ##
    diff = zeros(N_T)

    for i in range(N_T):
        for j in range(2*N_cos):
            diff[i] = diff[i] + DELTA_cos_mu[i]*(double_diff[i][j])

    INT = 0.0
    for i in range(N_T):
        INT = INT + DELTA_T_mu*(diff[i])/(A/2.0)

    SIGMA = INT

    return SIGMA,new_E_nu

###################################
## function to do the dipole fit ##
###################################
def make_total_xs_dipole(E_nu,M_A):

    E_nu_array = E_nu
    N = len(E_nu_array)
    SIGMA = array([0.0 for i in range(N)])
    for m  in range(N):
        print "Starting Calculation for E_nu = %s out of E_nu = %s" % (round_sig(E_nu_array[m]),round_sig(nanmax(E_nu_array)))
        N_cos = 100 + int(E_nu_array[m]+1)*800
        N_T = 100 + 20*int(E_nu_array[m])
        bin_size = int(2*N_cos/100)
        num_bins = int(2*N_cos/bin_size)
        cos_bin =array([[0.0 for j in range(bin_size)] for i in range(N_T)])
        ################################
        ## Create Kinematic Variables ##
        ################################
        T_mu,E_mu,P_mu,E_nu,cos_mu,DELTA_cos_mu,DELTA_T_mu = make_variables(N_T,N_cos,E_nu_array[m])

        for l in range(num_bins):
            for i in range(N_T):
                for j in range(bin_size):
                    cos_bin[i][j] = cos_mu[i][j+l*bin_size]

            #####################################################################
            ## Create RFG Variables #############################################
            ## For the last entry, put 1 for neutrinos, or 2 for antineutrinos ##
            #####################################################################
            #double_diff = make_double_diff_BW(E_mu,E_nu,P_mu,cos_bin,GAMMA,M_A,2)
            double_diff = make_double_diff_dipole(E_mu,E_nu,P_mu,cos_bin,M_A,1)
            double_diff = where(cos_bin >= cos(20.*pi/180.),double_diff,0.) 

            ## apply the angle cut of Minerva ##
            #double_diff = where((arccos(cos_bin)*180/pi <= 20) & (arccos(cos_bin)*180/pi >= -20), double_diff, 0.)

            #############################
            ## Calculate Cross Section ##
            #############################
            SIGMA_Temp,new_E_nu = calc_cross_section(E_nu,N_T,int(N_cos/100),DELTA_cos_mu,DELTA_T_mu,double_diff)

            #################################
            ## Add to total value of SIGMA ##
            #################################
            SIGMA[m] = SIGMA[m]+SIGMA_Temp
    ## uncomment these below if  you want to  get a rough idea  what the graph should look like, with lower number of T_mu, cos_mu points ##.
    for j in range(100):
        for i in range(len(E_nu_array)):
            if E_nu_array[i] >= 12.:
                SIGMA[i] = SIGMA[i-1]
    print(SIGMA)
    return SIGMA

#################################
## make sdcs for fitting BW FF ##
#################################
def make_single_diff(Q2_passed,N):
    ## parameters ##
    num = 12                                                # number of Nucleons
    cos = 0.974                                             # cosine of cabibbo angle
    M_A = 1.35

    Func = interp1d(E_nu,Flux,kind='cubic')

    num_flux = 1000
    E_nu_new = linspace(0.02,3.,num_flux,endpoint=True)
    Flux_new = Func(E_nu_new)

    Total_Flux = 0
    for i in range(len(Flux_new)):
        Total_Flux = Total_Flux + (3.-.02)/num_flux*(Flux_new[i])

    Q2,E_nu_new = meshgrid(Q2_passed,E_nu_new,indexing='ij')
    N_Q = len(Q2_passed)

    ## Define the weight functions needed to integrate over the flux ##
    weight = ((3.-.02)/num_flux)*(Flux_new/Total_Flux)/(num/2)
    weight = broadcast_to(weight,(N_Q,num_flux))

    #Q2 = broadcast_to(Q2_passed,(N_E,N_Q))
    #Q2 = swapaxes(Q2,0,1)
    #E_nu = broadcast_to(E_nu_new,(N_Q,N_E))

    Q2 = Q2_passed
    E_nu_new = 2.5

    ## calculate the form factors ##
    F_1,F_2,F_A,F_P,M_A = make_form_factors_dipole(Q2,M_A)

    ## Define some convenient variables for single differential cross section ##
    ## x = (s-u)  ##
    x = 4.0*m_N*E_nu_new - Q2 - sq(m_mu)
    y = Q2/(4.*sq(m_N))
    A = (m_mu**2 + Q2)/m_N**2 * ( (1. + y)*sq(F_A) - (1.-y)*sq(F_1) + (y-y**2)*sq(F_2) + 4*y*F_1*F_2 - (m_mu/2/m_N)**2*( sq(F_1 + F_2) + sq(F_A + 2.*F_P) - (Q2/m_N**2 + 4.)*sq(F_P) ))
    B = Q2/m_N**2*F_A*(F_1+F_2)
    C = 0.25*( sq(F_A) + sq(F_1) + y*sq(F_2))

    R = 1.
    #R = (1. + 6.*Q2*exp(-Q2/0.34))

    ## Calculate the single differential XS ##
    single_diff = N*((sq(m_N)*sq(G_F)*sq(cos))/(8.*pi*sq(E_nu_new)*sq(GeV_To_Cm)))*(A - x*B/sq(m_N) + sq(x)*C/sq(sq(m_N)))*R
    #single_diff = ((sq(m_N)*sq(G_F))/(8.*pi*sq(E_nu_new)*sq(GeV_To_Cm)))*(A + x*B/sq(m_N) + sq(x)*C/sq(sq(m_N)))

    #single_diff = weight_sum_2d(single_diff,weight).real

    return single_diff

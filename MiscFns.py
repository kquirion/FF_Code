## misc functions for calculating stuff ##
import numpy as np
import matplotlib.pyplot as plt
from pylab import subplot


###############################################################################
## Create a function for rounding to 4 significant figures (for readability) ##
###############################################################################
def roundSig(x, sig=3):
    if x == 0.:
        return 0.
    else:
        return round( x, sig - int( np.floor( np.log10( abs( x )) ) ) - 1 )
############################################################################
## make a function to convert 1D array into a 2D array of dimension (x,y) ##
############################################################################
def make2d(x,y,vector):
    a = len(x)
    b = len(y)
    vector2 = np.array([[0.0 for j in range(b)] for i in range(a)])
    for i in range(a):
        for j in range(b):
            vector2[i][j] = vector[i*b+j]
    return vector2
####################################################################
## Create a function to calculate point-wise and total xi squared ##
####################################################################
def CalcChiSq(Theo,Exp,Err):
    ChiSqInd =  np.array((( Theo.ravel() - Exp.ravel() ) / Err.ravel() )**2 )
    ChiSqTot = sum(ChiSqInd)
    return ChiSqInd,ChiSqTot
##################################################################################
## Create a function that performs a weighted sum over the last axis of 'vector'##
##################################################################################
def WeightSum3D(vector,weight):
    a = vector.shape
    x = a[0]
    y = a[1]
    z = a[2]
    output = np.array([[ 0. for i in range(y)] for j in range(x) ] )
    for i in range(x):
        for j in range(y):
            Int = 0
            for k in range(z-1):
                Int +=  weight[i][j][k] * vector[i][j][k] 
            output[i][j] = Int
    return output
##################################################################################
## Create a function that performs a weighted sum over the last axis of 'vector'##
##################################################################################
def WeightSum2d(vector,weight):
    a = vector.shape
    x = a[0]
    y = a[1]
    output = np.array( [0. for i in range(x) ] )
    for j in range(x):
        Int = 0
        for k in range(y):
            Int += weight[j][k] * vector[j][k] 
        output[j] = Int
    return output











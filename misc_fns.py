## misc functions for calculating stuff ##

from math import log10, floor
from numpy import array,linspace,longdouble,where,sqrt,broadcast_to,swapaxes,log,power,nanmin,nanmax,conjugate,sum,maximum,minimum,empty,meshgrid,arccos,amin,amax,exp,zeros,logspace,log10,cos
from math import pi
from scipy.integrate import quad
from sys import exit
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

###############################################################################
## calculates the square of a number, or the element-wise square of an array ##
###############################################################################
def sq(A):
    return A*conjugate(A)

###############################################################################
## Create a function for rounding to 4 significant figures (for readability) ##
###############################################################################
def round_sig(x, sig=3):
    if x == 0.:
        return 0.
    else:
        return round(x, sig-int(floor(log10(abs(x))))-1)

############################################################################
## make a function to convert 1D array into a 2D array of dimension (x,y) ##
############################################################################
def make2d(x,y,vector):
    a = len(x)
    b = len(y)
    vector2 = array([[0.0 for j in range(b)] for i in range(a)])
    for i in range(a):
        for j in range(b):
            vector2[i][j] = vector[i*b+j]
    return vector2

####################################################################
## Create a function to calculate point-wise and total xi squared ##
####################################################################
def calc_chi_squared(Theo,Exp,Exp_err):
    chi_squared_individual =  array(sq((Theo.ravel()-Exp.ravel())/Exp_err.ravel()))
    chi_squared = sum(chi_squared_individual)
    return chi_squared_individual,chi_squared

##################################################################################
## Create a function that performs a weighted sum over the last axis of 'vector'##
##################################################################################
def weight_sum_3d(vector,weight):
    a = vector.shape
    x = a[0]
    y = a[1]
    z = a[2]
    output = array([[0.0 for i in range(y)] for j in range(x)])
    for i in range(x):
        for j in range(y):
            Int = 0
            for k in range(z-1):
                Int = Int + (weight[i][j][k]*vector[i][j][k])
            output[i][j] = Int
    return output

def weight_sum_2d(vector,weight):
    a = vector.shape
    x = a[0]
    y = a[1]
    output = array([0.0 for i in range(x)])
    for j in range(x):
        Int = 0
        for k in range(y-1):
            Int = Int + 0.5*(weight[j][k]*vector[j][k] + weight[j][k+1]*vector[j][k+1] )
        output[j] = Int
    return output

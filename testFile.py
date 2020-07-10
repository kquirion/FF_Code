from scipy.interpolate import interp1d
from scipy.integrate import quad
import numpy as np
import matplotlib.pyplot as plt
import collections
from pylab import subplot



xs = [[1,2,3,4,5,6,7,8,9],[10,20,30,40,50,60,70,80,90]]
ys = [[x**2 for x in row] for row in xs]

for i in range(len(xs)):
    ax1 = subplot( len(xs), 1, i+1 , snap=False)
    ax1.scatter(xs[i],ys[i])
    
plt.show()
    









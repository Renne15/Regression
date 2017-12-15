import numpy as np
import json
import matplotlib.pyplot as plt
from math import exp, erfc, sqrt

c = 1.
D = 5.

t_size = 100
x_size = 100
phi = np.zeros((t_size,x_size))

for t in range(t_size):
    for x in range(x_size):
        if t == 0:
            phi[0,x] = 0.
        elif x == 0:
            phi[t,x] = 1.
        else:
            phi[t,x] = 0.5 * exp(x*c/(2*D)) * ( exp( -x*c/(2*D) ) * erfc( (x-c*t)/(2*sqrt(D*t)) )
                                            + exp( x*c/(2*D) ) * erfc( (x+c*t)/(2*sqrt(D*t)) ) )

for t in range(t_size):
    plt.plot(range(x_size),phi[t,:])
    plt.ylim([0,1])
    plt.draw()
    plt.pause(0.01)
    plt.clf()

filepath = './data/data_1D.csv'
np.savetxt(filepath, phi, delimiter=',')

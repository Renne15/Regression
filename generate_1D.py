import numpy as np
import json
import matplotlib.pyplot as plt
from math import exp, erfc, sqrt

c = 1.
D = 5.

x_split = 0.1
t_split = 0.1
t_size = 5000
x_size = 5000
phi = np.zeros((t_size,x_size))

for t in range(t_size):
    for x in range(x_size):
        if t == 0:
            phi[0,x] = 0.
        elif x == 0:
            phi[t,x] = 1.
        else:
            phi[t,x] = 0.5 * np.exp(x*x_split*c/(2*D)) * ( np.exp( -x*x_split*c/(2*D) ) * erfc( (x*x_split-c*t*t_split)/(2*sqrt(D*t*t_split)) )
                                            + np.exp( x*x_split*c/(2*D) ) * erfc( (x*x_split+c*t*t_split)/(2*sqrt(D*t*t_split)) ) )

# ### 移流拡散のプロット
# for t in range(t_size):
#     plt.plot( [x_split*x for x in range(x_size) ], phi[t,:] , lw=5)
#     plt.ylim([0,1])
#     plt.draw()
#     plt.pause(0.001)
#     plt.clf()

filepath = './data/data_1D.csv'
np.savetxt(filepath, phi, delimiter=',')

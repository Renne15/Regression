import numpy as np
import json
import matplotlib.pyplot as plt

c = -1.
D = 5.

phi(t,x) = 0.5 * exp(x*c/(2*D)) * ( exp( -x*c/(2*D) ) * erfc( (x-c*t)/(2*sqrt(D*t)) )
                                  + exp( x*c/(2*D) ) * erfc( (x+c*t)/(2*sqrt(D*t)) ) )

# filePath = './data/1D_data.json'
#
# with open(filePath, 'w') as outfile:
#     json.dump(dataList, outfile)

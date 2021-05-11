# -*- coding: utf-8 -*-
"""
Created on Wed May  5 16:17:09 2021

@author: xavier.mouy
"""
import os
import sys
sys.path.append(r'C:\Users\xavier.mouy\Documents\GitHub\ecosound') # Adds higher directory to python modules path.

from ecosound.core.tools import derivative_1d, find_peaks, envelop
from numpy import array, sign, zeros
from scipy.interpolate import interp1d
from matplotlib.pyplot import plot,show,grid

s = array([1,4,3,5,3,2,4,3,4,5,4,3,2,5,6,7,8,7,8]) #This is your noisy vector of values.

q_u = zeros(s.shape)
q_l = zeros(s.shape)

# #Prepend the first value of (s) to the interpolating values. This forces the model to use the same starting point for both the upper and lower envelope models.

# u_x = [0,]
# u_y = [s[0],]

# l_x = [0,]
# l_y = [s[0],]

# #Detect peaks and troughs and mark their location in u_x,u_y,l_x,l_y respectively.

# u_x, u_y = find_peaks(s,troughs=True)
# l_x, l_y = find_peaks(s,troughs=False)

# #Append the last value of (s) to the interpolating values. This forces the model to use the same ending point for both the upper and lower envelope models.
# u_x.append(len(s)-1)
# u_y.append(s[-1])
# l_x.append(len(s)-1)
# l_y.append(s[-1])

# #Fit suitable models to the data. Here I am using cubic splines, similarly to the MATLAB example given in the question.
# u_p = interp1d(u_x,u_y, kind = 'cubic',bounds_error = False, fill_value=0.0)
# l_p = interp1d(l_x,l_y,kind = 'cubic',bounds_error = False, fill_value=0.0)

# #Evaluate each model over the domain of (s)
# for k in range(0,len(s)):
#     q_u[k] = u_p(k)
#     q_l[k] = l_p(k)

q_u, q_l = envelop(s, interp='cubic')

#Plot everything
plot(s)
plot(q_u,'r')
plot(q_l,'g')
grid(True);show()
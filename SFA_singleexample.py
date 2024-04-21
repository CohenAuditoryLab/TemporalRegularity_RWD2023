# -*- coding: utf-8 -*-
"""
Demonstration of SFA and the other algorithms using illustrative example from 
Bellec et al. 2015 paper on SFA and STDP kernels.

Creates input signal, runs selected method, and then plots original input signal 
and three features of selected method.

When applying linear SFA should see the slow sine wave component as slowest feature.


2023-03-13
Code by RW DiTullio
"""

#%% Imports

import numpy as np
import matplotlib.pyplot as plt 
from SFA_Tools.SFA_Func import getSF

#%% Toggle selected algorithm

mode = 'Linear' #options PCA, ICA, Linear, quad.  Quad results are less interpretable since signal is first quadratically expanded.

#%% Set up input signals

    
#Signal properties in seconds
tres= 1.0/50000.0 #sampling rate of 50 kHZ used
T = .250 #250 ms
t=np.arange(0,T,tres)

n_scale = 0.0 #scale noise contribution

#parameters from example
f0 = 10;
f2 = 11; #ratio for cos
alpha = .5;

pi = np.pi #just laziness
#Create five input channels: update these to be all on same scale so noise works roughly accordingly.

x1 = np.sin(2*pi*f0*t) + alpha*np.cos(2*pi*f2*f0*t)**2 +n_scale*np.random.normal(0,1,t.size)
x2 = np.cos(2*pi*f2*f0*t) + n_scale*np.random.normal(0,1,t.size)
x3 = x1**2
x4 = x1*x2
x5 = x2**2

#create matrix in non-standard form (var x obs ) as used this in the rest of code
X = np.vstack([x1,x2,x3,x4,x5]) 



    
#%%  Run chosen analysis



retain = np.size(X,0) #Grab as many slow features as there are o.g. features
#NOTE above assumes that using non-standard form (var x obs instead of obs x var)


(transformed, mean, variance, data_SS, weights) = getSF(X,'Layer 1',mode,retain,transform = True)


#%% Plot result

#Setup code to plot og signal

plt.close('all')

for sig in range(0,np.size(X,0)):
    plt.figure(sig)
    plt.plot(t,X[sig,:])
    plt.title('Input signal # ' + str(sig+1))
    
feat2use = [0,1,2]
#plot slowest feature
for f in feat2use:
    plt.figure(sig+f+1)
    plt.plot(t,transformed[f,:])
    plt.title( mode + ' Feature # ' + str(f+1))
    
    
        


    
        
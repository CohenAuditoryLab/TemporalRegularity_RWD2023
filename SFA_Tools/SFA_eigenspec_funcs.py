# -*- coding: utf-8 -*-
"""

Helper module for analysis looking at the spectra of eigen values between SFA and
PCA.  Main use now is just single helper function for SFA_Func - PCA_eigens

2023-03-13
Code by RW DiTullio


"""

import numpy as np
import matplotlib.pyplot as plt 
import soundfile as sf 
import pyfilterbank.gammatone as g
import scipy.ndimage.filters as filt
import SFA_Tools.SFA_Sets as sf_s
import SFA_Tools.SFA_Func as sf_f



def coch_filter(signal,fs): #note: can add flexibility to this later for now hardcoding parameters from other code


    gfb = g.GammatoneFilterbank(samplerate= fs, order=4, density = 1.0, startband = -21, endband = 21, normfreq = 2200) #sets up parameters for our gammatone filter model of the cochlea.
    
    tFilter = sf_f.temporalFilter() #intial temporal filter with default parameters
    tFilter2 = np.repeat(tFilter,3)/3 #make second, wider filter (o.g factor is 3, testing impact of making this wider on eigen spec)
    tFilters = [tFilter]

    #Add toggle to downsample here if needed
    
    transformed = sf_f.gamma_transform(signal, gfb)
    transformed = sf_f.temporal_transform(transformed,tFilters)
    
    signal_cf = transformed
    
    ds_trig = False #adding toggle to look at effect of down sampling
    
    if ds_trig:
        
        ds_fac =1
        
        signal_cf = signal_cf[:,::ds_fac]
        
    
    
    return signal_cf


def gen_Z(X,fs, mode): 
    
    
    signal_cf = coch_filter(X,fs)
    
    (signal_normalized,mean,variance) = sf_s.norm(signal_cf) #keep output of mean and var for now if need to adapt to training and testing case
    
    #Would add quad expansion here if desired
    if (mode == 'quad'):
        signal_normalized = sf_s.quadExpand(signal_normalized)
    
    #Check Misclocal for reason to trust implementation as is.
    (signal_sphered, sphereingstage) = sf_s.PCA(signal_normalized)
    
    Z = np.diff(signal_sphered, axis = 1) #double check axis is correct
    
    return Z,signal_normalized

def PCA_eigens(mat):
    
    
    #kind of pointless to have this a separate function but leave it for now
    #for own logic
    
    #still following implementation in other code for consistency
    cov_mat = np.cov(mat)
    cov_mat = cov_mat + np.identity(cov_mat.shape[0]) *0.0000001
    U,S,Vh = np.linalg.svd(cov_mat)
    #in this application S is already the eigen values and no extra scaling is needed
    #U are the weights and in standard form of each column is a Feature/PC

    return U,S
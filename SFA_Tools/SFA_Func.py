"""

Module containing core functions called by ***_runSFA


03 13 2023
Code by R.W. DiTullio and C Parthiban 



"""


import numpy as np
import matplotlib.pyplot as plt 
import soundfile as sf 
import scipy.ndimage.filters as filt
from tqdm import tqdm
import SFA_Tools.SFA_Sets as s
import SFA_Tools.SFA_eigenspec_funcs as s_eigen
from sklearn.decomposition import FastICA as ICA


# Read and load vocalizations from a list of files
def get_data(file_list):
    vocalizations = [] #treating vocalizations as a list
    rate = 0
    for f in file_list:
        vocal, rate = sf.read(f) 
        #note sf reads assuming that there is NOT a leading /
        #rate is the original sampling rate that sf.read pull from the file
        vocalizations.append(vocal) 
    return vocalizations

# Compute the average power of a signal
def signal_power(sig):
    return np.sum(np.square(sig))/sig.size

# Scales a noise vector to have a specified signal-to-noise ratio with
# another waveform
def scale_noise(vocalizations, noise, ratio):
    data = np.zeros(1)
    for vocal in vocalizations:
        data = np.hstack((data,vocal))
        
    initial_ratio = signal_power(data)/signal_power(noise[:data.size])
    return noise * np.sqrt(initial_ratio/ratio)

# Applies a gammatone transform with a given filter bank to a waveform
def gamma_transform(data, gfb): #this is based on a somewhat poorly documented pyfilterbank code
    #http://siggigue.github.io/pyfilterbank/gammatone.html
    #data is vocalizations list 
    analysed = gfb.analyze(data)
    
    transformed = np.zeros((len(gfb.centerfrequencies),data.size))
    #gfb.centerfrequencies returns the center frequencies of each band in the gfb object in hz
    for i in range(len(gfb.centerfrequencies)):
        (band,state) = analysed.__next__()
        transformed[i] = abs(band)
        
    return transformed

# Applies a gammatone transform with given filter bank to a list of 
# different waveforms
def gamma_transform_list(data, filterbank):
    transformed = []
    
    for d in tqdm(data): # note tqdm is just something to generate a progress bar
        d_transformed = gamma_transform(d, filterbank)
        transformed.append(d_transformed)
        
    return transformed

# Plots gammatone transformed vocalizations effectively
def plot_input(inp, name):
    plt.figure(figsize=(12,3))
    plt.title(name)
    plt.imshow(inp, aspect = 'auto', origin = 'lower')
    plt.show()
    return

# Gamma Function
def gamma(n, a, b, m):
    arr = np.arange(1,n+1)
    return a*np.power(arr,m)*(np.exp(-b*arr))

# Creates a temporal filter
def temporalFilter():
    arr = gamma(400,1.5,0.04,2) - gamma(400,1,0.036,2) 
    arr = arr / np.var(arr)
    return arr

# Applies list of temporal filters to a transformed vocal
def temporal_transform(data,filters):
    transformed = None
    init = True
    for f in filters:
        filtered = filt.convolve(data,f[:,None].T)
        if(init):
            transformed = filtered
            init = False
        else:
            transformed = np.vstack((transformed,filtered))
            
    return transformed

# Applies list of temporal filters to a list of transformed vocals
def temporal_transform_list(data,filters):
    transformed = []
    
    for d in tqdm(data):
        d_transformed = temporal_transform(d, filters)
        transformed.append(d_transformed)
        
    return transformed

# Applies SFA To Data
def getSF(data,name, mode, retain = 20, transform = False):
    (data_normalized,mean,variance) = s.norm(data)
    print(name, ': Normalization Complete...')
    
    if mode == 'quad':
        data_normalized = s.quadExpand(data_normalized)
        print(name, ': Nonlinear Expansion Complete...')
    
    if mode == 'PCA': #note don't need to rerun as luckily the PCA stuff was the last call so it would work as expected
    #Now need to have the returns that break the function otherwise will run else as long as any one of the modes are not done.
        
        #Use same thing we used to get eigenvalues to plot from SFA_eigenspec_funcs
        #Not using weights function since this assumes trying to grab smallest eigen values
        [weights, eigensval] = s_eigen.PCA_eigens(data_normalized)
        weights = weights[0:retain]
        print(name, ': Weights Determined...')
        
        data_SS = [] #just set data_SS to an empty list as won't be used if using PCA
        if(transform):
            transformed = weights @ data_normalized
            return transformed, mean, variance, data_SS, weights
        else:
            return mean, variance, data_SS, weights
            
    if mode == 'ICA': 
        ICAmodel = ICA(n_components = 3, whiten='unit-variance', max_iter = 2000)
        transformed = ICAmodel.fit_transform(data_normalized.T) 
        #note: ICA is not always converging with this so will need to adjust things or tolerate this for now
        weights = ICAmodel.components_
        data_SS = []
        
        if(transform):
            
            return transformed.T, mean, variance, data_SS, weights
        else:
            return mean, variance, data_SS, weights
        
    else:
    
        (data_Sphered,data_SS) = s.PCA(data_normalized) #note: data_SS is the matrix of the sphering stage which is related to the used PCs of the data
        print(name, ': Sphering Complete...')
        
        weights = s.weights(data_Sphered, retain)    
        weights = np.flip(weights.T,0)
        print(name, ': Weights Determined...')
        #further note: weights @ data_SS will give you weights in quadexpanded space, the first data.shape[0] rows are the linear terms, the rest are the quad terms.
        if(transform):
            transformed = weights @ data_Sphered
    #regardless of method either apply weights or return normalization vars    
    if(transform):
        
        return transformed, mean, variance, data_SS, weights
    else:
        return mean, variance, data_SS, weights
    
def getSFNonNeg(data,name, mode = 'quad', retain = 20, transform = False):
        
        'Does not work'
        
        return 
    
 
    
    

# Tests SFA on Data
def testSF(data, name,mode, mean, variance, SS, weights):
    data_normalized = s.normTest(data, mean, variance) 
    print(name, ': Normalization Complete...')
    
    if (mode == 'quad'):
        data_normalized = s.quadExpand(data_normalized)
    print(name, ': Nonlinear Expansion Complete...')
    
    if mode == 'PCA':
        output = weights @ data_normalized
        return output
    if mode == 'ICA':
        output = weights @ data_normalized
        return output
    else:
        data_Sphered = s.PCATest(data_normalized,SS)
        print(name, ': Sphering Complete...')
    
        output = weights @ data_Sphered #regardless of method do projection
    
    return output #this is a post python 3.5 short hand for matrix multiplication

# Get Labels For Data
def getlabels(data):
    labels  = None 
    initialized = False

    for i,d in enumerate(data):
        if(not(initialized)):
            labels = np.zeros(d[0].size)
            initialized = True
        else:
            nextlabel = np.ones(d[0].size) * i
            labels = np.hstack((labels,nextlabel))
    
    #     for i in range(len(data)):
    #     if(not(initialized)):
    #         labels.append(np.zeros(data[0].size))
    #         initialized = True
    #     else:
    #         labels.append(np.ones(data[i].size)*i)
    # labels = np.hstack(labels)
    return labels

# SFA Plot Classifiers
def SFAClassifiedPlot(features,classifier, labels, n = 500, figure_size = (10,7)):
    x_min, x_max = features[0].min() - 1, features[0].max() + 1
    y_min, y_max = features[1].min() - 1, features[1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, n), np.linspace(y_min, y_max, n))
    
    arr = np.c_[xx.ravel(), yy.ravel()]
    Z = classifier.predict(arr)
    Z = Z.reshape(xx.shape)
    
    labelset = list(set(labels))
    pos = []
    for label in labelset:
        positions = [i for i,x in enumerate(labels) if x == label]
        pos.append(positions)
        
    plt.figure(figsize=figure_size)
    plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.8)
    for i,p in enumerate(pos):
        plt.scatter(features[0][p][::10], features[1][p][::10], c = 'C' + str(int(labels[p[0]])), cmap=plt.cm.Paired)
        
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.show() 
    return

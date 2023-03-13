"""

Helper Module containing functions called by SFA_Func


03 13 2023
Code by R.W. DiTullio and C Parthiban 


"""


import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import scipy.stats as stat
from sklearn.decomposition import NMF

############################################################################
######### Library for Conducting SFA on a 2d Matrix For Test Use  ##########
############################################################################

# Matrix should be in form np.array([[x(t)],[y(t)]...[z(t)]])

# Returns Matrix with 0 mean and Variance =1
def norm(arr):
	m = np.mean(arr,1)
	arr = arr-m[:,None]
	var = np.sqrt(np.mean(np.square(arr),1) + 1e-11) 
	arr = arr/var[:,None]
	return (arr,m,var)

def normTest(arr,m,var):
	arr = arr-m[:,None]
	arr = arr/var[:,None]
	return arr

# Returns quadratic expansion of Matrix

#Function that include auto corr terms

# # Returns quadratic expansion of Matrix
# # Ex. matrix [x1,x2,x3] -> matrix [x1,x2,x3,x1x1,x1x2,x2x2,x1x3]
# def quadExpand(arr):
# 	s = arr.shape
# 	l = int(s[0] + s[0]*(s[0]+1)/2)
# 	out = np.zeros((l,s[1]))
# 	count = 0
# 	for i in range(s[0]):
# 		out[i] = arr[i]
# 		count = count + 1
# 	for i in range(s[0]):
# 		for j in range(i+1):
# 			out[count] = np.multiply(arr[i],arr[j])
# 			count = count + 1
# 	return out

# Ex. matrix [x1,x2,x3] -> matrix [x1,x2,x3,x1x2,x1x3,x2x3,...]
def quadExpand(arr):
	s = arr.shape
	l = int(s[0]*(s[0]+1)/2)
	out = np.zeros((l,s[1]))
	count = s[0]
	out[0:s[0],:] = arr #set first rows to all of the linear terms as indicated above
	for i in range(s[0]):
		for j in range(i): #2022-11-07 apparently this got written without the necessary +1
			out[count] = np.multiply(arr[i],arr[j])
			count = count + 1
	return out

# Conducts PCA Whitening on a Matrix to return matrix  
# with 0 mean and identity covariance 
# Also Removes Redundant Eigenvectors (eig < 10^-9)
def PCA(arr): #PCA implemented through modified svd procedure, see code in MiscLocal if need refreseher
	m = np.mean(arr,1)
	arr = arr-m[:,None]
	cov = np.cov(arr)
	cov = cov + np.identity(cov.shape[0]) *0.0000001 #2020-06-13 I guess this ensures that there is some positive variance for all terms...
	U, S, Vh = np.linalg.svd(cov) 
	where = np.argwhere(  S > 1.1e-8)  #Update 2022-07-11: just using better temporal filter prevents this from hanging when smaller limits are used.
    #i.e. can set to 1.1e-4 and still find SFA eigen values of e-7 so original concern about eigenspectra is gone
    
	siz = where.size 
	S = S[:siz]
	S = 1/np.sqrt(S) #to set var to 1 after decorr
	S = S * np.identity(S.size)
	SS = np.matmul(S,U.T[0:siz,:]) #SS is sphering stage
	xwhite = np.matmul(SS,arr) #given this line it looks like SS is the matrix of principal component analysis
	return (xwhite,SS)

def Non_negative(): # figuring out where exactly to put non-negativity constraint.  
    ''
    return

def PCATest(arr,SS): 
	return np.matmul(SS,arr)

# Returns weight vectors for SFA input-output functions

def weights(arr, retain, mode = 'retain'):
	dx = np.diff(arr, axis = 1)
	dcov = np.cov(dx)
	dcov = dcov + np.identity(dcov.shape[0]) *0.0000001
	U, S, Vh = np.linalg.svd(dcov) 
	if (mode ==	 'retain'):
		U = U[:,U.shape[1]-retain:U.shape[1]] #take only 20 SF (i.e. retain = 20) 
	return U

def weights_non_neg(arr, retain, mode = 'retain'):
    #Same setup as above, get derivatve then use NMF to get components
    #Update 2021-05-18:
    #Made a silly mistake in how we thought NNMF actually worked. 
    #Can only do decomp of matrices that are already non-negative matrices
    #Thus I think this has to go in the PCA stage as the temporal differences have to be allowed to be negative
    #Rethinking this and perhapes it does make sense to try to apply non-negativity constraint here.
    dx = np.diff(arr, axis = 1)
    dcov = np.cov(dx)
    dcov = dcov + np.identity(dcov.shape[0]) *0.0000001
    nmf_model = NMF( init='nndsvd')
    nmf_model.fit(dcov)
    components = nmf_model.components_
    if (mode == 'retain'):
        components = components[:,components.shape[1]-retain:components.shape[1]]
    
    return

def SFAquad_WithoutTools(arr,j):
	xnorm = norm(arr)
	xquad = quadExpand(xnorm)
	xpca = PCA(xquad)
	xweights = weights(xpca, j)
	return np.matmul(xweights.T, xpca)

def SFA_Reverse(arr,S,weights):
	print(S.shape)
	print(weights.shape)
	w = np.matmul(weights.T,S)
	print(w.shape)
	inv = np.linalg.inv(np.matmul(weights,S))
	arr_base = np.matmul(inv,arr)
	return arr_base

def chirp(l, r , a, f, d):
    out = np.arange(0, l/r,step = 1/r)
    x = (a*(f/d))*np.exp(-1*f*out/d)*np.sin(2*np.pi*f*out)
    return x

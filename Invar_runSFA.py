# -*- coding: utf-8 -*-
"""
Same things as Clutter_runSFA but for generalization results.  Loads in multiple files
no noise applied

@author: ronwd
"""

import numpy as np
import matplotlib.pyplot as plt 
import soundfile as sf 
import pyfilterbank.gammatone as g
import scipy.ndimage.filters as filt
from sklearn import svm
from sklearn.linear_model import Perceptron
from tqdm import tqdm
import SFA_Tools.SFA_Sets as s
from SFA_Tools.SFA_Func import *
from pathlib import Path, PureWindowsPath
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_validate, ShuffleSplit


def run_SFA(vocal_foldername,vocal_pair,training_set,run_nulls):
    
    mode = 'ICA'
    
#Save SFA classification accuracy and the baseline model accuracy
    SFA_save_score = np.zeros((5,training_set.size)) 
    Baseline_save_score = np.zeros((5,training_set.size)) 
    
    #grab test file vocals since these are the same across training groups
    voc1 = vocal_foldername / vocal_pair[0][0]
    voc2 = vocal_foldername / vocal_pair[1][0]
    #convert to windows path i.e using \ instead of /
    voc1 = PureWindowsPath(voc1)
    voc2 = PureWindowsPath(voc2)

#set up list of vocal files to load for testing data
    vocal_files = [voc1, voc2]
    

    for training_group in range(0,training_set.size): #for each training set
     for a_round in range(0,5): #rounds are for averaging for each training group
        print(a_round)
    
        #Unfortunately don't really have an elegent solution for setting up all of the training files we want to pull...so trying below
        training_files = list()
        for i in range(0, training_set[0].size): #grab all vocal files in that training set (10 in our case as of 2020-08-18)
        #Note training set is now [set][vocal][0][0]
      
            t_voc = vocal_foldername / training_set[training_group][i][0][0]
            t_voc = PureWindowsPath(t_voc)
            
            training_files.append((t_voc))
            
        'from here on we have to be careful to make sure things match up properly'
        
        #set number of vocals to the training vocals since this is called throughout code.  May have to have another variable for testing vocals though
        num_vocals = len(training_files) 
        num_test_vocals = len(vocal_files)
        
        #set up gammatone filter
        gfb = g.GammatoneFilterbank(samplerate= 50000, order=4, density = 1.0, startband = -21, endband = 21, normfreq = 2200)
        
        down_sample = True #down sample stimulus to help with RAM issues and general dimensionality issues.  I believe mostly reduces resolution of frequency
     
     
     
        down_sample_pre = 2 #Factor by which to reduce Fs by (e.g. 10 reduces Fs by one order of magnitude) 
     
        down_sample_post = 2 #Factor by which to reduce Fs after applying filters 
        
    
     
         ##Training and testing data parameters  
        num_samples = num_vocals * 1
        gaps = True #toggle whether there can be gaps between presentation of each stimulus
        
        #skipping noise parameters for now
        #Set up classifiers
        classifier_baseline = SGDClassifier(max_iter = 100000, tol = 0.001) #Perceptron(max_iter = 10000, tol = 0.001) #from scikit (presumably) #set up Perceptron classifier
        classifier_SFA = SGDClassifier(max_iter = 100000, tol = 0.001) #Perceptron(max_iter = 10000, tol = 0.001)
        
   
        classifier_features = 5 #how many features from mode classifer gets to use
        baseline_features = 'all' #how many features the baseline model gets to use
     
    
         
        ## Load in files
    
        vocalizations = get_data(training_files) #get list object where each entry is a numpy array of each vocal file
        testvocalizations = get_data(vocal_files)
        print('Vocalizations Loaded...')
        
        ## Apply Gammatone Transform to training and test
        
        vocals_transformed = gamma_transform_list(vocalizations, gfb) #does what is says on the tin: applies gamma transform to list of numpy arrays
        
        testvocals_transformed = gamma_transform_list(testvocalizations, gfb) 
        print('Vocalizations Transformed...')
        
    
         ## Down sample for computation tractablility
        
         
        if down_sample:
              for i,vocal in enumerate(vocals_transformed):
                  vocals_transformed[i] = vocal[:,::down_sample_pre]
              for i, vocal in enumerate(testvocals_transformed):
                  testvocals_transformed[i] = vocal[:,::down_sample_pre]
                  
        print('Ready For Temporal Filters')
         
          
        tFilter = temporalFilter()
      
        tFilters = [tFilter]
         
        vocals_temporal_transformed = temporal_transform_list(vocals_transformed,tFilters)
        testvocals_temporal_transformed = temporal_transform_list(testvocals_transformed,tFilters)
        print('Vocals Temporally Filtered...')
         
        if down_sample:
            for i,vocal in enumerate(vocals_temporal_transformed):
                vocals_temporal_transformed[i] = vocal[:,::down_sample_post] 
            for i,vocal in enumerate(testvocals_temporal_transformed):
                testvocals_temporal_transformed[i] = vocal[:,::down_sample_post]
            
    
        samples = np.random.choice(num_vocals, num_samples, replace=False) #Have to switch to using random.choice so can remove replacement.  This means we can remove while loop too
        print('Equal presentation of vocalizations established') #note this mainly works because we are presenting each vocal once.  If that was not the case we would have to set up some kind of loop or additional code
        
        training_data = None
        initialized = False
        for i in tqdm(samples):
            if(not(initialized)):
                       training_data = vocals_temporal_transformed[i]
                       initialized = True
            else:
                       training_data = np.concatenate((training_data, vocals_temporal_transformed[i]),1)
                       
            if(gaps):
                       min_gap = np.round(.05 * vocals_temporal_transformed[0].shape[1]) #sets min range of gap as percentage of length of a single vocalizations
                       max_gap = np.round(.5 * vocals_temporal_transformed[0].shape[1]) #set max range of gap in same units as above
                       training_data = np.concatenate((training_data, np.zeros((training_data.shape[0], np.random.randint(min_gap,max_gap)))),1)     
        print('Data arranged...')
        
        print('No Noise Applied...')    
        
        print('Ready For SFA')
    
                
        
        
        (layer1, mean, variance, data_SS, weights) = getSF(training_data, 'Layer 1',mode, transform = True)
        print('SFA Training Complete')
   
        
        ## Test Results 
        
        samples = np.arange(num_test_vocals)
        
        testing_data = None
        initialized = False
        for i in tqdm(samples):
            if(not(initialized)):
                testing_data = testvocals_temporal_transformed[i]
                initialized = True
            else:
                testing_data = np.concatenate((testing_data, testvocals_temporal_transformed[i]),1) 
        print('Data arranged...')
         
        print('No Noise Applied...')
        
         
        print('Testing Data Ready')
    
        ## Apply SFA to Test Data, also toggles for using second layer
        
        test = testSF(testing_data, 'Layer 1', mode,mean, variance, data_SS, weights)
        print('SFA Applied To Test Set')
        
        ## If null runs, shuffle the labels.  Note can adjust this like we did for equal presentation if we end up with more than two test vocals.
        if run_nulls:
         labels = np.random.randint(0,2,test.shape[1]) #just put in random labels.
        else:
         labels = getlabels(testvocals_temporal_transformed)
        
        
        
        the_cv = ShuffleSplit(n_splits = 30, test_size = 0.99)
        
        print(mode + ' Based Classifier with ', classifier_features, ' features')
         #add a cv loop to pin down variance
        cv_sfa = cross_validate(classifier_SFA, test[0:classifier_features,:].T, labels,cv=the_cv)
      
        print(cv_sfa['test_score'])
        print('Mean CV ', np.mean(cv_sfa['test_score']))
         
        SFA_save_score[a_round,training_group] = np.mean(cv_sfa['test_score'])
         
        
        print('Baseline Classifier with ', baseline_features, ' features')
        cv_baseline = cross_validate(classifier_baseline, testing_data.T, labels,cv=the_cv)
    
        print(cv_baseline['test_score'])
        print('Mean CV ', np.mean(cv_baseline['test_score']))
         
        Baseline_save_score[a_round,training_group] = np.mean(cv_baseline['test_score'])#classifier_baseline.score(testing_data.T,labels)
         
    print('')
    print('')    
    print(SFA_save_score)
    
    print('') 
    
    print(Baseline_save_score)
          
    
    

    return SFA_save_score, Baseline_save_score
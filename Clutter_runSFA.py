# -*- coding: utf-8 -*-
"""

Code used for binary discrimination and discrimination in clutter results.
As with all other ***_runSFA code meant to be used with a batch wrapper module
that feeds in the vocalizations to compare.

@author: ronwd
"""

import numpy as np
import soundfile as sf 
import pyfilterbank.gammatone as g
from sklearn.linear_model import SGDClassifier
from tqdm import tqdm
from SFA_Tools.SFA_Func import *
from pathlib import Path, PureWindowsPath
from sklearn.model_selection import cross_validate, ShuffleSplit

def run_SFA(cur_pair,vocal_foldername,vocal_pair,clutterfoldername, clutterfiletemplate,run_nulls):


   
    snr_values = np.array([1e-7, 1e-5, 1e-3, .1, 1.0, 10.0, 100.0, 1000.0]) 
    mode = 'ICA' #quad has to be lower case! PCA and ICA have to be all upper case! linear is default.
    #Save mode classification accuracy and the baseline model classification accuracy - called "score" here
    SFA_save_score = np.zeros((5,snr_values.size)) 
    Baseline_save_score = np.zeros((5,snr_values.size)) 
    
    
    for a_round in range(0,5): #rounds are for averaging over for each set of snr (i.e. each snr is done 10 times)
        print(a_round)
        
        for iteration in range(0,snr_values.size): #iterations for doing each snr
             print(snr_values[iteration])
             ## Files for vocalizations and noise
             load_noise = True; #toggle whether noises is generated or pulled in from a pre-generated file
             noiselen = 200000 #if loading in a pre-generated file, only take this many samples
             noise = True #toggle for whether to load and process noise
             #this works using pathlib see outer file for explanation
             voc1 = vocal_foldername / vocal_pair[0][0]
             voc2 = vocal_foldername / vocal_pair[1][0]
             #convert to windows path i.e using \ instead of /
             voc1 = PureWindowsPath(voc1)
             voc2 = PureWindowsPath(voc2)
             #set up list of vocal files to load
             vocal_files = [voc1, voc2]
             #Get the clutter file for this particular pair.  Add 1 to the cur_pair due to difference in indexing between matlab and python
             clutter_file = str(cur_pair+1)+clutterfiletemplate
             clutter_file = Path(clutter_file)
             noise_file = clutterfoldername / clutter_file
             #Same as above now switch to windows path
             noise_file = PureWindowsPath(noise_file)
             
             num_vocals = len(vocal_files) #for use later, get number of unique stimulus files loaded
        
            ## Parameters for vocalization and noise pre processing
             signal_to_noise_ratio = snr_values[iteration]#scales by average power across noise and vocalizations
             
             gfb = g.GammatoneFilterbank(samplerate= 20000, order=4, density = 1.0, startband = -21, endband = 21, normfreq = 2200) 

             down_sample = True #down sample stimulus to help with RAM issues and general dimensionality issues.  

             
             down_sample_pre = 2 #Factor by which to reduce Fs by (e.g. 10 reduces Fs by one order of magnitude) 
             down_sample_post = 2 #Factor by which to reduce Fs after applying filters 
               
                ## Parameters for training data 
            
            #new number of presentations, just 1 for one-shot learning paradigm
             num_samples = num_vocals * 1 #note current results on screen were with 15 examples, trying one last go at three vocalization then going to cut loses for tonight and stop #choose how many times you see each stimulus
             gaps = True #toggle whether there can be gaps between presentation of each stimulus
             apply_noise = False #toggle for applying noise to training set
  
             test_noise = True #Toggle for adding noise to testing set, note same noise as training for now
             #Will add another toggle for loading in new noise later.
            
             
             classifier_baseline = SGDClassifier(max_iter = 100000, tol = 0.001) #Perceptron(max_iter = 10000, tol = 0.001) #from scikit (presumably) #set up Perceptron classifier
             classifier_SFA = SGDClassifier(max_iter = 100000, tol = 0.001) #Perceptron(max_iter = 10000, tol = 0.001)
     
             classifier_features = 5 #how many features from mode the classifer gets to use
             baseline_features = 'all' #how many features the baseline model gets to use
     
             
             
             ## Load in files
        
             vocalizations = get_data(vocal_files) #get list object where each entry is a numpy array of each vocal file
             print('Vocalizations Loaded...')
            
            ##Load in and adjust noise power accordingly to sigal to noise ratio
            
             if(load_noise):
                noise, _ = sf.read(noise_file)
            
             print('Noises loaded...')
             print('Ready for preprocessing.')
            
             if noise is not None:
                noise = scale_noise(vocalizations,noise,signal_to_noise_ratio) #scales based on average power
                noise = noise[:noiselen]
             print('Noise Scaled...')
             print('Ready For Gammatone Transform')
            
            ## Apply Gammatone Transform to signal and noise
            
             vocals_transformed = gamma_transform_list(vocalizations, gfb) #does what is says on the tin: applies gamma transform to list of numpy arrays
             print('Vocalizations Transformed...')
            
             if noise is not None:
                noise_transformed = gamma_transform(noise, gfb)
                print('Noise Transformed...')
                
            ## Down sample for computation tractablility
            #reeval gammatone transform accordingly
                
             if down_sample: 
                for i,vocal in enumerate(vocals_transformed):
                    vocals_transformed[i] = vocal[:,::down_sample_pre] #down samples by factor set in above code (e.g. 10 means reduce fs by one order of magnitude)
                if noise is not None:
                    noise_transformed = noise_transformed[:,::down_sample_pre]
            
             print('Ready For Temporal Filters')
            
            ## Apply temporal filters
             tFilter = temporalFilter()
             tFilters = [tFilter]
             
             vocals_temporal_transformed = temporal_transform_list(vocals_transformed,tFilters)
             print('Vocals Temporally Filtered...')
            
             if noise is not None:
                noise_temporal_transformed = temporal_transform(noise_transformed,tFilters)
                print('Noise Temporally Filtered')
            
            #again re-evaluate if down sampled
                
             if down_sample:
                for i,vocal in enumerate(vocals_temporal_transformed):
                    vocals_temporal_transformed[i] = vocal[:,::down_sample_post] 
                if noise is not None:
                    noise_temporal_transformed = noise_temporal_transformed[:,::down_sample_post]
                    
#%% Create Training Dataset 
        
             samples = np.random.randint(num_vocals, size = num_samples)
             even_samples_check = np.sum(samples==1)
         
             while even_samples_check != np.round(num_samples/num_vocals): #while samples are not even across vocalizations
               print('Ensuring equal presentation of both vocalizations')
               samples = np.random.randint(num_vocals, size = num_samples)
               even_samples_check = np.sum(samples==1)
             print('Equal presentation of both vocalizations established')
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
    
             if(apply_noise): 
                while(noise_temporal_transformed[0].size < training_data[0].size):
                    noise_temporal_transformed = np.hstack((noise_temporal_transformed,noise_temporal_transformed))
                training_data = training_data + noise_temporal_transformed[:,0:training_data[0].size]
                print('Applied Noise...')

                print('No Noise Applied...')
            
             print('Ready For ' + mode)
             
#%%        Train Algorithm On Training Data
        
        
             (layer1, mean, variance, data_SS, weights) = getSF(training_data, 'Layer 1',mode, transform = True)
             print('SFA Training Complete')
        
        
#%%         Test Data
            
             samples = np.arange(num_vocals)
            
             testing_data = None
             initialized = False
             for i in tqdm(samples):
                if(not(initialized)):
                    testing_data = vocals_temporal_transformed[i]
                    initialized = True
                else:
                    testing_data = np.concatenate((testing_data, vocals_temporal_transformed[i]),1) 
             print('Data arranged...')
            
             if(test_noise):
                
                testing_data = testing_data + noise_temporal_transformed[:,0:testing_data[0].size]
                print('Applied Noise...')
             else:
                print('No Noise Applied...')
            
                
             print('Testing Data Ready')
            
#%% Apply SFA to Test Data, also toggles for using second layer
            
             test = testSF(testing_data, 'Layer 1',mode, mean, variance, data_SS, weights)
             print( mode +' Applied To Test Set')
            #old code related to adding a second layer.
            #test = np.vstack((test[:,5:], test[:,:-5]))
            #test = testSF(test, 'Layer 2', mean2, variance2, data_SS2, weights2)
            
           
             if run_nulls:
              labels = np.random.randint(0,2,test.shape[1]) #just put in random labels.
             else:
              labels = getlabels(vocals_temporal_transformed)
          
            ## Compare SFA With Baseline For Linear Classification
              the_cv = ShuffleSplit(n_splits = 30, test_size = 0.25)
    
             print(mode + ' Based Classifier with ', classifier_features, ' features')
            
             cv_sfa = cross_validate(classifier_SFA, test[0:classifier_features,:].T, labels,cv=the_cv)
            
             SFA_save_score[a_round,iteration] = np.mean(cv_sfa['test_score']) #take average of all CV folds as the score for that round for that SNR
            
             print('Baseline Classifier with ', baseline_features, ' features')
             cv_baseline = cross_validate(classifier_baseline, testing_data.T, labels,cv=the_cv)
             
             Baseline_save_score[a_round,iteration] = np.mean(cv_baseline['test_score']) #same thing for the baseline classifier, this should have consistent performance, but can use the variance present in this as a check on the classifier
        
        
        
                
             
    
    return SFA_save_score, Baseline_save_score
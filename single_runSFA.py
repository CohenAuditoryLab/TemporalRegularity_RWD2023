# -*- coding: utf-8 -*-
"""

Exploratory code that allows several different conditions to be run on a few example
pairs of noise.  Code in the other ***_runSFA files is based on this core code.

"""



import numpy as np
import matplotlib.pyplot as plt 
import soundfile as sf 
import pyfilterbank.gammatone as g
from sklearn import svm
from sklearn.linear_model import SGDClassifier
from tqdm import tqdm
from SFA_Tools.SFA_Func import *
from scipy import io
from pathlib import Path, PureWindowsPath
from sklearn.model_selection import cross_validate, ShuffleSplit



snr_values = np.array([1e-7, 1e-6, 1e-5, 1e-3, .1, 1.0, 10.0, 100.0, 1000.0])



mode = 'SFA' #toggle what method to use
#gets these example pairs from the vocalizations that are included in this repo
cur_pair = np.array([0, 53, 55, 57, 59, 68])
set_size = cur_pair.shape[0]

filepath = "Your\\Cool\\Filepath\\Here" 
    
vocal_foldername = PureWindowsPath(filepath + "\\GitHub\\TemporalRegularity_RWD2023\\\\Examples\\Noise Examples\\")
vocpairfile =  PureWindowsPath(filepath + "\\GitHub\\TemporalRegularity_RWD2023\\\\Examples\\Example PairsList.mat") 
clutterfoldername = PureWindowsPath(filepath + "\\GitHub\\TemporalRegularity_RWD2023\\Examples\\Clutter Examples")
unpack_pairs = io.loadmat(vocpairfile) 

#Template for loading in each clutter chorus.
clutterfiletemplate = '_ClutterChorus_20calls.wav'

all_pairs = unpack_pairs['listofpairs']
#Save classification accuracy
SFA_save_score = np.zeros((1,snr_values.size,cur_pair.size)) 
Baseline_save_score = np.zeros((1,snr_values.size,cur_pair.size)) 

for pairs in range(0,cur_pair.size):

    print('')
    print(cur_pair[pairs])
    vocal_pair = all_pairs[cur_pair[pairs]] 
    print('')
    print(vocal_pair)
    


    
    #special case toggles
    mismatch_data = False #Set to make training and test data entirely different
    #in this case, make training data noise and test data vocalizations
    
    random_features = False #Use linear null model that picks a random 5 features.
    
    shuffle_test = False #Shuffle testing data to get null performance
    
    gammatone_only_test = False
    
    non_negative = False #doesn't work


    
    for iteration in range(0,snr_values.size): #iterations for doing each snr
         print(snr_values[iteration])
         ## Files for vocalizations and noise
         load_noise = False #toggle whether noises is generated or pulled in from a pre-generated file
         noiselen = 100000 #if loading in a pre-generated file, only take this many samples
         noise = None #toggle for whether testing with noise or not, toggle true and None
          #2020-07-28 if just do noise in testing, unable to converge (waited over 45 minutes)
         apply_noise = False #toggle for applying noise to training data 
         test_noise = False #Toggle for adding unique noise in test case that is different from training case
         
      
         #this works using pathlib see outer file for explanation
         voc1 = vocal_foldername / vocal_pair[0][0]
         voc2 = vocal_foldername / vocal_pair[1][0]
         #convert to windows path i.e using \ instead of /
         voc1 = PureWindowsPath(voc1)
         voc2 = PureWindowsPath(voc2)
         
         if toy_examples: #if toy examples switch vocals to two different two exampels taken from original 2002 paper and Bellec et al 2016 paper
             #first try, just put in the same vocal twice to check what the weights do
             voc1 = PureWindowsPath("C:\\Users\\ronwd\\OneDrive\\Documents\\GitHub\\SFA_PostCOSYNEAPP\\dummyexample1c.wav")
             voc2 = PureWindowsPath("C:\\Users\\ronwd\\OneDrive\\Documents\\GitHub\\SFA_PostCOSYNEAPP\\dummyexample1c.wav")
            
         #set up list of vocal files to load
         vocal_files = [voc1, voc2]
         
         
         #Get the clutter file for this particular pair.  Add 1 to the cur_pair due to difference in indexing between matlab and python
         clutter_file = str(cur_pair[pairs]+1)+clutterfiletemplate #'white_noise.wav' #
         clutter_file = Path(clutter_file)
        
         noise_file = clutterfoldername / clutter_file
   
         
    
         
         num_vocals = len(vocal_files) #for use later, get number of unique stimulus files loaded'
            ##Training and testing data parameters  
         num_samples = num_vocals * 1
         gaps = True #toggle whether there can be gaps between presentation of each stimulus
               
       
    
        ## Parameters for vocalization and noise pre processing by cochlear model
         signal_to_noise_ratio = snr_values[iteration]
         
         gfb = g.GammatoneFilterbank(samplerate= 50000, order=4, density = 1.0, startband = -21, endband = 21, normfreq = 2200) 
                              
         down_sample = True #down sample stimulus to help with RAM issues and general dimensionality issues.  I believe mostly reduces resolution of frequency
         down_sample_pre = 2 #Factor by which to reduce Fs by (e.g. 10 reduces Fs by one order of magnitude) 
         down_sample_post = 2 #Factor by which to reduce Fs after applying filters 
            
            ## Parameters for training data
        
       
         if test_noise: #try loading in novel noise, in this case white noise
              clutter_file = 'white_noise1.wav' #second white noise file
              clutter_file = Path(clutter_file)
              noise_file1 = clutterfoldername / clutter_file
             
         

         classifier_baseline = SGDClassifier(max_iter = 100000, tol = 0.001)  #LinearSVC(max_iter = 10000000, tol = 0.001) #
         classifier_SFA = SGDClassifier(max_iter = 100000, tol = 0.001) #LinearSVC(max_iter = 10000000, tol = 0.001) #
         
         
         
       
         classifier_features = 5 #how many features from mode classifer gets to use
         if random_features: #if random use same number as SFA
             baseline_features = classifier_features
             
         else:
             baseline_features = 'all' #otherwise use all features for baseline model
         
         ##plotting toggles
         plot_vocals = False #plot individual vocals after gamatone and temporal transformed
         plot_noise = False
         plot_training = False #plot training stream
         plot_test = False #plotting toggle for  test stream
         plot_scores = False
         plot_features = False #plotting toggle for filters found by SFA
         plot_splits = False #plots split of data for the last iteration
         plot_temporal_filters = False
         ## Load in files
    
         vocalizations = get_data(vocal_files) #get list object where each entry is a numpy array of each vocal file
         print('Vocalizations Loaded...')
        
        ##Load in and adjust noise power accordingly to signal to noise ratio
        
         if(load_noise):
            noise, _ = sf.read(noise_file)
            if test_noise: #could make noise code work with lists like other code, but just doing this quick change for now.
                noise2, _ = sf.read(noise_file1)
          
         else: #use white noise
             noise = np.random.normal(0.0 , 1.0, noiselen)
             if test_noise: #could make noise code work with lists like other code, but just doing this quick change for now.
                noise2, _ = sf.read(noise_file1)
        
         print('Noises loaded...')
         print('Ready for preprocessing.')
        
         if noise is not None:
            noise = scale_noise(vocalizations,noise,signal_to_noise_ratio) #scales based on average power
            noise = noise[:noiselen]
            if test_noise: #could make noise code work with lists like other code, but just doing this quick change for now.
                noise2 = scale_noise(vocalizations,noise2, signal_to_noise_ratio)
         print('Noise Scaled...')
         print('Ready For Gammatone Transform')
        
        ## Apply Gammatone Transform to signal and noise
        
         vocals_transformed = gamma_transform_list(vocalizations, gfb) #does what is says on the tin: applies gamma transform to list of numpy arrays
         print('Vocalizations Transformed...')
        
         if noise is not None:
            noise_transformed = gamma_transform(noise, gfb)
            if test_noise: #could make noise code work with lists like other code, but just doing this quick change for now.
                noise_transformed2 = gamma_transform(noise2,gfb)
            print('Noise Transformed...')
         if plot_vocals:
             for i in range(0,num_vocals):
                 plt.figure()
                 plt.imshow(vocals_transformed[i],aspect = 'auto', origin = 'lower')
                 plt.title('Gammatone transformed')
        ## Down sample for computation tractablility
        
            
         if down_sample: 
            for i,vocal in enumerate(vocals_transformed):
                vocals_transformed[i] = vocal[:,::down_sample_pre] #down samples by factor set in above code (e.g. 10 means reduce fs by one order of magnitude)
            if noise is not None:
                noise_transformed = noise_transformed[:,::down_sample_pre]
                if test_noise: #could make noise code work with lists like other code, but just doing this quick change for now.
                   noise_transformed2 = noise_transformed2[:,::down_sample_pre]
         print('Ready For Temporal Filters')
        
        ## Apply temporal filters
        
         tFilter = temporalFilter()
         tFilters = [tFilter] 
         
         
         
         if plot_temporal_filters:
             for index in range(0,2): #just hard coding for now
                 plt.figure()
                 plt.plot(tFilters[index])
             
         
         vocals_temporal_transformed = temporal_transform_list(vocals_transformed,tFilters)
         print('Vocals Temporally Filtered...')
         
         if noise is not None:
            noise_temporal_transformed = temporal_transform(noise_transformed,tFilters)
            print('Noise Temporally Filtered')
            if test_noise: #could make noise code work with lists like other code, but just doing this quick change for now.
                 noise_temporal_transformed2 = temporal_transform(noise_transformed2,tFilters)
                 print('New Noise Temporally Filtered')
        #again re-evaluate if down sampled
            
         if down_sample:
            for i,vocal in enumerate(vocals_temporal_transformed):
                vocals_temporal_transformed[i] = vocal[:,::down_sample_post] #I guess this does a separate down sample after the temporal filters have been applied?
            if noise is not None:
                noise_temporal_transformed = noise_temporal_transformed[:,::down_sample_post]
                if test_noise: #could make noise code work with lists like other code, but just doing this quick change for now.
                    noise_temporal_transformed2 = noise_temporal_transformed2[:,::down_sample_post]
         if plot_vocals:
             for i in range(0,num_vocals):
                 plt.figure()
                 plt.imshow(vocals_temporal_transformed[i],aspect = 'auto', origin = 'lower')
                 plt.title('temporal transformed')
                 
                 
            
            
         
    ## Create Training Dataset
         if mismatch_data: #If mismatch, replace vocals with noise file.
             
             training_data = noise_temporal_transformed
             
             if(gaps): # can keep the gaps the same
               min_gap = np.round(.05 * vocals_temporal_transformed[0].shape[1]) #sets min range of gap as percentage of length of a single vocalizations
               max_gap = np.round(.5 * vocals_temporal_transformed[0].shape[1]) #set max range of gap in same units as above
               training_data = np.concatenate((training_data, np.zeros((training_data.shape[0], np.random.randint(min_gap,max_gap)))),1)     
             print('Data arranged...')
             #Put in ability to apply noise, but will just shut off in noise case since it doesn't make a whole lot of sense.
             if(apply_noise): 
                while(noise_temporal_transformed[0].size < training_data[0].size):
                    noise_temporal_transformed = np.hstack((noise_temporal_transformed,noise_temporal_transformed))
                if plot_noise:
                    plt.figure()
                    plt.title('Noise')
                    plt.imshow(noise_temporal_transformed[:,0:training_data[0].size], aspect = 'auto', origin = 'lower')
                    
                training_data = training_data + noise_temporal_transformed[:,0:training_data[0].size]
                print('Applied Noise...')
               
             else:
                print('No Noise Applied...')
             
         else: #If not mistmatch, run code normally
                 
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
           
             if plot_training:
               plt.figure()
               plt.title('training_data')
               plt.imshow(training_data, aspect = 'auto', origin = 'lower')
        
             if(apply_noise): 
                while(noise_temporal_transformed[0].size < training_data[0].size):
                    noise_temporal_transformed = np.hstack((noise_temporal_transformed,noise_temporal_transformed))
                if plot_noise:
                    plt.figure()
                    plt.title('Noise')
                    plt.imshow(noise_temporal_transformed[:,0:training_data[0].size], aspect = 'auto', origin = 'lower')
                    
                training_data = training_data + noise_temporal_transformed[:,0:training_data[0].size]
                print('Applied Training Noise...')
               
             else:
                print('No Training Noise Applied...')
        
        
       #Another check; only have gammatone filter be applied and no noise
         if gammatone_only_test:
           training_data = np.hstack(vocals_transformed)
        
         print('Ready For ' + mode)
         if plot_training:
                plt.figure()
                this_title = 'Training Stream with Noise SNR: ' +  str(signal_to_noise_ratio)
                plt.title(this_title)
                plt.imshow(training_data, aspect = 'auto', origin = 'lower')
          # Train SFA On Data
    
         if non_negative: #not non_negative doesn't work right now
              (layer1, mean, variance, data_SS, weights) = getSFNonNeg(training_data, 'Layer 1',mode, transform = True)
         else:
                 
              (layer1, mean, variance, data_SS, weights) = getSF(training_data, 'Layer 1', mode,transform = True)
         print(mode + ' Training Complete')
    
    
        ## Test Results
        
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
         
        
         if gammatone_only_test:
             testing_data = np.hstack(vocals_transformed)
        
         if(test_noise):

            
            testing_data = testing_data + noise_temporal_transformed2[:,0:testing_data[0].size]
            print('Applied Noise...')
         else:
            print('No Testing Noise Applied...')
            
  
        
        
         if shuffle_test:
         #quick sanity check, shuffle test
             testing_data = testing_data[:,np.random.randint(0, np.shape(testing_data)[1],np.shape(testing_data)[1])]
             
        
         
         print('Testing Data Ready')
         if plot_test:
             plt.figure()
             this_title = 'Testing Stream with Noise SNR: ' +  str(signal_to_noise_ratio)
             plt.title(this_title)
             plt.imshow(testing_data, aspect = 'auto', origin = 'lower')
        
        ## Apply mode to Test Data
        
         test = testSF(testing_data, 'Layer 1',mode, mean, variance, data_SS, weights)
         print( mode + ' Applied To Test Set')
       
        
        
         if gammatone_only_test:
             labels = getlabels(vocals_transformed)#
         else:
             labels = getlabels(vocals_temporal_transformed)
  
        ## Compare SFA With Baseline For Linear Classification
        
         cv_toggle = 1
         
      
         
         
        
         if cv_toggle > 0:
             
             reverse_features = 0
             
             random_SFA_features = 0
             
             the_cv = ShuffleSplit(n_splits = 30, test_size = 0.25) 
            
             print(mode + ' Based Classifier with ', classifier_features, ' features')
             #add a cv loop to pin down variance
             
             if reverse_features>0:
                 stop_point = 20 - classifier_features-1
                 cv_sfa = cross_validate(classifier_SFA, test[-1:stop_point:-1,:].T, labels,cv=the_cv)
                 
             elif random_SFA_features>0:
                 
                 feature_set = np.random.randint(1, 20,classifier_features)
                 cv_sfa = cross_validate(classifier_SFA, test[feature_set,:].T, labels,cv=the_cv)
             else:
                 
                 cv_sfa = cross_validate(classifier_SFA, test[0:classifier_features,:].T, labels,cv=the_cv)
                 
        
             
             print(cv_sfa['test_score'])
             print('Mean CV ', np.mean(cv_sfa['test_score']))
             
             SFA_save_score[0,iteration,pairs] = np.mean(cv_sfa['test_score'])
             
            
             print('Baseline Classifier with ', baseline_features, ' features')
             
             if baseline_features != 'all': #Only trigger if using a subset of features
                 
                 testing  = testing_data[np.random.randint(1, np.shape(testing_data)[0],5),:]
                 plt.figure()
                 plt.title('Random 5 channels of cochlear model for baseline')
                 plt.imshow(testing,aspect = 'auto', origin = 'lower')
             else:
                     
                     testing = testing_data
             
             cv_baseline = cross_validate(classifier_baseline, testing.T, labels,cv=the_cv)

             print(cv_baseline['test_score'])
             print('Mean CV ', np.mean(cv_baseline['test_score']))
             
             Baseline_save_score[0,iteration,pairs] = np.mean(cv_baseline['test_score'])#classifier_baseline.score(testing_data.T,labels)
             
         else:
              
              SFA_classifier = svm.SVC(kernel = 'linear')
              SFA_classifier.fit(test[0:classifier_features,:].T, labels)
              SFA_save_score[0,iteration,pairs] = SFA_classifier.score(test[0:classifier_features,:].T, labels)
              
              if baseline_features != 'all': #Only trigger if using a subset of features
                 
                 testing  = testing_data[np.random.randint(1, np.shape(testing_data)[0],5),:]
                 plt.figure()
                 plt.imshow(testing,aspect = 'auto', origin = 'lower')
              else:
                     
                     testing = testing_data
              
              Baseline_classifier = svm.SVC(kernel = 'linear')
              Baseline_classifier.fit(testing.T, labels)
              Baseline_save_score[0,iteration,pairs] = Baseline_classifier.score(testing.T, labels)
    print('')
    print(cur_pair)
    print('')    
    print(SFA_save_score)
    print('') 
    
    print(Baseline_save_score)
    print('') 
    print(mode + ' mean: ' + str(SFA_save_score.mean()))
    print('Baseline mean: ' + str(Baseline_save_score.mean()))
    

#%% PLOTTING Results

if plot_scores:
    
    if quick_set:
        
        print(np.mean(SFA_save_score, axis = 2))
            
        leg_labels = []
        for ex in range(0,set_size):
            
            leg_labels.append(all_pairs[cur_pair][ex][0][0] +' w/ '+ all_pairs[cur_pair][ex][1][0])
        
        plt.figure()
        plt.plot(np.log10(snr_values), SFA_save_score[0,:,:]*100.0)
        plt.plot(np.log10(snr_values)[-1], np.mean(Baseline_save_score[0,:,:]*100.0),'*') #np.log10(snr_values)[-1]
        plt.ylabel('CV Average Percent Classified Correct')
        plt.xlabel('log10 SNR')
        plt.title('5 vocals for ' + mode +' case'  )
        plt.legend(leg_labels) #This probably won't conveniently work
        
    else:
            
            
    
        plt.figure()
        plt.plot(np.log10(snr_values), SFA_save_score[0,:]*100.0)
        plt.plot(np.log10(snr_values)[-1], np.mean(Baseline_save_score[0,:]*100.0),'r.') #np.log10(snr_values)[-1]
        plt.ylabel('CV Average Percent Classified Correct')
        plt.xlabel('log10 SNR')

if plot_splits: #note this spits out a lot of figures
    for train_index, test_index in the_cv.split(test.T):
    
    
        plt.figure()
        plt.hist(labels[train_index])
        plt.title('training data distribution for each split')
    
    
    for train_index, test_index in the_cv.split(test.T):
    
    
        plt.figure()
        plt.hist(labels[test_index])
        plt.title('training data distribution for each split')


if plot_features:
    

    
    for i in range(classifier_features): #quickly plot the used features over time on the same figure.  Simple move the figure generation to get the separately
        plt.figure()
        plt.plot(np.arange(0,test.shape[1]),test[i,:])


        
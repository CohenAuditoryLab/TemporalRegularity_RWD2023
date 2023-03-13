# -*- coding: utf-8 -*-
"""
Copy for generating basic stats used in paper

@author: ronwd
"""

import numpy as np
from pathlib import Path, PureWindowsPath
from scipy import stats, io


#%% Feature sweep data
#Set file

filepath = "Your cool file path goes here"
results_file = PureWindowsPath( filepath + "\\GitHub\\TemporalRegularity_RWD2023\\Results\\FeatureSweep.mat")
feature_range = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,20])

unpack_data = io.loadmat(results_file)
SFA_scores_batch = unpack_data['SFA_save_score']



#%% Basic Stats



method = 'ICA' #PCA, ICA, Linear, Quad
noisemode = 'testclutter' #trainclutter, testclutter #double clutter is done later
stimtype = 'vocals' #vocals, noise, App ( App only for Quad).

results_file = PureWindowsPath(filepath + "\\GitHub\\TemporalRegularity_RWD2023\\Results\\Clutter Results\\SNR_" + method + "_" + noisemode  +"_newclass_"+ stimtype + ".mat") 


snr_values = np.array([1e-7, 1e-5, 1e-3, .1, 1.0, 10.0, 100.0, 1000.0]) 

unpack_data = io.loadmat(results_file)
unpack_pairs = unpack_data['stimuli_ind_list']
all_pairs = unpack_data['all_pairs']

Baseline_scores_batch = unpack_data['Baseline_scores_batch']
SFA_scores_batch = unpack_data['SFA_scores_batch']
#only for altbaseline
baseline_file = PureWindowsPath("C:\\Users\\ronwd\\OneDrive\\Documents\\GitHub\\SpectralManifoldWork\\SNR_New\\SNR_altbaseline_"+ noisemode + "_newclass_" + stimtype +".mat")
unpack_data = io.loadmat(baseline_file)
LB_Baseline_scores_batch = unpack_data['Baseline_scores_batch']



#grab avg and std for 5 iterations of each of the vocalizations, then repeat over all vocalizations
avg_per_vocal_SFA = np.mean(SFA_scores_batch, axis = 1)
std_per_vocal_SFA = np.std(SFA_scores_batch, axis = 1)



avg_per_vocal_Baseline = np.mean(Baseline_scores_batch, axis = 1)
std_per_vocal_Baseline = np.std(Baseline_scores_batch, axis = 1)

avg_per_vocal_LB_Baseline = np.mean(LB_Baseline_scores_batch, axis = 1)
std_per_vocal_LB_Baseline = np.std(LB_Baseline_scores_batch, axis = 1)


#Get overall stats

ov_avg_SFA = np.mean(avg_per_vocal_SFA,axis = 0)
ov_median_SFA = np.median(avg_per_vocal_SFA,axis = 0)
ov_std_SFA = np.std(avg_per_vocal_SFA,axis = 0)
ov_Q1_SFA = np.quantile(avg_per_vocal_SFA, .25, axis = 0)
ov_Q3_SFA = np.quantile(avg_per_vocal_SFA, .75, axis = 0)



ov_avg_Baseline = np.mean(avg_per_vocal_Baseline,axis = 0)
ov_std_Baseline = np.std(avg_per_vocal_Baseline,axis = 0)
ov_median_Baseline = np.median(avg_per_vocal_Baseline,axis = 0)
ov_Q1_Baseline = np.quantile(avg_per_vocal_Baseline, .25, axis = 0)
ov_Q3_Baseline = np.quantile(avg_per_vocal_Baseline, .75, axis = 0)
ov_avg_LB_Baseline = np.mean(avg_per_vocal_LB_Baseline,axis = 0)
ov_std_LB_Baseline = np.std(avg_per_vocal_LB_Baseline,axis = 0)
ov_median_LB_Baseline = np.median(avg_per_vocal_LB_Baseline,axis = 0)
ov_Q1_LB_Baseline = np.quantile(avg_per_vocal_LB_Baseline, .25, axis = 0)
ov_Q3_LB_Baseline = np.quantile(avg_per_vocal_LB_Baseline, .75, axis = 0)


#First calculate the index of the entry and then put in that value for the CIs

z_star = 1.96 #Doing 95% confidence intervals.  Calculate for all for later graph.
#Note: I think these are all the same size so the indecies should be the same, but better safe than sorry

ci_sfa_ind_low = int(np.round((np.shape(avg_per_vocal_SFA)[0]+1)/2 - z_star*np.sqrt(np.shape(avg_per_vocal_SFA)[0])/2))
ci_sfa_ind_high = int(np.round((np.shape(avg_per_vocal_SFA)[0]+1)/2 + z_star*np.sqrt(np.shape(avg_per_vocal_SFA)[0])/2))

ci_UB_ind_low = int(np.round((np.shape(avg_per_vocal_Baseline)[0]+1)/2 - z_star*np.sqrt(np.shape(avg_per_vocal_Baseline)[0])/2))
ci_UB_ind_high = int(np.round((np.shape(avg_per_vocal_Baseline)[0]+1)/2 + z_star*np.sqrt(np.shape(avg_per_vocal_Baseline)[0])/2))

ci_LB_ind_low = int(np.round((np.shape(avg_per_vocal_LB_Baseline)[0]+1)/2 - z_star*np.sqrt(np.shape(avg_per_vocal_LB_Baseline)[0])/2))
ci_LB_ind_high = int(np.round((np.shape(avg_per_vocal_LB_Baseline)[0]+1)/2 + z_star*np.sqrt(np.shape(avg_per_vocal_LB_Baseline)[0])/2))


#I think these only work for sorted vectors

sort_avg_sfa = np.sort(avg_per_vocal_SFA,0)
sort_avg_UB = np.sort(avg_per_vocal_Baseline,0)
sort_avg_LB = np.sort(avg_per_vocal_LB_Baseline,0)

CI_SFA_high = np.array([sort_avg_sfa[ci_sfa_ind_low,-1], sort_avg_sfa[ci_sfa_ind_high,-1]])
CI_SFA_low = np.array([sort_avg_sfa[ci_sfa_ind_low,0], sort_avg_sfa[ci_sfa_ind_high,0]])
CI_SFA_midlow = np.array([sort_avg_sfa[ci_sfa_ind_low,2], sort_avg_sfa[ci_sfa_ind_high,2]])
CI_SFA_midhigh = np.array([sort_avg_sfa[ci_sfa_ind_low,4], sort_avg_sfa[ci_sfa_ind_high,4]])
CI_SFA_2ndhigh = np.array([sort_avg_sfa[ci_sfa_ind_low,-2], sort_avg_sfa[ci_sfa_ind_high,-2]])
#np.array([z_star *np.sqrt(np.shape(avg_per_vocal_SFA)[0])])

CI_Baseline = np.array([sort_avg_UB [ci_UB_ind_low,-1], sort_avg_UB [ci_UB_ind_high,-1]])
#np.array([z_star *ov_std_Baseline/np.sqrt(np.shape(avg_per_vocal_Baseline)[0])])

CI_LB_Baseline = np.array([sort_avg_LB[ci_LB_ind_low,-1], sort_avg_LB[ci_LB_ind_high,-1]])
#np.array([z_star*ov_std_LB_Baseline/np.sqrt(np.shape(avg_per_vocal_LB_Baseline)[0])])
#High is highest, low is lowest, midlow is -2 log10sne, midhigh is 1 log10snr, 2ndhigh is second highest snr, which might not be used but we have it
print('Confidence intervals for medians in order specified in code comments')
print('Highest: ' , CI_SFA_high, '\n') #highest SNR
print('Lowest: ' ,CI_SFA_low, '\n') #lowest SNR
print('-2: ' ,CI_SFA_midlow, '\n') #lowest SNR
print('1: ',CI_SFA_midhigh, '\n') #lowest SNR
print('100: ',CI_SFA_2ndhigh, '\n') #2nd highest SNR
print('full linear: ', CI_Baseline, '\n')
print('random 5: ',CI_LB_Baseline, '\n')




avg_overall_SFA = np.nanmean(avg_per_vocal_SFA)*100
sem_overall_SFA = np.nanstd(avg_per_vocal_SFA)*100 / np.sqrt(avg_per_vocal_SFA.size - np.sum(np.isnan(avg_per_vocal_SFA)))


print('')
print('Avg performance and SEM across all training SNRs for ' + method + ':')
print(str(avg_overall_SFA) + ' +/- ' + str(sem_overall_SFA))






#%% Medians



stats_test = 'Mann_Whitney U, one-tailed, X > Y'
stats_test2 = 'Mann_Whitney U, two-tailed, X = Y'
print('')
print( method + ' SNR = '  + str(snr_values[-1])  +' vs Chance test statistic and P value using ' + stats_test + ':')
SNR_Highest_Chanceresults = stats.mannwhitneyu(avg_per_vocal_SFA[:,-1], .5*np.ones(avg_per_vocal_SFA[:,-1].shape), alternative = 'greater')
print(SNR_Highest_Chanceresults.statistic)
print(SNR_Highest_Chanceresults.pvalue)

print('') #I'm sure there is code for setting a new line, but this works for now
print(method + ' SNR = '   + str(snr_values[-1])  +' vs Lower Bound test statistic and P value using ' + stats_test + ':')
SNR_Highest_LBresults = stats.mannwhitneyu(avg_per_vocal_SFA[~np.isnan(avg_per_vocal_LB_Baseline[:,-1]),-1], avg_per_vocal_LB_Baseline[~np.isnan(avg_per_vocal_LB_Baseline[:,-1]),-1], alternative = 'greater')
print(SNR_Highest_LBresults.statistic)
print(SNR_Highest_LBresults.pvalue)
print('')
print(method + ' SNR = '   + str(snr_values[-1])  +' vs Upper Bound test statistic and P value using'  + stats_test2 + ':')
SNR_Highest_UBresults = stats.mannwhitneyu(avg_per_vocal_SFA[:,-1],avg_per_vocal_Baseline[:,-1], alternative = 'two-sided')
print(SNR_Highest_UBresults.statistic)
print(SNR_Highest_UBresults.pvalue)
#%% Variance
stats_test = 'Levene comparison of variance, X = Y'

print('') #I'm sure there is code for setting a new line, but this works for now
print(method + ' SNR = '   + str(snr_values[-1])  +' vs Linear models test statistic and P value using ' + stats_test + ':')
SNR_Highest_Vartogether = stats.levene(avg_per_vocal_SFA[~np.isnan(avg_per_vocal_LB_Baseline[:,-1]),-1],avg_per_vocal_Baseline[~np.isnan(avg_per_vocal_LB_Baseline[:,-1]),-1],avg_per_vocal_LB_Baseline[~np.isnan(avg_per_vocal_LB_Baseline[:,-1]),-1])
print(SNR_Highest_Vartogether.statistic)
print(SNR_Highest_Vartogether.pvalue)
print('') #I'm sure there is code for setting a new line, but this works for now
print(method + ' SNR = '   + str(snr_values[-1])  +' vs just null model test statistic and P value using ' + stats_test + ':')
SNR_Highest_VarLB = stats.levene(avg_per_vocal_SFA[~np.isnan(avg_per_vocal_LB_Baseline[:,-1]),-1],avg_per_vocal_LB_Baseline[~np.isnan(avg_per_vocal_LB_Baseline[:,-1]),-1])
print(SNR_Highest_VarLB.statistic)
print(SNR_Highest_VarLB.pvalue)
print('') #I'm sure there is code for setting a new line, but this works for now
print(method + ' SNR = '   + str(snr_values[-1])  +' vs just upper bound test statistic and P value using ' + stats_test + ':')
SNR_Highest_VarUB = stats.levene(avg_per_vocal_SFA[:,-1],avg_per_vocal_Baseline[:,-1])
print(SNR_Highest_VarUB.statistic)
print(SNR_Highest_VarUB.pvalue)



#%% Checks
stats_test = 'Mann_Whitney U, one-tailed, X > Y'
#check above chance at lowest snr
print('') #I'm sure there is code for setting a new line, but this works for now
print(method + ' SNR = '   + str(snr_values[0])  +' vs Chance test statistic and P value using ' + stats_test + ':')
SNR_Lowest_Chanceresults = stats.mannwhitneyu(avg_per_vocal_SFA[:,0],.5*np.ones(np.shape(avg_per_vocal_SFA[:,0])),alternative = 'greater')
print(SNR_Lowest_Chanceresults.statistic)
print(SNR_Lowest_Chanceresults.pvalue)


#check limit on when first above lower bound
print('') #I'm sure there is code for setting a new line, but this works for now
print(method + ' SNR = '   + str(snr_values[2])  +' vs Lower Bound test statistic and P value using ' + stats_test + ':')
SNR_1_LBresults = stats.mannwhitneyu(avg_per_vocal_SFA[:,2],avg_per_vocal_LB_Baseline[:,-1],alternative = 'greater')
print(SNR_1_LBresults.statistic)
print(SNR_1_LBresults.pvalue)

#%% Spearman

SNR_all_Spearman = stats.spearmanr(snr_values, ov_avg_SFA)
print('Spearman Correlation between overall average performance and SNR')
print(SNR_all_Spearman.correlation)
print(SNR_all_Spearman.pvalue)

#doing this accross vocals as well since the ov_avg has so few data points and is so significant
#this is the value reported in the paper
snr_repeated = np.tile(snr_values,(100,1))
snr_repeated = snr_repeated.reshape(800,1)
SNR_acrossall_Spearman = stats.spearmanr(snr_repeated,avg_per_vocal_SFA.reshape(800,1))
print('Spearman Correlation between per vocal average performance and SNR')
print(SNR_acrossall_Spearman.correlation)
print(SNR_acrossall_Spearman.pvalue)

#Simply check to make sure nothing is off (i.e. this should come out as not significant)
print('') #I'm sure there is code for setting a new line, but this works for now
print('Check that test works correctly')
print(method + ' SNR = '   + str(snr_values[-2])  +' vs Upper Bound test statistic and P value using ' + stats_test + ':')
SNR_100_UBresults = stats.mannwhitneyu(avg_per_vocal_SFA[:,-2],avg_per_vocal_Baseline[:,-1], alternative = 'greater')
print(SNR_100_UBresults.statistic)
print(SNR_100_UBresults.pvalue)



#%% Cross method stuff
#assumes ran above method at least once

method = ['PCA','ICA','Linear','Quad']

avg_per_vocal_SFA_hold = np.zeros((len(method),avg_per_vocal_SFA.shape[0],avg_per_vocal_SFA.shape[1]))

for i in range(len(method)):

    results_file = PureWindowsPath(filepath + "\\GitHub\\TemporalRegularity_RWD2023\\Results\\Clutter Results\\SNR_" + method + "_" + noisemode  +"_newclass_"+ stimtype + ".mat") 
    
    #New range 2020-08-04
    snr_values = np.array([1e-7, 1e-5, 1e-3, .1, 1.0, 10.0, 100.0, 1000.0]) 
    
    unpack_data = io.loadmat(results_file)
    unpack_pairs = unpack_data['stimuli_ind_list']
    all_pairs = unpack_data['all_pairs']
    
    Baseline_scores_batch = unpack_data['Baseline_scores_batch']
    SFA_scores_batch = unpack_data['SFA_scores_batch']
    
    baseline_file = PureWindowsPath("C:\\Users\\ronwd\\OneDrive\\Documents\\GitHub\\SpectralManifoldWork\\SNR_New\\SNR_altbaseline_"+ noisemode + "_newclass_" + stimtype +".mat")
    unpack_data = io.loadmat(baseline_file)
    LB_Baseline_scores_batch = unpack_data['Baseline_scores_batch']
    

    
    #grab avg and std for 5 iterations of each of the vocalizations, then repeat over all vocalizations
    avg_per_vocal_SFA_hold[i,:,:] = np.mean(SFA_scores_batch, axis = 1)
    
    

stats_test2 = 'Mann_Whitney U, two-tailed, X = Y'
print('')
print( method[1] + ' ' + method[0] + ' SNR = '  + str(snr_values[-1])  +' vs Chance test statistic and P value using ' + stats_test2 + ':')
SNR_PCAvsLinear = stats.mannwhitneyu(avg_per_vocal_SFA_hold[0,:,-1], avg_per_vocal_SFA_hold[1,:,-1])
print(SNR_PCAvsLinear.statistic)
print(SNR_PCAvsLinear.pvalue)   

stats_test2 = 'Mann_Whitney U, two-tailed, X = Y'
print('')
print( method[0] + ' ' + method[2] + ' SNR = '  + str(snr_values[-1])  +' vs Chance test statistic and P value using ' + stats_test2 + ':')
SNR_PCAvsQuad = stats.mannwhitneyu(avg_per_vocal_SFA_hold[0,:,-1], avg_per_vocal_SFA_hold[2,:,-1])
print(SNR_PCAvsQuad.statistic)
print(SNR_PCAvsQuad.pvalue)

stats_test2 = 'Mann_Whitney U, two-tailed, X = Y'
print('')
print( method[1] + ' ' + method[2] + ' SNR = '  + str(snr_values[-1])  +' vs Chance test statistic and P value using ' + stats_test2 + ':')
SNR_LinearvsQuad = stats.mannwhitneyu(avg_per_vocal_SFA_hold[1,:,-1], avg_per_vocal_SFA_hold[2,:,-1])
print(SNR_LinearvsQuad.statistic)
print(SNR_LinearvsQuad.pvalue)




#%% STATS rewrite for double clutter

mode = "ICA"

#Need to write an additional coding chunk for the dc double clutter case.

results_file = PureWindowsPath(filepath + "\\GitHub\\TemporalRegularity_RWD2023\\Results\\Clutter Results\\SNR_" + method + "_dc_full_newclass_vocals.mat") #newsplit_try_Clutter_Aug.mat #newsplit_try2_Clutter_Aug.mat


unpack_data = io.loadmat(results_file)
unpack_pairs = unpack_data['stimuli_ind_list']
all_pairs = unpack_data['all_pairs']

Baseline_scores_batch = unpack_data['Baseline_scores_batch']
SFA_scores_batch = unpack_data['SFA_scores_batch']

baseline_file = PureWindowsPath("C:\\Users\\ronwd\\OneDrive\\Documents\\GitHub\\SpectralManifoldWork\\SNR_New\\SNR_altbaseline_testclutter_newclass_vocals.mat")
unpack_data = io.loadmat(baseline_file)
LB_Baseline_scores_batch = unpack_data['Baseline_scores_batch']

#Set up SNR variable
snr_values = np.log10(np.array([1e-7, 1e-5, 1e-3, .1, 1.0, 10.0, 100.0, 1000.0])) #New range 2020-08-04 #1e-5, 1e-3, .1, 1.0, 10.0, 100.0,
snr_values = snr_values.astype('int')
snr_num = snr_values.shape[0]



#Need to unroll score matrices so we have one column of the data frame that is the correct size
#Then we can use this to check that we didn't mess up the sizing for the other columns
#For test case just used 1 then 2 "rows" of 4-d matrix to make sure the ordering is correct

#For the sake of simplicity.  Going to average across the 5 repeats of each analysis ahead of making the data frame
#So it will now be num_pairs x train_snr x test_snr


SFA_scores_batch = np.mean(SFA_scores_batch, axis = 1)*100
Baseline_scores_batch = np.mean(Baseline_scores_batch, axis =1)*100
LB_Baseline_scores_batch = np.mean(LB_Baseline_scores_batch , axis =1)*100
#Note: since Baseline is not run in the double clutter code proper we have first take the mean and then repeat 8 times along the 2nd snr axis so it is the same size as the other variables
LB_Baseline_scores_batch = np.repeat(np.reshape(LB_Baseline_scores_batch,(unpack_pairs.size,snr_num,1)),snr_num, axis = 2)


#In plots we grouped by test snr into two groups.  Low test SNR and high test SNR
#Let's keep that for the stats.

SFA_scores_lowtest = SFA_scores_batch[:,:,:4]
SFA_scores_hightest = SFA_scores_batch[:,:,4:]

Baseline_scores_lowtest = Baseline_scores_batch[:,:,:4]
Baseline_scores_hightest = Baseline_scores_batch[:,:,4:]

LB_Baseline_scores_lowtest = LB_Baseline_scores_batch[:,:,:4]
LB_Baseline_scores_hightest = LB_Baseline_scores_batch[:,:,4:]


#Grab average across all training SNRs and then average across grouped testing SNRs
#essentially get a single average for each graph with its std

avg_low_overall_SFA = np.nanmean(SFA_scores_lowtest)
sem_low_overall_SFA = np.nanstd(SFA_scores_lowtest)/np.sqrt(SFA_scores_lowtest.size - np.sum(np.isnan(SFA_scores_lowtest)))
median_low_overall_SFA = np.nanmedian(SFA_scores_lowtest)

avg_high_overall_SFA = np.nanmean(SFA_scores_hightest)
sem_high_overall_SFA = np.nanstd(SFA_scores_hightest)/np.sqrt(SFA_scores_hightest.size - np.sum(np.isnan(SFA_scores_hightest)))

print('')
print('Avg performance and SEM across all training SNRs for low testing SNRs for ' + method + ':')
print(str(avg_low_overall_SFA) + ' +/- ' + str(sem_low_overall_SFA))


print('')
print('Avg performance and SEM across all training SNRs for high testing SNRs for ' + method + ':')
print(str(avg_high_overall_SFA) + ' +/- ' + str(sem_high_overall_SFA))

#Unclear if any additional stats need to be done here, can set up a loop to get median
#and do mann-whitney but not sure if that is necessary.  








#%% Generalization

chance_lvl = list([50.0,33.3,25.0,20.0]) #list of chance levels for changing axis limit
#update 2022-10-31: not needed right now since only use 2 voxs but leaving in just because

all_trials = list(['trial1', 'trial2', 'trial3', 'trial4'])

all_prefixes = {"trial1" : 'co',
                "trial2" : 'ha',
                "trial3" : 'gt',
                "trial4" : 'sb',
               }
 

all_training_cat = {"trial1" : list(['all coos']),
                "trial2" : list(['all arches']),
                "trial3" : list(['all grunts']),
                "trial4" : list(['all barks']),
                }

num_vocs = 2



mode = "ICA"
#testing above chance
for index in  range(0,np.size(all_trials)): #temp_list: #
        cur_trial = all_trials[index]
        results_file = PureWindowsPath(filepath + "\\GitHub\\TemporalRegularity_RWD2023\\Results\\Generalization Results\\" + mode + "_" + str(num_vocs) + "vocs_" +  all_prefixes[cur_trial] + "_newclass.mat")
        #set which vocals are used
        prefixes = all_prefixes[cur_trial] #['co', 'gt']#['co', 'ha']#['sb', 'gt'] #['sb', 'co'] #['ha', 'gt'] #['ha', 'sb'] 
        #can leave out for now 2021-04-22# compare_cat = list(['coo vs arch', 'coo vs coo', 'arch vs arch' ])
        training_cat = all_training_cat[cur_trial]
        
        unpack_data = io.loadmat(results_file)
        
        unpack_pairs = unpack_data['stimuli_ind_list']
        all_pairs = unpack_data['all_pairs']
    
        
    
        #20 vocal sets x 5 repeats x 1 training condition since there is only all coos.
        SFA_scores_batch = unpack_data['SFA_scores_batch']
        
        if SFA_scores_batch.shape[2] == 3:
            SFA_scores_batch = SFA_scores_batch[:,:,0]
            
        avg_per_vocal_SFA = np.mean(SFA_scores_batch, axis = 1)*100.0
        #Basic stats on above figure
        stats_test = 'Mann_Whitney U, one-tailed, X > Y'
        #check above chance at lowest snr
        print('') #I'm sure there is code for setting a new line, but this works for now
        print(mode + ' ' + all_prefixes[cur_trial] +' vs Chance test statistic and P value using ' + stats_test + ':')
        genvschance = stats.mannwhitneyu(avg_per_vocal_SFA[:,0],chance_lvl[0]*np.ones(np.shape(avg_per_vocal_SFA[:,0])),alternative = 'greater')
        print(genvschance.statistic)
        print(genvschance.pvalue)
            
#%% Generalization testing difference is vocal classes.            
  

num_vocs = 2



modes = ["PCA","ICA","Linear","Quad"]

SFA_together = np.zeros((20,4,4)) #note hardcoding for now for ease

this_ind = 0 #silly workaround for indexing system
for mode in modes:
    for index in  range(0,np.size(all_trials)): #temp_list: #
        cur_trial = all_trials[index]
        results_file = PureWindowsPath(filepath + "\\GitHub\\TemporalRegularity_RWD2023\\Results\\Generalization Results\\" + mode + "_" + str(num_vocs) + "vocs_" +  all_prefixes[cur_trial] + "_newclass.mat")
        #set which vocals are used
        prefixes = all_prefixes[cur_trial] #['co', 'gt']#['co', 'ha']#['sb', 'gt'] #['sb', 'co'] #['ha', 'gt'] #['ha', 'sb'] 
        #can leave out for now 2021-04-22# compare_cat = list(['coo vs arch', 'coo vs coo', 'arch vs arch' ])
        training_cat = all_training_cat[cur_trial]
        
        unpack_data = io.loadmat(results_file)
        
        unpack_pairs = unpack_data['stimuli_ind_list']
        all_pairs = unpack_data['all_pairs']
    
        
    
        #20 vocal sets x 5 repeats x 1 training condition since there is only all coos.
        SFA_scores_batch = unpack_data['SFA_scores_batch']
        
        if SFA_scores_batch.shape[2] == 3:
            SFA_scores_batch = SFA_scores_batch[:,:,0]
            
        avg_per_vocal_SFA = np.mean(SFA_scores_batch, axis = (1,2))*100.0 #changed axis to prevent broadcasting issue
        
        SFA_together[:,index,this_ind] = avg_per_vocal_SFA #store this for stats out of loop

    this_ind += 1 # go to the next part of the array
    
#Can't do two-factor kwallis test, so going to take average over all algorithms
#Then do that test and a follow up mannwhitt


avg_across_algo = np.mean(SFA_together, axis = 2)


stats_test = 'Kruskal Wallis, medians are equal across groups'

print('') #I'm sure there is code for setting a new line, but this works for now
print('Average across all algos for each vocalization' +' vs Chance test statistic and P value using ' + stats_test + ':')
vocalsvssame = stats.kruskal(avg_across_algo[:,0],avg_across_algo[:,1],avg_across_algo[:,2],avg_across_algo[:,3])
print(vocalsvssame.statistic)
print(vocalsvssame.pvalue)

print('Median for Coo, HarmArch, Grunt, Bark where each row is a vox and each column is a method')
print(np.median(SFA_together, axis = 0))
print('IQR in same order as above')
print(stats.iqr(SFA_together, axis = 0))


#%% Post-hoc tests 
#First coo vs all

stats_test = 'Mann_Whitney U, two-tailed, X = Y'

main_trial = all_trials[0]
for index in range(0,avg_across_algo.shape[1]-1):
    
    cur_trial = all_trials[index+1]
    print('') #I'm sure there is code for setting a new line, but this works for now
    print(all_prefixes[main_trial] + ' vs ' + all_prefixes[cur_trial] +' test statistic and P value using ' + stats_test + ':')
    voxvsvox = stats.mannwhitneyu(avg_across_algo[:,0],avg_across_algo[:,index+1])
    print(voxvsvox.statistic)
    print(voxvsvox.pvalue)

#Not going to report, but now check other vocalization classes.   Note with two tailed coo will still come out as signficant for all of them

#for simplicity just going to do a straightforward hard code.  Just need this done

stats_test = 'Mann_Whitney U, two-tailed, X = Y'
#harmonic arch vs all (if this doesn't show a difference then no need for the rest)
#Update: great that is the case.  Only significantly difference for coo due to two tailed test
main_trial = all_trials[1]
for index in [0,2,3]:
    
    cur_trial = all_trials[index]
    print('') #I'm sure there is code for setting a new line, but this works for now
    print(all_prefixes[main_trial] + ' vs ' + all_prefixes[cur_trial] +' test statistic and P value using ' + stats_test + ':')
    voxvsvox = stats.mannwhitneyu(avg_across_algo[:,1],avg_across_algo[:,index])
    print(voxvsvox.statistic)
    print(voxvsvox.pvalue)

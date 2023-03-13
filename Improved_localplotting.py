# -*- coding: utf-8 -*-
"""

@author: ronwd

"""

#%% Import everything

import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path, PureWindowsPath
from scipy import io
import matplotlib.pyplot as plt 
#%% Set seaborn stuff
sns.set_theme() #set seaborn default theme


#grabbed hex codes if needed for plots, set size of font consistent across figures
#set up easy way to label SFA vs linear model with same color

muted=["#4878CF", "#6ACC65", "#D65F5F",
           "#B47CC7", "#C4AD66", "#77BEDB", "#abacb3"]

#blue, green, red, purple, tan, teal/lightblue, grey



newPal   = dict([('SFA' , muted[0]), ('Linear Upper Bound', muted[2]), ('Linear Null Model' , muted[2])])



#sns.set(font_scale = 1.5)
sns.set_context("paper")
#%% Feature sweep and SNR plots


######################Feature Sweep Plot#########################################

filepath = "Your cool file path goes here"
results_file = PureWindowsPath( filepath + "\\GitHub\\TemporalRegularity_RWD2023\\Results\\FeatureSweep.mat")
feature_range = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,20])
feature_range = np.reshape(np.tile(feature_range,(100,1)).T, (1600,1))

unpack_data = io.loadmat(results_file)
SFA_scores_batch = unpack_data['SFA_save_score'] 
SFA_scores_batch  = np.reshape(SFA_scores_batch,[100,16])
SFA_scores_batch = np.hstack(SFA_scores_batch.T)

#Having each row is an obvservation and each column is a number of features used
#5 repeats don't matter so will get 1600 x 2 one for the performance and one for the number of features that goes with that observation

feature_data= np.hstack([np.reshape(SFA_scores_batch, [100*16,1])*100,feature_range])

data_frame1 = pd.DataFrame(feature_data, columns = list(['Performance', '# of Features Used']))
data_frame1['# of Features Used'] = data_frame1['# of Features Used'].astype('int')
    



plt.figure()
sns.set_style("ticks")

p1 = sns.pointplot(x = '# of Features Used', y = 'Performance', data = data_frame1,
 estimator = np.mean,
 linestyles= '--',
 color = 'k',
 capsize = .5
 )
plt.ylim([80, 101])
plt.xlabel('Number of Features Used')
plt.ylabel('Classifer Performance (% Correct)')



#%%
##################SNR Plots ###################################################

mode = "ICA"
condition = "testclutter"
stimulus_type = "vocals"

#Plot highest SNR and SNR sweep
results_file = PureWindowsPath(filepath + "\\GitHub\\TemporalRegularity_RWD2023\\Results\\Clutter Results\\SNR_" + mode + "_" + condition + "_newclass_" + stimulus_type + ".mat") #newsplit_try_Clutter_Aug.mat #newsplit_try2_Clutter_Aug.mat




#Unpack everything

unpack_data = io.loadmat(results_file)
unpack_pairs = unpack_data['stimuli_ind_list']
all_pairs = unpack_data['all_pairs']
Baseline_scores_batch = unpack_data['Baseline_scores_batch']
SFA_scores_batch = unpack_data['SFA_scores_batch']
baseline_file = PureWindowsPath(filepath + "\\GitHub\\TemporalRegularity_RWD2023\\Results\\Clutter Results\\\\SNR_altbaseline_" + condition + "_newclass_" + stimulus_type + ".mat") #newsplit_try_Clutter_Aug.mat #newsplit_try2_Clutter_Aug.mat  #PureWindowsPath("C:\\Users\\ronwd\\OneDrive\\Documents\\GitHub\\SFA_results_and_figures\\Post_Fix_Results\\Results_2021_04_19\\Alt_Baseline_try1_newcode_JAN.mat")

unpack_data = io.loadmat(baseline_file)
LB_Baseline_scores_batch = unpack_data['Baseline_scores_batch']

#Set up condition variables for data frame 
snr_values = np.log10(np.array([1e-7, 1e-5, 1e-3, .1, 1.0, 10.0, 100.0, 1000.0])) 
snr_num = snr_values.shape[0]
snr_values = np.reshape(np.tile(snr_values,(SFA_scores_batch.shape[0]*SFA_scores_batch.shape[1],1)).T, (SFA_scores_batch.shape[0]*SFA_scores_batch.shape[1]*snr_values.shape[0]))
snr_values = snr_values.astype('int')
UB_label = ['UB'] * snr_values.size #'UB' if want on separate column rather than highest SNR or put 3 if want on same line
Null_label  = ['Null'] * snr_values.size #'Null' (see above)

#Set up labels for which algorithm as used, slight redundant with above but need these for hue

SFA_label = ['SFA'] * snr_values.size
Linear_UB_label = ['Linear Upper Bound']* snr_values.size
Linear_Null_label = ['Linear Null Model']* snr_values.size

#stack data and labels


SFA_scores_batch  = np.reshape(SFA_scores_batch,[SFA_scores_batch.shape[0]*SFA_scores_batch.shape[1],snr_num])*100
SFA_scores_batch = np.hstack(SFA_scores_batch.T)

Baseline_scores_batch  = np.reshape(Baseline_scores_batch,[Baseline_scores_batch.shape[0]*Baseline_scores_batch.shape[1],snr_num])*100
Baseline_scores_batch = np.hstack(Baseline_scores_batch.T)

LB_Baseline_scores_batch  = np.reshape(LB_Baseline_scores_batch,[LB_Baseline_scores_batch.shape[0]*LB_Baseline_scores_batch.shape[1],snr_num])*100
LB_Baseline_scores_batch = np.hstack(LB_Baseline_scores_batch.T)

#Get each vocal and put it in its own column.  Squeeze is because it is an array of arrays nonsense, stack helps a bit but still and array of an array
Vocal1 = np.stack(np.squeeze(np.tile(all_pairs[:,0],(1,120)))).tolist()
Vocal1 = np.concatenate(Vocal1) 
Vocal2 = np.stack(np.squeeze(np.tile(all_pairs[:,1],(1,120)))).tolist()
Vocal2 = np.concatenate(Vocal2) 

pair_numbers = np.stack(unpack_pairs).tolist()
pair_numbers = list(np.concatenate(pair_numbers).flat)
pair_numbers = np.stack(np.squeeze(np.tile(pair_numbers,(1,120)))).tolist()


pair_ID = np.empty([Vocal1.size,1], dtype = 'object')
pair_type = np.empty([Vocal1.size,1], dtype = 'object')

types = ['Between-Vocalization Classes', 'Within-Vocalization Classes']

for vox in range(0,Vocal1.size):
    
       pair_ID[vox,:] = [Vocal1[vox][0:2]+Vocal2[vox][0:2]]
       pair_type[vox,:] = types[Vocal1[vox][0:2]==Vocal2[vox][0:2]]
       
pair_ID = np.reshape(pair_ID, Vocal1.shape)
pair_type = np.reshape(pair_type,Vocal1.shape)

Performance = np.hstack([SFA_scores_batch, Baseline_scores_batch, LB_Baseline_scores_batch])

Condition = np.hstack([snr_values, UB_label, Null_label])

Algorithm = np.hstack([SFA_label,Linear_UB_label,Linear_Null_label])



data_frame2 = pd.DataFrame({ 
    
    'Performance': Performance, # % correctly classified
    'Condition': Condition,     # SNR level if SFA or linear model conditions if linear model
    'Method Used': Algorithm,     # somewhat replicates above but groups all SNR levels under the heading SFA
    'Vocal 1': Vocal1,          # Actual wav file names in case need that info
    'Vocal 2': Vocal2,
    'Pair Number': pair_numbers, #what pair number it was in that set (0-99)
    'Pair Identity': pair_ID   ,      #what classes were compared and in what order (e.g. cogt is coo vs grunt with coo as vocal1)
    'Pair Type': pair_type #whether the comparison was within or between vocal classes
    })

    
# Just plot highest SNR as violin plot with other two conditions
##################NOTE HARD CODING INDEX FOR PLOT HERE###############################
#Change to 350 and change y lim to 115 for applause
plt.figure()
sns.set_style("ticks") #############3500 for normal case, adjust for app to 350
p2 = sns.violinplot(x = 'Method Used', y = 'Performance',data = data_frame2[3500:data_frame2.shape[0]],
 estimator = np.median,
 palette = newPal, 
 )

plt.ylim([45, 110])
plt.title(mode)


plt.figure()
sns.set_style("ticks")
p2s = sns.violinplot(x = 'Pair Type', y = 'Performance',data = data_frame2.loc[((data_frame2['Method Used']=='SFA') & ( data_frame2['Condition']  == '3'))],
 estimator = np.median,
 color = 'b'
 
 )

plt.ylim([45, 105])
sns.set_style("ticks")
plt.ylabel('Classifer Performance (% Correct)')
plt.title(mode)

plt.figure()



p3 = sns.pointplot(x = 'Condition', y = 'Performance', hue = 'Method Used',data = data_frame2,
 estimator = np.median,
 linestyles= '--',
 palette = newPal,
 #capsize = .25,

 )
p3.set(xlabel = condition +' Signal-to-Noise Ratio (dB; log 10)')

plt.setp(p3.lines, zorder = 100)
plt.title(mode)


plt.ylim([50, 101])


plt.ylabel('Classifer Performance (% Correct)')

p3 = sns.pointplot(x = 'Condition', y = 'Performance', hue = 'Method Used',data = data_frame2,
 estimator = np.median,
 linestyles= '--',
 palette = newPal,

 capsize = .25,

 )
p3.set(xlabel = 'Signal-to-Noise Ratio (dB; log 10)')
plt.setp(p3.collections, zorder = 100)

p3.legend_.remove()
#Having issues with this so just going to take a legend over in illustrator

plt.ylabel('Classifer Performance (% Correct)')

#%% SNR plots for double clutter (clutter in training and testing data sets)

mode = "ICA"
condition = "dc_full"
stimulus_type = "vocals"

#Plot highest SNR sweep
results_file = PureWindowsPath(filepath + "\\GitHub\\TemporalRegularity_RWD2023\\Results\\Clutter Results\\SNR_" + mode + "_" + condition + "_newclass_" + stimulus_type + ".mat") #newsplit_try_Clutter_Aug.mat #newsplit_try2_Clutter_Aug.mat




#Unpack everything

unpack_data = io.loadmat(results_file)
unpack_pairs = unpack_data['stimuli_ind_list']
all_pairs = unpack_data['all_pairs']
Baseline_scores_batch = unpack_data['Baseline_scores_batch']
SFA_scores_batch = unpack_data['SFA_scores_batch']
#In this case only use test clutter
baseline_file  = PureWindowsPath(filepath + "\\GitHub\\TemporalRegularity_RWD2023\\Results\\Clutter Results\\SNR_altbaseline_testclutter_newclass_" + stimulus_type + ".mat")  #PureWindowsPath("C:\\Users\\ronwd\\OneDrive\\Documents\\GitHub\\SFA_results_and_figures\\Post_Fix_Results\\Results_2021_04_19\\Alt_Baseline_try1_newcode_JAN.mat")

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
#Update: apparently there is the function in numpy called ravel that will do this for me more or less...the more you know

SFA_scores_batch = np.mean(SFA_scores_batch, axis = 1)*100
Baseline_scores_batch = np.mean(Baseline_scores_batch, axis =1)*100
LB_Baseline_scores_batch = np.mean(LB_Baseline_scores_batch , axis =1)*100
#Note: since Baseline is not run in the double clutter code proper we have first take the mean and then repeat 8 times along the 2nd snr axis so it is the same size as the other variables
LB_Baseline_scores_batch = np.repeat(np.reshape(LB_Baseline_scores_batch,(unpack_pairs.size,snr_num,1)),snr_num, axis = 2)

#Ravel will give us a column where each grouping of 8 are the 8 different TEST SNR for that particular TRAINING SNR
SFA_scores_batch = np.ravel(SFA_scores_batch)
Baseline_scores_batch = np.ravel(Baseline_scores_batch)
LB_Baseline_scores_batch = np.ravel(LB_Baseline_scores_batch)
#For training snr, have the same training snr for each test_snr.  So first repeat snr_num times, then tile this for each vocal pair
train_snr  = np.tile(np.repeat(snr_values,snr_num),unpack_pairs.size)
#For testing, simply tile the SNR so you have a set for each training snr for each vocal pair.
test_snr = np.tile(snr_values,int(SFA_scores_batch.size/snr_num))

UB_label = ['UB'] * train_snr.size #'UB' if want on separate column rather than highest SNR or put 3 if want on same line
Null_label  = ['Null'] * train_snr.size #'Null' (see above)

#Set up labels for which algorithm as used, slight redundant with above but need these for hue

SFA_label = ['SFA'] * train_snr.size
Linear_UB_label = ['Linear Upper Bound']* train_snr.size
Linear_Null_label = ['Linear Null Model']* train_snr.size

#2022-09-13 This also no longer works...unforunately.  Wrong repeats.
#Think we need to do repeat instead of tile we used above.  Need to do this like
#train_snr

Vocal1 = np.stack(np.tile(np.repeat(all_pairs[:,0],snr_num),snr_num)).tolist()
Vocal1 = np.concatenate(Vocal1)
Vocal2 = np.stack(np.tile(np.repeat(all_pairs[:,1],snr_num),snr_num)).tolist()
Vocal2 = np.concatenate(Vocal2) #finally gets it out of that dumb array of array... form


pair_numbers = np.stack(unpack_pairs).tolist()
pair_numbers = list(np.concatenate(pair_numbers).flat)
pair_numbers = np.stack(np.tile(np.repeat(pair_numbers,snr_num),snr_num)).tolist()

#Nevermind this works provided Vocal1 and Vocal2 are right
#Sucks to do this as a for loop but I can't think of another solution right now.
pair_ID = np.empty([Vocal1.size,1], dtype = 'object')
pair_type = np.empty([Vocal1.size,1], dtype = 'object')

types = ['Between-Vocalization Classes', 'Within-Vocalization Classes']

for vox in range(0,Vocal1.size):
    
       pair_ID[vox,:] = [Vocal1[vox][0:2]+Vocal2[vox][0:2]]
       pair_type[vox,:] = types[Vocal1[vox][0:2]==Vocal2[vox][0:2]]
       
pair_ID = np.reshape(pair_ID, Vocal1.shape)
pair_type = np.reshape(pair_type,Vocal1.shape)

#Finally repeat everything so we have the duplicated information for each row
#only alternative is to have scores for each algorithm be their own column...
#not a terrible idea...but would make the plotting quite different from above and
#potentially require even more rewrites of the above code... no think stick with current format
#since it is less multiple values for each column style then the alternative.

Vocal1 = np.tile(Vocal1,3)
Vocal2 = np.tile(Vocal2,3)
pair_ID = np.tile(pair_ID,3)
pair_type = np.tile(pair_type,3)
pair_numbers = np.tile(pair_numbers,3)
train_snr = np.tile(train_snr,3) #Technically doesn't matter/count for ub and null model but just puting in for ease
test_snr = np.tile(test_snr,3)

#need to fix condition too and add an snr train and snr test column to the chart

Performance = np.hstack([SFA_scores_batch, Baseline_scores_batch, LB_Baseline_scores_batch])

Algorithm = np.hstack([SFA_label,Linear_UB_label,Linear_Null_label])

data_frame3 = pd.DataFrame({ 
    
    'Performance': Performance, # % correctly classified
    'TrainingSNR': train_snr,     # SNR level for training (technically only matters for SFA but puting in for convenience)
    'TestingSNR': test_snr,
    'Method Used': Algorithm,     # somewhat replicates above but groups all SNR levels under the heading SFA
    'Vocal1': Vocal1,          # Actual wav file names in case need that info
    'Vocal2': Vocal2,
    'PairNumber': pair_numbers, #what pair number it was in that set (0-99)
    'PairIdentity': pair_ID   ,      #what classes were compared and in what order (e.g. cogt is coo vs grunt with coo as vocal1)
    'PairType': pair_type #whether the comparison was within or between vocal classes
    })

#Data frame looks like it loads correctly
#Now remove the missing entries from pair 37

data_frame3 = data_frame3.dropna()


# Just plot highest SNR as violin plot with other two conditions
#2022-09-14 Note: these plots should look exactly the same as the ones for the highest conditions above unless something weird is happening

#This code now works.  May want to modify above to match it as it is a cleaner solution...but leave alone for now
plt.figure()
sns.set_style("ticks") 
p2 = sns.violinplot(x = 'Method Used', y = 'Performance',data = data_frame3[(data_frame3.TrainingSNR == 3) & (data_frame3.TestingSNR == 3)],
 estimator = np.median,
 palette = newPal, 
 )
plt.ylim([45, 115])

#Interesetingly...this looks slightly different from before for quad...but is not a major part of the story
plt.figure()
sns.set_style("ticks")
p2s = sns.violinplot(x = 'PairType', y = 'Performance',data = data_frame3[(data_frame3.TrainingSNR == 3)
& (data_frame3.TestingSNR == 3)
&(data_frame3['Method Used']=='SFA')],
 estimator = np.median,
 color = 'b'
 
 )

plt.ylim([45, 105])
sns.set_style("ticks")
plt.ylabel('Classifer Performance (% Correct)')

plt.figure()

#Now this will take some doing...essentially need a line for each testing SNR
#that goes across each traing SNR...Then maybe add the linear model dots at the end
#Making two plots this time since sweeping testing snr as well
#Update: Seaborn calculates the confidence interval non-parametrically using bootstrapping
#hence the non-symmetric confidence interval


p3 = sns.pointplot(x = 'TrainingSNR', y = 'Performance', hue = 'TestingSNR',data = data_frame3[(data_frame3['Method Used']=='SFA')&(data_frame3.TestingSNR<0)],
 estimator = np.median,
 linestyles= '-',
 capsize = .25,

 )
p3.set(xlabel = 'Training Signal-to-Noise Ratio (dB; log 10)')

plt.setp(p3.lines, zorder = 100)
plt.legend(ncol = 4,loc = 'upper left', title = 'Testing SNR')
plt.ylim([45, 105])
plt.title(mode)

plt.figure()
p4 = sns.pointplot(x = 'TrainingSNR', y = 'Performance', hue = 'TestingSNR',data = data_frame3[(data_frame3['Method Used']=='SFA')&(data_frame3.TestingSNR>=0)],
 estimator = np.median,
 linestyles= '-',
 capsize = .25,

 )
p4.set(xlabel = 'Training Signal-to-Noise Ratio (dB; log 10)')

plt.setp(p4.lines, zorder = 100)
plt.legend(ncol = 4,loc = 'upper left', title = 'Testing SNR')
plt.ylim([45, 105])
plt.title(mode)

#%%
##################Novel Vocalization Plots#####################################
#plt.close('all') #leave for now so we don't get 20 figures

#2022-09-13 this needs to be completely rewritten for the new generalization work.
#Copy and pasting it to the old code section and then copying bits that still
#can be used as it goes.
#Can probably borrow a bit from above code too.  This will be first thing tomorrow morning work
#
#2022-09-12: Note we now do the leave out procedure for testing generalization

#2022-09-15 Need to have this working for at least two vocalizations before meeting tomorrow
#Update: Success we have the basic plot working




#Set up things universal to all data sets
columns = ['Performance','Pair', 'Pair Number','NumVox','Trial','Vocal1','Vocal2','Vocal3','Vocal4']

data_frame4 = pd.DataFrame(columns = columns) #initialize data frame, just keep appending to get things

chance_lvl = list([50.0,33.3,25.0,20.0]) #list of chance levels for changing axis limit

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

for index in  range(0,np.size(all_trials)): #temp_list: #
        cur_trial = all_trials[index]
        results_file = PureWindowsPath(filepath + "\\GitHub\\TemporalRegularity_RWD2023\\Results\\Generalization Results\\"  + mode + "_" + str(num_vocs) + "vocs_" +  all_prefixes[cur_trial] + "_newclass.mat") #multi_try1_newcode_JAN.mat
        #set which vocals are used
        prefixes = all_prefixes[cur_trial] #['co', 'gt']#['co', 'ha']#['sb', 'gt'] #['sb', 'co'] #['ha', 'gt'] #['ha', 'sb'] 
        #can leave out for now 2021-04-22# compare_cat = list(['coo vs arch', 'coo vs coo', 'arch vs arch' ])
        training_cat = all_training_cat[cur_trial]
        
        unpack_data = io.loadmat(results_file)
        
        unpack_pairs = unpack_data['stimuli_ind_list']
        all_pairs = unpack_data['all_pairs']
        
        #Get each vocal and put it in its own column.  Squeeze is because it is an array of arrays nonsense, stack helps a bit but still and array of an array
        Vocal1 = np.stack(np.squeeze(np.tile(all_pairs[:,0],(1,5)))).tolist()
        Vocal1 = np.concatenate(Vocal1) #finally gets it out of that dumb array of array... form
        Vocal2 = np.stack(np.squeeze(np.tile(all_pairs[:,1],(1,5)))).tolist()
        Vocal2 = np.concatenate(Vocal2) #finally gets it out of that dumb array of array... form
        #I'm sure there is a matrix way to do this but I'm sick of looking for it
        


        #Think we need this too to generate the plot Yale had in mind
        
        
        
        #20 vocal sets x 5 repeats x 1 training condition since there is only all coos.
        SFA_scores_batch = unpack_data['SFA_scores_batch']
        
        if SFA_scores_batch.shape[2] == 3:
            SFA_scores_batch = SFA_scores_batch[:,:,0]
        
        SFA_scores_batch  = np.reshape(SFA_scores_batch,[SFA_scores_batch.shape[0]*SFA_scores_batch.shape[1],1])
        SFA_scores_batch = np.hstack(SFA_scores_batch.T)*100.0
        
        #Pairs are just prefixes for this trial
        pair_ID = all_prefixes[cur_trial][0]+all_prefixes[cur_trial][1]
        
        #Pair number is so can reference the same vocal set, deal with repeats first then training conditions
        temp = np.concatenate(unpack_pairs.tolist())
        pair_num = np.tile(temp, (5,1))
        pair_num = np.hstack(pair_num)
        pair_num = np.tile(pair_num, 1)
        

        
   
        
 
        #num vocs goes straight in
        
  
    
        df_temp = pd.DataFrame({
            
            'Performance': SFA_scores_batch,
            'Pair' : pair_ID,
            'Pair Number' : pair_num ,
            'NumVox': num_vocs,
            'Trial': cur_trial,
            'Vocal1': Vocal1, #Don't really need these but nice to have a double check/reference
            'Vocal2': Vocal2,
      


            }
                                )
        
        
        data_frame4 = data_frame4.append(df_temp) #simply 

#Change color scheme to avoid confusion with above color scheme


b2g = ["#4f4e4e", "#706f6f","#a3a0a0","#e3e1e1"]

newPal   = dict([('co' , b2g[0]), ('ha', b2g[1]), ('gt' , b2g[2]),('sb' , b2g[3])])


plt.figure()
sns.set_style("ticks")
p2s = sns.violinplot(x = 'Pair', y = 'Performance',data = data_frame4,
 estimator = np.median,
  palette = newPal,
 
 )
    
plt.ylim([35, 109])


%% Running notes
%2022-06-02: Setting this script up to practice with the modfiltering and
%with the inverse mod spectrum.  First just trying the modfiltering on
%white noise exemplar to make sure things are working as predicted


%% Load example
%folderused ='C:\Users\ronwd\OneDrive\Documents\GitHub\Local_EigenSpecAudioStim\ClutterVocal_Eigendata\NoiseStimuliWav\';

use_vocals = 0;

if use_vocals
    
folderused = 'C:\Users\ronwd\.spyder-py3\SFA_PostCOSYNEAPP-master\Monkey_Calls\HauserVocalizations\Monkey_Wav_50K\Wav\';


    
    
else

folderused = 'C:\Users\ronwd\OneDrive\Documents\GitHub\SFA_PostCOSYNEAPP\Clutter_Stim_juneon\21-Jul-2020_clutter\';

temp = dir(folderused);

filenames = {temp(3:end).name};

index = 10;

[soundin, samprate] = audioread(char([folderused...
            filenames{index}]));
        
end
        
soundsc(soundin,samprate) %note: this can have weird slow down effects if you don't pass the sample rate
        
%% Try modfilter code

%Borrowing parameters from onemodfilter code
%from this code can see wants to pass in filters are multiple struct object
%2022-06-02: the code runs but I do not hear a difference with these
%parameters or see a difference in the plotted modulation spectra

filter.fband = 32; %Use 32 for human speech in o.g. code so just starting here
filter.method = 1; %Use notch filter
filter.wf_high = 0.003
filter.wt_high = 1;
filter.wf_it = 0;
filter.wt_it = 0;
%These are just saving things that have to be passed.
filter.song_path = cd;
filter.mod_song_name = ['modded_' filenames{index}];
filter.mod_song_path = cd;

sound_filtered = modfilter(soundin,samprate,filter.fband,filter.method,filter.wf_high,...
    filter.wt_high,filter.wf_it,filter.wt_it,filter);

%Note: amplitude plot IS modspec from our other code.

soundsc(sound_filtered,samprate)
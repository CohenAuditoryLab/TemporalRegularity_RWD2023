
%% Running notes

%2022-09-29 Decide to make this its own separate file instead as toggling
%this in addition to all of the other toggles will be obnoxious.  Need to
%figure out exactly what Yale meant...only do this for correlation
%analysis?  Make the null distribution from randomly sampling from all
%correlation matrices for all types of vocalizations?  Random sampling
%before or after making into a spectogram?  Difference from shuffling?
%Difference from mod spec of whitenoise?

%For first idea going to try to generate a stack of spectrograms and then
%roll their columns randomly between each one to make a null distribution.
%For simplicty of recoding and computational load.  First going to make a
%separate null for each class and catagory in voc_types.  And at first
%going to do this for the non-quadratic case as it doesn't take as long to
%run.

%Update: changed mind doing shuffle over all class types

%%

% Some parameters
samprate = 50000;                          % 50k sampling rate
use_textures = 0; 
if use_textures
    
    samprate = 20000; %sampling rate was only 20k for McDermott files
    %Needt remember this for when port over to check performance
    
end

fband = 32;                                % Frequency band in Hz
twindow = 1000*6/(fband*2.0*pi);           % Window length in ms - 6 times the standard dev of the gaussian window
winLength = fix(twindow*samprate/1000.0);  % Window length in number of points
increment = fix(0.001*samprate);            % Sampling rate of spectrogram in number of points - set at 1 kHz
debug_fig = 0;                              % Set to 1 to see spectrograms
maxlenuser = 1000;                          % Maximun length of window in ms for estimating modspectrum.  Vocalizations
corr_spec = 0;                              %Set whether to take mod spec of og spectrogram or quadratically expanded (i.e include correlations)
nolin = 1;                                  %Toggle whether to remove linear component in quadratic expansion
                          %Toggle whether to use vocalizations or textures 



% that are longer will be divided into chunks.  Vocalizations that are
% shorter will be padded with zero

% List of sound files for the different vocalizations 2022-04-19 update
% adding clutter and white noise values
ag_filenames = {'ag10944.wav',  'ag16944.wav',  'ag1e55.wav',   'ag1o49.wav',   'ag3575.wav',   'ag5944.wav',  ... 
'ag11944.wav',  'ag1710.wav',   'ag1f24.wav',   'ag2575.wav',   'ag393b.wav',   'ag693b.wav',   ...
'ag12944.wav',  'ag1713.wav',   'ag1g62.wav',   'ag282b.wav',   'ag3944.wav',   'ag6944.wav',   ...
'ag13944.wav',  'ag1787.wav',   'ag1h79.wav',   'ag293b.wav',   'ag3k57.wav',   'ag793b.wav',   ...
'ag141c.wav',   'ag182b.wav',   'ag1i04.wav',   'ag2944.wav',   'ag3k89.wav',   'ag7944.wav',   ...
'ag1470.wav',   'ag182c.wav',   'ag1j24.wav',   'ag2978.wav',   'ag4575.wav',   'ag8944.wav',   ...
'ag14944.wav',  'ag193b.wav',   'ag1k57.wav',   'ag2k57.wav',   'ag493b.wav',   'ag9944.wav',   ...
'ag1535.wav',   'ag1944.wav',   'ag1k89.wav',   'ag2k89.wav',   'ag4944.wav',   ...
'ag1575.wav',   'ag1978.wav',   'ag1n37.wav',   'ag2o27.wav',   'ag4k57.wav',   ...
'ag15944.wav',  'ag1c75.wav',   'ag1o27.wav',   'ag2o49.wav',   'ag593b.wav' };

co_filenames = { 'CO1A94.wav',   'co1491.wav',   'co1671.wav',   'co1933.wav',   'co2d14.wav',   'co4845.wav',   ...
'CO1A96.wav',   'co151f.wav',   'co16845.wav',  'co1o13.wav',   'co2f24.wav',   'co4f24.wav',   ...
'co1405.wav',   'co159b.wav',   'co168f.wav',   'co1z67.wav',   'co3845.wav',   'co5f24.wav',   ...
'co1480.wav',   'co1660.wav',   'co176e.wav',   'co2845.wav',   'co3f24.wav' };

cs_filenames = { 'cs10g62.wav',  'cs2e54.wav',   'cs3g62.wav',   'cs4o49.wav',   'cs6g62.wav',   'cs9a59.wav',  ... 
'cs11a59.wav',  'cs2g62.wav',   'cs3hh6.wav',   'cs5a59.wav',   'cs6o49.wav',   'cs9g62.wav',   ...
'cs1t53.wav',   'cs2h46.wav',   'cs3o49.wav',   'cs5e54.wav',   'cs7a59.wav',   ...
'cs2a59.wav',   'cs2o49.wav',   'cs4a59.wav',   'cs5g62.wav',   'cs7g62.wav',   ...
'cs2a74.wav',   'cs2t53.wav',   'cs4e54.wav',   'cs5o49.wav',   'cs8a59.wav',   ...
'cs2c99.wav',   'cs3354.wav',   'cs4g62.wav',   'cs6a59.wav',   'cs8g62.wav' };

gk_filenames = { 'gk151f.wav',   'gk1f777.wav',  'gk1fn37.wav',  'gk2fn37.wav',  'gk4fx33.wav',  ...
'gk160f.wav',   'gk1fe04.wav',  'gk1fx33.wav',  'gk2fx33.wav',  'gk5fx33.wav',  ...
'gk169f.wav',   'gk1fe55.wav',  'gk251f.wav',   'gk3f11c.wav',  'gk6fx33.wav',  ...
'gk173f.wav',   'gk1fk80.wav',  'gk2f11c.wav',  'gk3fn37.wav',  'gk7fx33.wav',  ...
'gk1f39a.wav',  'gk1fllc.wav',  'gk2fk80.wav',  'gk3fx33.wav' };

gt_filenames = { 'gt10a94.wav',  'gt1a94.wav',   'gt1k3.wav',    'gt2i04.wav',   'gt3k3.wav',    'gt7a94.wav',   ...
'gt117d.wav',   'gt1a96.wav',   'gt1n37.wav',   'gt2k3.wav',    'gt4845.wav',   'gt9a94.wav',   ...
'gt176e.wav',   'gt1b10.wav',   'gt217d.wav',   'gt382c.wav',   'gt4a94.wav',   ...
'gt182c.wav',   'gt1e55.wav',   'gt282c.wav',   'gt3845.wav',   'gt5845.wav',   ...
'gt1842.wav',   'gt1f29.wav',   'gt2845.wav',   'gt3a94.wav',   'gt5a94.wav',   ...
'gt1845.wav',   'gt1g32.wav',   'gt2a96.wav',   'gt3a96.wav',   'gt6845.wav',   ...
'gt190b.wav',   'gt1i04.wav',   'gt2f29.wav',   'gt3g32.wav',   'gt6a94.wav',   ...
'gt1912.wav',   'gt1i10.wav',   'gt2g32.wav',   'gt3i04.wav',   'gt7845.wav'};

gy_filenames = { 'gy114d.wav',  'gy1a94.wav',  'gy1z75.wav',  'gy3275.wav',  'gy5275.wav',  ...
'gy117d.wav',  'gy1g35.wav',  'gy217d.wav',  'gy3845.wav',  'gy5845.wav',  ...
'gy1275.wav',  'gy1k80.wav',  'gy2275.wav',  'gy3g35.wav',  'gy6275.wav',  ...
'gy1845.wav',  'gy1l45.wav',  'gy2845.wav',  'gy4257.wav',  'gy7275.wav',  ...
'gy1944.wav',  'gy1l63.wav',  'gy2g35.wav',  'gy4845.wav',  'gya94.wav' };

ha_filenames = { 'ha1480.wav',  'ha1h38.wav',  'ha1n41.wav',  'ha2845.wav',  'ha3d14.wav',  ...
'ha1842.wav',  'ha1i06.wav',  'ha1unk.wav',  'ha2d14.wav',  'ha3unk.wav',  ...
'ha1845.wav',  'ha1i77.wav',  'ha1ya.wav',   'ha2unk.wav',  'ha4d14.wav',  ...
'ha1g61.wav',  'ha1k83.wav',  'ha2480.wav',  'ha2ya.wav'};

sb_filenames = { 'sb1548.wav',  'sb1h85.wav',  'sb2837.wav',  'sb3837.wav',  'sb5837.wav',  ...
'sb1832.wav',  'sb1i05.wav',  'sb2933.wav',  'sb3933.wav',  'sb6837.wav',  ...
'sb1837.wav',  'sb1l68.wav',  'sb2h85.wav',  'sb3h85.wav',  'sb7837.wav',  ...
'sb1933.wav',  'sb2832.wav',  'sb2l68.wav',  'sb4837.wav',  'sb8837.wav', };

sc_filenames = { 'sc10g65.wav',  'sc169f.wav',   'sc1d15.wav',   'sc251f.wav',   'sc3d15.wav',   'sc5g65.wav',  ... 
'sc11g65.wav',  'sc16g65.wav',  'sc1fk89.wav',  'sc2710.wav',   'sc3ft86.wav',  'sc6ft86.wav',  ...
'sc12g65.wav',  'sc170f.wav',   'sc1ft86.wav',  'sc293b.wav',   'sc3g65.wav',   'sc6g65.wav',   ...
'sc131d.wav',   'sc1710.wav',   'sc1g65.wav',   'sc2a91.wav',   'sc3i03.wav',   'sc7g65.wav',   ...
'sc13g65.wav',  'sc17g65.wav',  'sc1i03.wav',   'sc2d15.wav',   'sc4ft86.wav',  'sc8g65.wav',   ...
'sc14g65.wav',  'sc193a.wav',   'sc1i27.wav',   'sc2ft86.wav',  'sc4g65.wav',   'sc9g65.wav',   ...
'sc151f.wav',   'sc193b.wav',   'sc1v92.wav',   'sc2g65.wav',   'sc4i03.wav',   ...
'sc15g65.wav',  'sc1a91.wav',   'sc1z75.wav',   'sc2i03.wav',   'sc5ft86.wav' };

wb_filenames = { 'wb1480.wav',  'wb1d14.wav',  'wb3d14.wav',  'wb4d14.wav' };

temp = dir('C:\Users\ronwd\OneDrive\Documents\GitHub\SFA_PostCOSYNEAPP\Clutter_Stim_juneon\21-Jul-2020_clutter');

clutter_filenames ={temp(3:102).name};

temp = dir('C:\Users\ronwd\OneDrive\Documents\GitHub\Local_EigenSpecAudioStim\ClutterVocal_Eigendata\NoiseStimuliWav\wave 1\');

whitenoise_filenames = {temp(3:end).name};


%2022-05-30: Adding textures from McDermott work

temp = dir('C:\Users\ronwd\OneDrive\Documents\GitHub\Local_EigenSpecAudioStim\Textures_Eigendata\Texture_Stimuli\textureStimuli_subgroups\Applause');

applause_filesnames = {temp(3:end).name};

temp = dir('C:\Users\ronwd\OneDrive\Documents\GitHub\Local_EigenSpecAudioStim\Textures_Eigendata\Texture_Stimuli\textureStimuli_subgroups\Fire');

fire_filenames = {temp(3:end).name};

temp = dir('C:\Users\ronwd\OneDrive\Documents\GitHub\Local_EigenSpecAudioStim\Textures_Eigendata\Texture_Stimuli\textureStimuli_subgroups\Water_Rain');

water_filenames = {temp(3:end).name};

temp = dir('C:\Users\ronwd\OneDrive\Documents\GitHub\Local_EigenSpecAudioStim\Textures_Eigendata\Texture_Stimuli\textureStimuli_subgroups\RadioStatic');

radio_filenames = {temp(3:end).name};

foldersofwavs = {'C:\Users\ronwd\.spyder-py3\SFA_PostCOSYNEAPP-master\Monkey_Calls\HauserVocalizations\Monkey_Wav_50K\Wav\'...
    ,'C:\Users\ronwd\OneDrive\Documents\GitHub\SFA_PostCOSYNEAPP\Clutter_Stim_juneon\21-Jul-2020_clutter\'...
    ,'C:\Users\ronwd\OneDrive\Documents\GitHub\Local_EigenSpecAudioStim\ClutterVocal_Eigendata\NoiseStimuliWav\wave 1\'...
    ,'C:\Users\ronwd\OneDrive\Documents\GitHub\Local_EigenSpecAudioStim\Textures_Eigendata\Texture_Stimuli\textureStimuli_subgroups\Applause\'...
    ,'C:\Users\ronwd\OneDrive\Documents\GitHub\Local_EigenSpecAudioStim\Textures_Eigendata\Texture_Stimuli\textureStimuli_subgroups\Fire\'...
    ,'C:\Users\ronwd\OneDrive\Documents\GitHub\Local_EigenSpecAudioStim\Textures_Eigendata\Texture_Stimuli\textureStimuli_subgroups\Water_Rain\'...
    ,'C:\Users\ronwd\OneDrive\Documents\GitHub\Local_EigenSpecAudioStim\Textures_Eigendata\Texture_Stimuli\textureStimuli_subgroups\RadioStatic\'};


%2022-05-30: Added toggle for switching between using textures and using OG
%data


%all_filenames = {ag_filenames; co_filenames; cs_filenames; gk_filenames; gt_filenames;...
        %gy_filenames; ha_filenames; sc_filenames; sb_filenames; wb_filenames; clutter_filenames; whitenoise_filenames};
        
%voctypes_labels = {'Aggressive'; 'Coo'; 'Copulation Scream'; 'Gekker'; 'Grunt'; ...
%'Girney'; 'Harmonic Arch'; 'Scream'; 'Shrillbark'; 'Warble'; 'Clutter'; 'White Noise'};
        
if use_textures 
    %since at different sampling rates simply only run the textures
all_filenames={applause_filesnames};%fire_filenames;water_filenames;radio_filenames}; %;clutter_filenames;whitenoise_filenames};
voctypes_labels = {'Applause'};%, 'Fire', 'Water or Rain','Radio Static'};%, 'Clutter', 'White noise'};
nvoctypes = length(all_filenames);
    
else
all_filenames={co_filenames;ha_filenames;gt_filenames;sb_filenames;clutter_filenames;whitenoise_filenames};
voctypes_labels = {'Coo', 'Arch', 'Grunt','Bark', 'Clutter', 'White noise'};
nvoctypes = length(all_filenames);


end    
        
% Loop through files and find longest one
jfile = 1;
for itypes=1:nvoctypes
    nfiles(itypes) = length(all_filenames{itypes});
    
    if use_textures
   
          
        folderused = foldersofwavs{3+itypes};    
            
        
    else
    
    
        if itypes == nvoctypes-1

        folderused = foldersofwavs{2};

        elseif itypes == nvoctypes

        folderused = foldersofwavs{3};

        else
        folderused = foldersofwavs{1};

        end
    
    end
    
    for ifile=1:nfiles(itypes)
        [voc_sound, fs] = audioread(char([folderused...
            all_filenames{itypes}{ifile}]));
        if (fs ~= samprate)
            fprintf(1, 'Error: the sampling rate of wavefile is incorrect\n');
        end
        soundlen(jfile) = length(voc_sound);
        jfile = jfile + 1;
    end
end
totfiles = sum(nfiles);
meanlen = mean(soundlen);
maxlen = max(soundlen);

fprintf(1,'Total number of files is %d\n', totfiles);
fprintf(1,'Mean length of sound is %f (ms)\n', meanlen*1000/samprate);
fprintf(1,'Max length of sound is %f (ms)\n', maxlen*1000/samprate);
soundlen = soundlen.*1000/samprate;
figure(1);
hist(soundlen);
xlabel('Sound Length (ms)');
ylabel('Frequency');

% find the length of the spectrogram and get a time label in ms
maxlenused = min(maxlen, maxlenuser*samprate/1000);
maxlenint = ceil(maxlenused/increment)*increment;
w = hamming(maxlenint);
frameCount = floor((maxlenint-winLength)/increment)+1;
t = 0:frameCount-1;
t = t + (winLength*1000.0)/(2*samprate);
%This is in ms

% Make space for modspectrum
input = zeros(1,maxlenint);
[s, fo, pg] = GaussianSpectrum(input, increment, winLength, samprate);

if corr_spec
    
    s = RWD_quadexpand(s,nolin); %Do quad expansion to get correct sizing for these variables
    
    for itypes=1:nvoctypes
        sumfpow{itypes} = zeros(size(s));
        sumfpow2{itypes} = zeros(size(s));
        fpowcv{itypes} = zeros(size(s));
    end
    
    sumfpowall = zeros(size(s));
    sumfpow2all = zeros(size(s));
    fpowcvall = zeros(size(s));
    sumfpowtyp = zeros(size(s));
    sumfpow2typ = zeros(size(s));
    fpowcvtyp = zeros(size(s));
    
else

    for itypes=1:nvoctypes
        sumfpow{itypes} = zeros(size(s));
        sumfpow2{itypes} = zeros(size(s));
        fpowcv{itypes} = zeros(size(s));
    end
    sumfpowall = zeros(size(s));
    sumfpow2all = zeros(size(s));
    fpowcvall = zeros(size(s));
    sumfpowtyp = zeros(size(s));
    sumfpow2typ = zeros(size(s));
    fpowcvtyp = zeros(size(s));
    
end

%Make space for saving fake spectrograms later
all_sabs = cell(nvoctypes,1);

% Loop through files and make spectrograms
for itypes=1:nvoctypes
    voctypes_labels{itypes}
    ncount(itypes) = 0;
    sabs_temp = zeros(size(s,1), size(s,2), nfiles(itypes));%make empty matrix for each spectrogram for each file for doing the column shuffle later for a type
    for ifile=1:nfiles(itypes)
    
        ifile
        
    if use_textures
   
          
        folderused = foldersofwavs{3+itypes};    
            
        
    else
    
    
        if itypes == nvoctypes-1

        folderused = foldersofwavs{2};

        elseif itypes == nvoctypes

        folderused = foldersofwavs{3};

        else
        folderused = foldersofwavs{1};

        end
    
    end
        % Read sound file
        [voc_sound, fs] = audioread(char([folderused...
            all_filenames{itypes}{ifile}]));
        soundlen = length(voc_sound);

        % Loop through chunks as needed
        voc_sound_beg = 1;
        while voc_sound_beg < soundlen 
            
            % Stuff the temporary input
            input = zeros(1,maxlenint);
            voc_sound_end = min(soundlen, voc_sound_beg+maxlenint-1);            
            chunklen = voc_sound_end - voc_sound_beg + 1;
            nzeros = fix((maxlenint - chunklen)/2);
            input(1+nzeros:nzeros+chunklen) = voc_sound(voc_sound_beg:voc_sound_end);
            
            % window the sound before doing the spectrogram to avoid edge
            % effects
            input = input .* w';
            
            % Get the spectrogram
            [s, fo, pg] = GaussianSpectrum(input, increment, winLength, samprate);
            s = abs(s);
            sabs = log(abs(s)+1.0);
            if corr_spec %If true, quadratically expand spectrogram like in quadratic SFA and then do mod spec.
                
                %2022-09-29 Note: if doing resampling for correlation
                %matrix have to do it here.   Again, not sure exactly what
                %that looks like...and have concerns about memory issues
                %for holding so many large matrices together at the same
                %time which is essential for this type of resampling.
                
                sabs = RWD_quadexpand(sabs,nolin);
                
                              
                
            end
            
            sabs_temp(:,:,ifile) = sabs;
            
            
        % Display the spectrogram (need to edit to work in loop)
%             if (debug_fig) && ifile ==1 %plot the first file of each catagory so we get spectrograms
%                 figure(5+itypes);
%                 imagesc(t/1000, fo/1000, sabs);%ask yale about how toget time in seconds
%                 ylabel('Frequency (kHz)')
%                 xlabel('Time (seconds)')
%                 axis xy;
%                 title(voctypes_labels{itypes})
%                 %pause;    
%             end
            

            %2022-09-29 need to move this to a separate additional loop as
            %we need to make false spectrogram in this loop
%             % calculate the 2D fft
%             fabs = fft2(sabs);
%             fpow = real(fabs.*conj(fabs));
%             sumfpow{itypes} = fpow + sumfpow{itypes};
%             sumfpow2{itypes} = fpow.*fpow + sumfpow2{itypes};
           % We are doing non-overlapping chunks
             voc_sound_beg = voc_sound_beg + maxlenint;
             ncount(itypes) = ncount(itypes) + 1;
        end
    end
    all_sabs{itypes,1} = sabs_temp;
end

%Now decide to column shuffle within a voc_type or across
%voc_types...changing idea from above shuffl across all types. Otherwise
%don't cat and set up for loop

all_sabs = cat(3,all_sabs{:});
%Saving ordering of permutation to be sure things are working as expected.
shuff1 = randperm(length(t));
shuff2 = randperm(size(all_sabs,3));
%Trying different shuffles Update: double shuffle doesn't work as
%anticipated.  Just shuffles the order of the files then shuffle column
%within files instead of across files as desired. Noticed this because
%results do not change if shuff2 is removed.  Setting up for loop to
%shuffle each time ind instead

for t_col = 1:length(t) 

    %all_sabs(:,t_col,:) = all_sabs(:,t_col,randperm(size(all_sabs,3)));
    %Just curious: shuffling spectra too
    all_sabs(:,t_col,:) = all_sabs(randperm(size(all_sabs,1)),t_col,:);
    
end

% for f_row = 1:length(fo) 
%     actually create an artifical signal that has even more slow components
%     than natural vocalizations.  I think because we are creating broad
%     band held tones by mixing this way.
%     all_sabs(f_row,:,:) = all_sabs(f_row,:,randperm(size(all_sabs,3)));
%     Just curious: chaning order of shuffling
%     all_sabs(f_row,:,:) = all_sabs(f_row,randperm(size(all_sabs,2)),:);
%     
% end

%Look at a couple of examples
show_examples = 1;
   % Display the spectrogram
if show_examples ==1 %plot the first file of each catagory so we get spectrograms
    rand_ex = randsample(size(all_sabs,3),5);
    for iii = 1:length(rand_ex)
        figure();
        imagesc(t/1000, fo/1000, all_sabs(:,:,rand_ex(iii)));
        ylabel('Frequency (kHz)')
        xlabel('Time (seconds)')
        axis xy;
        title('Nulls')
    end   
end

%2022-09-29-noon: left off here

disp('Null spectrograms created')

for itypes=1 %since combined all together
   
    ncount(itypes) = 0; %needed for later division to get average mod spec
    for ifile=1:size(all_sabs,3)
        ifile
        
            fabs = fft2(all_sabs(:,:,ifile));
            fpow = real(fabs.*conj(fabs));
            sumfpow{itypes} = fpow + sumfpow{itypes};
            sumfpow2{itypes} = fpow.*fpow + sumfpow2{itypes};
            
            % We are doing non-overlapping chunks
            voc_sound_beg = voc_sound_beg + maxlenint;
            ncount(itypes) = ncount(itypes) + 1
        
        
    end
    
end

disp('Mod specs calculated')

% calculate the meam, std, and coefficient of variation
for itypes=1
    'Null'
    % Perform sums over types to get overal mod spectrum and variance
    sumfpowall = sumfpowall + sumfpow{itypes};
    sumfpow2all = sumfpow2all + sumfpow2{itypes};
    
    % The modespectrum for each type
    sumfpow{itypes} = sumfpow{itypes}./ncount(itypes);

    % The variance and standard deviation of each type
    sumfpow2{itypes} = (sumfpow2{itypes}./(ncount(itypes)-1)) - (ncount(itypes)/(ncount(itypes)-1))*(sumfpow{itypes}.*sumfpow{itypes}); 
    sumfpow2{itypes} = sqrt(sumfpow2{itypes});
    
    % Reorganize for plotting
    sumfpow{itypes} = fftshift(sumfpow{itypes});
    sumfpow2{itypes} = fftshift(sumfpow2{itypes});    
    
%     % Coefficient of variation
%    fpowcv{itypes} = sumfpow2{itypes}./sumfpow{itypes};
end

% % Overall modspectrum, standard deviation and cv
% totcount = sum(ncount);
% sumfpowall = sumfpowall./totcount;
% sumfpow2all = sumfpow2all./(totcount-1) - (totcount/(totcount-1))*(sumfpowall.*sumfpowall); 
% sumfpow2all = sqrt(sumfpow2all); 
% sumfpowall = fftshift(sumfpowall);
% sumfpow2all = fftshift(sumfpow2all);
% fpowcvall = sumfpow2all./sumfpowall;
% 
% % Overall modspectum, standard deviation and cv averaged over types
%  for itypes=1:nvoctypes  
%      sumfpowtyp = sumfpowtyp + sumfpow{itypes};
%      sumfpow2typ = sumfpow2typ + sumfpow{itypes}.*sumfpow{itypes};
%  end
%  sumfpowtyp = sumfpowtyp./nvoctypes;
%  sumfpow2typ = sumfpow2typ./(nvoctypes-1) - (nvoctypes/(nvoctypes-1))*(sumfpowtyp.*sumfpowtyp); 
%  sumfpow2typ = sqrt(sumfpow2typ);
%  fpowcvtyp = sumfpow2typ./sumfpowtyp;
 
% Find labels for x and y axis
% f_step is the separation between frequency bands
fstep = fo(2);
nb = size(sabs,1);%length(fo);
if ( rem(nb,2) == 0 )
    for i=1:nb
        dwf(i)= (i-nb/2)*(1/(fstep*nb));
    end
else
    for i=1:nb
        dwf(i)= (i-(nb+1)/2)*(1/(fstep*nb));
    end   
end

% 1 ms (1000 on the numerator) is the sampling rate
nt = length(t);
if ( rem(nt,2) == 0 )
    for i=1:nt
        dwt(i) = (i-nt/2)*(1000.0/nt);
    end
else
    for i=1:nt
        dwt(i) = (i-(nt+1)/2)*(1000.0/nt);
    end 
end
%% Plotting blocks

if corr_spec
clearvars -except nvoctypes voctypes_labels sumfpow dwt dwf ncol
end
% Plot the modspectrum and contours for all the different voc types in log coordinates
figure;
ncol = 6;
nrow = ceil(nvoctypes/ncol);   
fracpower = [0.5 0.6 0.7 0.8];
fracvalues_mps = zeros(nvoctypes, length(fracpower));

temp_means = nan(nvoctypes,length(dwt)); %save marginal for temporal modulation (i.e. average over spec mod for each temp mod)

disp('Plotting')
for itypes=1:nvoctypes
    
    voctypes_labels{itypes}
    subplot(nrow,ncol, itypes);
    scaled2plot = log(sumfpow{itypes})-min(min(log(sumfpow{itypes})));
    scaled2plot = scaled2plot./max(max(scaled2plot));
    temp_means(itypes,:) = mean(scaled2plot,1);
    % Plot mod spectrum, set max point to 1 so colorbar is always the same
    imagesc(dwt,dwf*1000,scaled2plot);
    
    %shading interp;
    %lighting phong;
    colormap(jet);
    caxis([0,1]);
    axis xy;
    hold on;
    
    plot(dwt,temp_means(itypes,:),'k', 'Markersize', 20)

    hold off;
    
    title(voctypes_labels{itypes});    
    axis([-80 80 0 8]);
    if ( fix(itypes/ncol) == (nrow-1))
        xlabel('{\omega}_{t}(Hz)');
    end
    if ( rem(itypes, ncol) == 1 )
        ylabel('{\omega}_{x}(Cycles/kHz)');
    end
end

figure
for itypes = 1:nvoctypes
    
    subplot(nrow,ncol,itypes)
    plot(dwt,temp_means(itypes,:))
    
    title(voctypes_labels{itypes});  
    
    if ( fix(itypes/ncol) == (nrow-1))
    xlabel('{\omega}_{t}(Hz)');
    end
    axis([-80 80 0 1])
    
end

% % Plot the coefficient of variation for all the different voc types in log coordinates
% figure(3);
% fracpower = [0.5 0.6 0.7 0.8 0.9];
% 
% for itypes=1:nvoctypes
%     subplot(nrow,ncol, itypes);
%     tobezero = find(sumfpow{itypes} < fracvalues_mps(itypes,end));
%     fpowcv{itypes}(tobezero) = 0.0;
%     imagesc(dwt,dwf*1000,log(fpowcv{itypes}));
%     %shading interp;
%     %lighting phong;
%     colormap(jet);
%     axis xy;
%     hold on;
%     
%     fracvalues = calc_contour_values(fpowcv{itypes}, fracpower);
%     [C h] = contour(dwt,dwf*1000,fpowcv{itypes},fracvalues,'k-');
%     hold off;
%     axis([-80 80 0 8]); 
%     
%     title(voctypes_labels{itypes}); 
%     if ( fix(itypes/ncol) == (nrow-1))
%         xlabel('{\omega}_{t}(Hz)');
%     end
%     if ( rem(itypes, ncol) == 1 )
%         ylabel('{\omega}_{x}(Cycles/kHz)');
%     end
%     
% end

% 
% % Plot the modspectrum and contours for all voc types analyzed together
% figure(4);
% fracpower = [0.5 0.6 0.7 0.8 0.9];
% imagesc(dwt,dwf*1000,log(sumfpowall));
% %shading interp;
% %lighting phong;
% colormap(jet);
% axis xy;
% hold on;
% fracvalues_mpsall = calc_contour_values(sumfpowall, fracpower);
% [C h] = contour(dwt,dwf*1000,sumfpowall,fracvalues_mpsall,'k-');
% hold off;
% title('All vocs analyzed together');    
% axis([-80 80 0 8]);
% xlabel('{\omega}_{t}(Hz)');
% ylabel('{\omega}_{x}(Cycles/kHz)');
% 
% % Plot the coefficient of variation for all vocs analyzed together
% figure(5);
% fracpower = [0.5 0.6 0.7 0.8 0.9];
% tobezero = find(sumfpowall < fracvalues_mpsall(end));
% fpowcvall(tobezero) = 0.0;
% imagesc(dwt,dwf*1000,log(fpowcvall));
% %shading interp;
% %lighting phong;
% colormap(jet);
% axis xy;
% hold on;
% fracvalues = calc_contour_values(fpowcvall, fracpower);
% [C h] = contour(dwt,dwf*1000,fpowcvall,fracvalues,'k-');
% hold off;
% axis([-80 80 0 8]); 
% title('All vocs analyzed together'); 
% xlabel('{\omega}_{t}(Hz)');
% ylabel('{\omega}_{x}(Cycles/kHz)');
% 
% % Plot the modspectrum and contours for all averaged across voc types
% figure(6);
% fracpower = [0.5 0.6 0.7 0.8 0.9];
% imagesc(dwt,dwf*1000,log(sumfpowtyp));
% %shading interp;
% %lighting phong;
% colormap(jet);
% axis xy;
% hold on;
% fracvalues_mpstyp = calc_contour_values(sumfpowtyp, fracpower);
% [C h] = contour(dwt,dwf*1000,sumfpowtyp,fracvalues_mpstyp,'k-');
% hold off;
% title('All vocs averaged across types');    
% axis([-80 80 0 8]);
% xlabel('{\omega}_{t}(Hz)');
% ylabel('{\omega}_{x}(Cycles/kHz)');
% 
% % Plot the coefficient of variation for all averaged across voc types
% figure(7);
% fracpower = [0.5 0.6 0.7 0.8 0.9];
% tobezero = find(sumfpowtyp < fracvalues_mpstyp(end));
% fpowcvtyp(tobezero) = 0.0;
% imagesc(dwt,dwf*1000,log(fpowcvtyp));
% %shading interp;
% %lighting phong;
% colormap(jet);
% axis xy;
% hold on;
% fracvalues = calc_contour_values(fpowcvtyp, fracpower);
% [C h] = contour(dwt,dwf*1000,fpowcvtyp,fracvalues,'k-');
% hold off;
% axis([-80 80 0 8]); 
% title('All vocs averaged across types');   
% xlabel('{\omega}_{t}(Hz)');
% ylabel('{\omega}_{x}(Cycles/kHz)');


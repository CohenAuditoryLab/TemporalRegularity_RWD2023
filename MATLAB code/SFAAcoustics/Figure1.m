close all
clear all

% Some parameters
samprate = 50000;                          % 50k sampling rate
fband = 32;                                % Frequency band in Hz
twindow = 1000*6/(fband*2.0*pi);           % Window length in ms - 6 times the standard dev of the gaussian window
winLength = fix(twindow*samprate/1000.0);  % Window length in number of points
increment = fix(0.001*samprate);            % Sampling rate of spectrogram in number of points - set at 1 kHz
debug_fig = 0;                              % Set to 1 to see spectrograms
maxlenuser = 1000;                          % Maximun length of window in ms for estimating modspectrum.  Vocalizations
% that are longer will be divided into chunks.  Vocalizations that are
% shorter will be padded with zero
figure(2); set(gcf,'position',[367 146 861 620])
tiledlayout(2,4,'TileSpacing','compact')

path(path,'/Users/Yale/Dropbox/HauserVocalizations')
 
         

% % List of sound files for the different vocalizations
% ag_filenames = {'ag10944.wav',  'ag16944.wav',  'ag1e55.wav',   'ag1o49.wav',   'ag3575.wav',   'ag5944.wav',  ... 
% 'ag11944.wav',  'ag1710.wav',   'ag1f24.wav',   'ag2575.wav',   'ag393b.wav',   'ag693b.wav',   ...
% 'ag12944.wav',  'ag1713.wav',   'ag1g62.wav',   'ag282b.wav',   'ag3944.wav',   'ag6944.wav',   ...
% 'ag13944.wav',  'ag1787.wav',   'ag1h79.wav',   'ag293b.wav',   'ag3k57.wav',   'ag793b.wav',   ...
% 'ag141c.wav',   'ag182b.wav',   'ag1i04.wav',   'ag2944.wav',   'ag3k89.wav',   'ag7944.wav',   ...
% 'ag1470.wav',   'ag182c.wav',   'ag1j24.wav',   'ag2978.wav',   'ag4575.wav',   'ag8944.wav',   ...
% 'ag14944.wav',  'ag193b.wav',   'ag1k57.wav',   'ag2k57.wav',   'ag493b.wav',   'ag9944.wav',   ...
% 'ag1535.wav',   'ag1944.wav',   'ag1k89.wav',   'ag2k89.wav',   'ag4944.wav',   ...
% 'ag1575.wav',   'ag1978.wav',   'ag1n37.wav',   'ag2o27.wav',   'ag4k57.wav',   ...
% 'ag15944.wav',  'ag1c75.wav',   'ag1o27.wav',   'ag2o49.wav',   'ag593b.wav' };

co_filenames = { 'CO1A94.wav',   'co1491.wav',   'co1671.wav',   'co1933.wav',   'co2d14.wav',   'co4845.wav',   ...
'CO1A96.wav',   'co151f.wav',     'co1o13.wav',   'co2f24.wav',   'co4f24.wav',   ...
'co1405.wav',   'co159b.wav',   'co168f.wav',   'co1z67.wav',   'co3845.wav',   'co5f24.wav',   ...
'co1480.wav',   'co1660.wav',   'co176e.wav',   'co2845.wav',   'co3f24.wav' };
%missing 'co16845.wav',

% cs_filenames = { 'cs10g62.wav',  'cs2e54.wav',   'cs3g62.wav',   'cs4o49.wav',   'cs6g62.wav',   'cs9a59.wav',  ... 
% 'cs11a59.wav',  'cs2g62.wav',      'cs5a59.wav',   'cs6o49.wav',   'cs9g62.wav',   ...
% 'cs1t53.wav',   'cs2h46.wav',   'cs3o49.wav',   'cs5e54.wav',   'cs7a59.wav',   ...
% 'cs2a59.wav',   'cs2o49.wav',   'cs4a59.wav',   'cs5g62.wav',   'cs7g62.wav',   ...
% 'cs2a74.wav',   'cs2t53.wav',   'cs4e54.wav',   'cs5o49.wav',   'cs8a59.wav',   ...
% 'cs2c99.wav',      'cs4g62.wav',   'cs6a59.wav',   'cs8g62.wav' };
% %missing 'cs3hh6.wav','cs3354.wav',
% 
% gk_filenames = { 'gk151f.wav',   'gk1f777.wav',  'gk1fn37.wav',  'gk2fn37.wav',  'gk4fx33.wav',  ...
% 'gk160f.wav',   'gk1fe04.wav',  'gk1fx33.wav',  'gk2fx33.wav',  'gk5fx33.wav',  ...
% 'gk169f.wav',   'gk1fe55.wav',  'gk251f.wav',   'gk3f11c.wav',  'gk6fx33.wav',  ...
% 'gk173f.wav',   'gk1fk80.wav',  'gk2f11c.wav',  'gk3fn37.wav',  'gk7fx33.wav',  ...
% 'gk1f39a.wav',  'gk1fllc.wav',  'gk2fk80.wav',  'gk3fx33.wav' };

gt_filenames = { 'gt10a94.wav',  'gt1a94.wav',   'gt1k3.wav',    'gt2i04.wav',   'gt3k3.wav',    'gt7a94.wav',   ...
'gt117d.wav',   'gt1a96.wav',   'gt1n37.wav',   'gt2k3.wav',    'gt4845.wav',   'gt9a94.wav',   ...
'gt176e.wav',   'gt1b10.wav',   'gt217d.wav',   'gt382c.wav',   'gt4a94.wav',   ...
'gt182c.wav',   'gt1e55.wav',   'gt282c.wav',   'gt3845.wav',   'gt5845.wav',   ...
'gt1842.wav',   'gt1f29.wav',   'gt2845.wav',   'gt3a94.wav',   'gt5a94.wav',   ...
'gt1845.wav',   'gt1g32.wav',   'gt2a96.wav',   'gt3a96.wav',   'gt6845.wav',   ...
'gt190b.wav',   'gt1i04.wav',   'gt2f29.wav',   'gt3g32.wav',   'gt6a94.wav',   ...
'gt1912.wav',   'gt1i10.wav',   'gt2g32.wav',   'gt3i04.wav',   'gt7845.wav'};

% gy_filenames = { 'gy114d.wav',  'gy1a94.wav',  'gy1z75.wav',  'gy3275.wav',  'gy5275.wav',  ...
% 'gy117d.wav',  'gy1g35.wav',  'gy217d.wav',  'gy3845.wav',  'gy5845.wav',  ...
% 'gy1275.wav',  'gy1k80.wav',  'gy2275.wav',  'gy3g35.wav',  'gy6275.wav',  ...
% 'gy1845.wav',  'gy1l45.wav',  'gy2845.wav',  'gy4257.wav',  'gy7275.wav',  ...
% 'gy1944.wav',  'gy1l63.wav',  'gy2g35.wav',  'gy4845.wav',  'gya94.wav' };
% 
ha_filenames = { 'ha1480.wav',  'ha1h38.wav',  'ha1n41.wav',  'ha2845.wav',  'ha3d14.wav',  ...
'ha1842.wav',  'ha1i06.wav',  'ha1unk.wav',  'ha2d14.wav',  'ha3unk.wav',  ...
'ha1845.wav',  'ha1i77.wav',  'ha1ya.wav',   'ha2unk.wav',  'ha4d14.wav',  ...
'ha1g61.wav',  'ha1k83.wav',  'ha2480.wav',  'ha2ya.wav'};

sb_filenames = { 'sb1548.wav',  'sb1h85.wav',  'sb2837.wav',  'sb3837.wav',  'sb5837.wav',  ...
'sb1832.wav',  'sb1i05.wav',  'sb2933.wav',  'sb3933.wav',  'sb6837.wav',  ...
'sb1837.wav',  'sb1l68.wav',  'sb2h85.wav',  'sb3h85.wav',  'sb7837.wav',  ...
'sb1933.wav',  'sb2832.wav',  'sb2l68.wav',  'sb4837.wav',  'sb8837.wav', };
% 
% sc_filenames = { 'sc10g65.wav',  'sc169f.wav',   'sc1d15.wav',   'sc251f.wav',   'sc3d15.wav',   'sc5g65.wav',  ... 
% 'sc11g65.wav',  'sc16g65.wav',  'sc1fk89.wav',  'sc2710.wav',   'sc3ft86.wav',  'sc6ft86.wav',  ...
% 'sc12g65.wav',  'sc170f.wav',   'sc1ft86.wav',  'sc293b.wav',   'sc3g65.wav',   'sc6g65.wav',   ...
% 'sc131d.wav',   'sc1710.wav',   'sc1g65.wav',   'sc2a91.wav',   'sc3i03.wav',   'sc7g65.wav',   ...
% 'sc13g65.wav',  'sc17g65.wav',  'sc1i03.wav',   'sc2d15.wav',   'sc4ft86.wav',  'sc8g65.wav',   ...
% 'sc14g65.wav',  'sc193a.wav',   'sc1i27.wav',   'sc2ft86.wav',  'sc4g65.wav',   'sc9g65.wav',   ...
% 'sc151f.wav',   'sc193b.wav',   'sc1v92.wav',   'sc2g65.wav',   'sc4i03.wav',   ...
% 'sc15g65.wav',  'sc1a91.wav',   'sc1z75.wav',   'sc2i03.wav',   'sc5ft86.wav' };
% 
% wb_filenames = { 'wb1480.wav',  'wb1d14.wav',  'wb3d14.wav',  'wb4d14.wav' };

all_filenames = {co_filenames; gt_filenames;...
         ha_filenames; sb_filenames};

nvoctypes = length(all_filenames);
voctypes_labels = {'Coo'; 'Grunt'; 'Harmonic Arch'; 'Shrillbark'};

% Loop through files and find longest one
jfile = 1;
for itypes=1:nvoctypes
    nfiles(itypes) = length(all_filenames{itypes});
    for ifile=1:nfiles(itypes)
        [voc_sound, fs] = audioread(char(all_filenames{itypes}{ifile}));
        if (fs ~= samprate)
           % fprintf(1, ['Error: the sampling rate (' num2str(fs) ') of wavefile is incorrect...interpolating\n']);
            t=[1:1:length(voc_sound)];
            t2=[1:fs/samprate:length(voc_sound)];
            voc_sound=interp1(t,voc_sound,t2);
        end
        soundlen(jfile) = length(voc_sound);
        jfile = jfile + 1;
    end
end
totfiles = sum(nfiles);
meanlen = mean(soundlen);
maxlen = max(soundlen);

% fprintf(1,'Total number of files is %d\n', totfiles);
% fprintf(1,'Mean length of sound is %f (ms)\n', meanlen*1000/samprate);
% fprintf(1,'Max length of sound is %f (ms)\n', maxlen*1000/samprate);
soundlen = soundlen.*1000/samprate;
% figure(1);
% hist(soundlen);
% xlabel('Sound Length (ms)');
% ylabel('Frequency');

% find the length of the spectrogram and get a time label in ms
maxlenused = min(maxlen, maxlenuser*samprate/1000);
maxlenint = ceil(maxlenused/increment)*increment;
w = hamming(maxlenint);
frameCount = floor((maxlenint-winLength)/increment)+1;
t = 0:frameCount-1;
t = t + (winLength*1000.0)/(2*samprate);


% Make space for modspectrum
input = zeros(1,maxlenint);
[s, fo, pg] = GaussianSpectrum(input, increment, winLength, samprate);
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

msmin=[];msmax=[];dtwmin=[];dtwmax=[];
% Loop through files and make spectrograms
for itypes=1:nvoctypes
    ncount(itypes) = 0;
   for ifile=1:nfiles(itypes)
     IndividualModSpec{ifile} = zeros(size(s));
   end
  
    for ifile=1:nfiles(itypes) %calculate each voc/type
        
        % Read sound file
        [voc_sound, fs] = audioread(char(all_filenames{itypes}{ifile}));
        if (fs ~= samprate)
            %fprintf(1, ['Error: the sampling rate (' num2str(fs) ') of wavefile is incorrect...interpolating\n']);
            t=[1:1:length(voc_sound)];
            t2=[1:fs/samprate:length(voc_sound)];
            voc_sound=interp1(t,voc_sound,t2)';
        end
        soundlen = length(voc_sound);
        

        % Loop through chunks as needed
        voc_sound_beg = 1;
        while voc_sound_beg < soundlen %each voc modspec calc
            
            % Stuff the temporary input
            input = zeros(1,maxlenint);
            voc_sound_end = min(soundlen, voc_sound_beg+maxlenint-1);            
            chunklen = voc_sound_end - voc_sound_beg + 1;
            nzeros = fix((maxlenint - chunklen)/2);
            input(1+nzeros:nzeros+chunklen) = voc_sound(voc_sound_beg:voc_sound_end);
            
            % window the sound before doing the spectrogram to avoid edge
            % effects
            input = input .* w';
            IndividualVocFiles(ifile,:) = input';
            
            % Get the spectrogram
            [s, fo, pg] = GaussianSpectrum(input, increment, winLength, samprate);
            sabs = log(abs(s)+1.0);
            
            
            % plot the spectrogram
            if ifile == 1
                nexttile;
                imagesc(t/samprate, fo, sabs);
                axis xy; axis square;
                colormap(jet);
                %title(voctypes_labels{itypes});
                set(gca,'fontname','arial','fontsize',14)
                if itypes==1
                    ylabel('Frequency (kHz)');
                    xlabel('Time (s)');
                    set(gca,'yticklabels',[{'0'},{''},{'1'},{''},{'2'},{''}])
                else
                     set(gca,'yticklabels',[{''},{''},{''},{''},{''},{''}])
                end       
            end
            
            switch itypes
                case 1
                    xticks([0,0.25 0.5])
                case 2
                    xticks([0,0.05 0.1])
                case 3
                    xticks([0,0.35 0.7])
                case 4
                    xticks([0,0.098 0.195])
            end
            
            % calculate the 2D fft
            fabs = fft2(sabs);
            fpow = real(fabs.*conj(fabs));
            sumfpow{itypes} = fpow + sumfpow{itypes};
            sumfpow2{itypes} = fpow.*fpow + sumfpow2{itypes};
            
            %calculate and store for each vocalization as f() type
            IndividualModSpec{ifile} = fpow + IndividualModSpec{ifile};
            
            % We are doing non-overlapping chunks
            voc_sound_beg = voc_sound_beg + maxlenint;
            ncount(itypes) = ncount(itypes) + 1;
        end %while voc_sound_beg < soundlen
    end %for ifile=1:nfiles(itypes)
    ncol = 5;
    nrow = ceil(nvoctypes/ncol);
    CorrMatrix=[];
    dtwCorrMatrix=[];
    %calculate correlation matrix for each voc
    
    for xx=1:nfiles(itypes)
      for yy=1:nfiles(itypes)   
         foo=corrcoef(log(IndividualModSpec{xx}),log(IndividualModSpec{yy}));
%          CorrMatrix(xx,yy)=foo(1,2);
%          dtwfoo=dtw(IndividualVocFiles(xx,:),IndividualVocFiles(yy,:));
%          dtwCorrMatrix(xx,yy)=dtwfoo;
      end
    end
%     figure(100)
%     subplot(nrow,ncol, itypes);
%     imagesc(flipud(CorrMatrix));
%     title(voctypes_labels{itypes});
%     axis xy
%     colormap(jet)
%     colorbar
%     hold on; 
%     
%     figure(101)
%     subplot(nrow,ncol, itypes);
%     imagesc(flipud(dtwCorrMatrix));
%     
%     title(voctypes_labels{itypes});
%     axis xy
%     colormap(jet)
%     colorbar
%     hold on;   
    
    %collect min and max values for each graph so later can put
    %on same scale
    msmin=[msmin min(min(CorrMatrix))];
    rCorrMatrix=round(CorrMatrix,2,'significant');
    foo=find(rCorrMatrix~=1);
    msmax=[msmax max(rCorrMatrix(foo))];
    dtwmin=[dtwmin min(min(dtwCorrMatrix))];
    dtwmax=[dtwmax max(max(dtwCorrMatrix))];
end %for itypes=1:nvoctypes

% %put graphs on same scale
% ncol = 4;
% nrow = ceil(nvoctypes/ncol);
% for itypes=1:nvoctypes
%   figure(100)    
%   subplot(nrow,ncol, itypes);
%   caxis([min(msmin) max(msmax)]);
%   figure(101)
%   subplot(nrow,ncol, itypes);
%   caxis([min(dtwmin) max(dtwmax)]);
% end


% calculate the mean, std, and coefficient of variation
for itypes=1:nvoctypes
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
    
    % Coefficient of variation
    fpowcv{itypes} = sumfpow2{itypes}./sumfpow{itypes};
end

% Overall modspectrum, standard deviation and cv
totcount = sum(ncount);
sumfpowall = sumfpowall./totcount;
sumfpow2all = sumfpow2all./(totcount-1) - (totcount/(totcount-1))*(sumfpowall.*sumfpowall); 
sumfpow2all = sqrt(sumfpow2all); 
sumfpowall = fftshift(sumfpowall);
sumfpow2all = fftshift(sumfpow2all);
fpowcvall = sumfpow2all./sumfpowall;

% Overall modspectum, standard deviation and cv averaged over types
 for itypes=1:nvoctypes  
     sumfpowtyp = sumfpowtyp + sumfpow{itypes};
     sumfpow2typ = sumfpow2typ + sumfpow{itypes}.*sumfpow{itypes};
 end
 sumfpowtyp = sumfpowtyp./nvoctypes;
 sumfpow2typ = sumfpow2typ./(nvoctypes-1) - (nvoctypes/(nvoctypes-1))*(sumfpowtyp.*sumfpowtyp); 
 sumfpow2typ = sqrt(sumfpow2typ);
 fpowcvtyp = sumfpow2typ./sumfpowtyp;
 
% Find labels for x and y axis
% f_step is the separation between frequency bands
fstep = fo(2);
nb = length(fo);
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

% Plot the modspectrum and contours for all the different voc types in log coordinates
ncol = 4;
%nrow = ceil(nvoctypes/ncol);   
nrow=2;
fracpower = [0.5 0.6 0.7 0.8];
fracvalues_mps = zeros(nvoctypes, length(fracpower));

for itypes=1:nvoctypes
    nexttile
    
    % Plot mod spectrum
    imagesc(dwt,dwf*1000,log(sumfpow{itypes}));
    %shading interp;
    %lighting phong;
    colormap(jet);
    axis xy;
    axis square
    
    % Calculate and plot contour lines
    fracvalues_mps(itypes,:) = calc_contour_values(sumfpow{itypes}, fracpower);
    %[C h] = contour(dwt,dwf*1000,sumfpow{itypes},fracvalues_mps,'k-');
    
    axis([-55 55 0 8]);
    set(gca,'fontname','arial','fontsize',14)
    if ( rem(itypes, ncol) == 1 )
        ylabel('{\omega}_{x}(Cycles/kHz)');
        xlabel('{\omega}_{t}(Hz)');
        set(gca,'yticklabels',[{'0'},{''},{'4'},{''},{'8'}])
    else
        set(gca,'yticklabels',[{''},{''},{''},{''},{''},{''}])
    end
end


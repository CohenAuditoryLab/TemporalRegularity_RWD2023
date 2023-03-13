% List of sound files to be filtered
nsongs = 10;
for i=1:nsongs
filenames{i} = sprintf('flatrip%d', i);
end

file_dirs = {'mlnoise0_5_0_1/', 'mlnoise0_5_1_2/', 'mlnoise0_5_2_3/', 'mlnoise0_5_3_4/', ...
    'mlnoise5_10_0_1/', 'mlnoise5_10_1_2/', 'mlnoise5_10_2_3/', 'mlnoise5_10_3_4/', ...
    'mlnoise10_15_0_1/', 'mlnoise10_15_1_2/', 'mlnoise10_15_2_3/', 'mlnoise10_15_3_4/', ...
    'mlnoise15_20_0_1/', 'mlnoise15_20_1_2/', 'mlnoise15_20_2_3/', 'mlnoise15_20_3_4/'}; 

voctypes_labels = {'W_t 0-5Hz W_x 0-1 cyc/kHz'; 'W_t 0-5Hz W_x 1-2 cyc/kHz'; 'W_t 0-5Hz W_x 2-3 cyc/kHz'; 'W_t 0-5Hz W_x 3-4 cyc/kHz'; ...
    'W_t 5-10 Hz W_x 0-1 cyc/kHz'; 'W_t 5-10 Hz W_x 1-2 cyc/kHz'; 'W_t 5-10 Hz W_x 2-3 cyc/kHz'; 'W_t 5-10 Hz W_x 3-4 cyc/kHz'; ...
    'W_t 10-15Hz W_x 0-1 cyc/kHz'; 'W_t 10-15Hz W_x 1-2 cyc/kHz'; 'W_t 10-15Hz W_x 2-3 cyc/kHz'; 'W_t 10-15Hz W_x 3-4 cyc/kHz'; ...
    'W_t 15-20-5Hz W_x 0-1 cyc/kHz'; 'W_t 15-20Hz W_x 1-2 cyc/kHz'; 'W_t 15-20Hz W_x 2-3 cyc/kHz'; 'W_t 15-20Hz W_x 3-4 cyc/kHz'};

nvoctypes = length(file_dirs);

% Load transfer function of delivery syste.
load CalCurve;
f = CalCurve(:,1);                   % Frequency for transfer function
tspeaker = CalCurve(:,2)+20;            % Transfer function of speaker system in DB.  Twenty dB are addded so that things don't blow up.

% Loop through files to find rms and max for each sound
jf = 1;
for itypes=1:nvoctypes
    nfiles(itypes) = nsongs;
    for ifile=1:nfiles(itypes)
        fullpath = sprintf('%s%s.wav', file_dirs{itypes}, filenames{ifile});
        [voc_sound, fs, nbits] = wavread(fullpath);
        npts = length(voc_sound);
        rmsval(jf) = norm(voc_sound)/sqrt(npts);
        absval(jf) = max(abs(voc_sound));
        jf = jf+1;
    end
end

maxabs = max(absval);
maxrms = max(rmsval);
corrfact = min(rmsval./(maxabs*maxrms));

% Write our power normalized files that span -1 to 1
jf = 1;
for itypes=1:nvoctypes
    nfiles(itypes) = nsongs;
    for ifile=1:nfiles(itypes)
        fullpath = sprintf('%s%s.wav', file_dirs{itypes}, filenames{ifile});
        [voc_sound, fs, nbits] = wavread(fullpath);
        voc_sound = voc_sound.*((maxrms*corrfact)/rmsval(jf));
        pause;
        newpath = sprintf('%s%s_norm.wav', file_dirs{itypes}, filenames{ifile});
        nclipped = length(find(abs(voc_sound) > 1.0));
        if (nclipped > 0 ) 
            fprintf(1, 'Warning %d points are clipped in file %s\n', nclipped, newpath);
        end
        wavwrite( voc_sound, fs, nbits, newpath);
        jf = jf +1;
    end
end

% Loop through files and apply the inverse filter
for itypes=1:nvoctypes
    nfiles(itypes) = nsongs;
    for ifile=1:nfiles(itypes)
        fullpath = sprintf('%s%s_norm.wav', file_dirs{itypes}, filenames{ifile});
        [voc_sound, fs, nbits] = wavread(fullpath);
        npts = length(voc_sound);
        fprintf(1, 'rms before filtering = %g\n', norm(voc_sound)/sqrt(npts));
        % Take Fourier Transform of input file
        fvoc = fft(voc_sound);
        fval_voc = (0:fix(npts/2))*(fs/npts);
        
        t_voc = interp1(f, tspeaker, fval_voc);
        
        corr_factor = 10.^(-t_voc./20);
        
        for i=1:fix(npts/2)
            if (fval_voc(i) > 130 && fval_voc(i) < 10000.0 )    % Filtering is done between 130 and 10000 Hz
                fvoc(i) = fvoc(i)*corr_factor(i);
                if ( i > 1 )
                    fvoc(npts-i+2) = fvoc(npts-i+2)*corr_factor(i);
                end
            end
        end
        
        voc_sound_filtered = real(ifft(fvoc));
        fprintf(1, 'rms after filtering = %g\n', norm(voc_sound_filtered)/sqrt(npts));
        plot(voc_sound_filtered);
        pause;
        newpath = sprintf('%s%s_filt.wav', file_dirs{itypes}, filenames{ifile});
        nclipped = length(find(abs(voc_sound_filtered) > 1.0));
        if (nclipped > 0 ) 
            fprintf(1, 'Warning %d points are clipped in file %s\n', nclipped, newpath);
        end
        wavwrite( voc_sound_filtered, fs, nbits, newpath);
    end
end
        

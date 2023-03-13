function ReturnValue = MakeModSpec(voc,debug)


%voc is an ascii file of sampled data

% Some parameters
samprate = 50000;                          % 50k sampling rate
fband = 32;                                % Frequency band in Hz
twindow = 1000*6/(fband*2.0*pi);           % Window length in ms - 6 times the standard dev of the gaussian window
winLength = fix(twindow*samprate/1000.0);  % Window length in number of points
increment = fix(0.001*samprate);            % Sampling rate of spectrogram in number of points - set at 1 kHz
debug_fig = 0;                              % Set to 1 to see spectrograms
maxlenuser = 1000;                          % Maximun length of window in ms for estimating modspectrum.  Vocalizations


soundlen = length(voc);
totfiles = 1;
meanlen = length(voc);
maxlen = length(voc);

% fprintf(1,'Total number of files is %d\n', totfiles);
% fprintf(1,'Mean length of sound is %f (ms)\n', meanlen*1000/samprate);
% fprintf(1,'Max length of sound is %f (ms)\n', maxlen*1000/samprate);
soundlen = soundlen.*1000/samprate;



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

sumfpow = zeros(size(s));
sumfpow2 = zeros(size(s));
fpowcv = zeros(size(s));

sumfpowall = zeros(size(s));
sumfpow2all = zeros(size(s));
fpowcvall = zeros(size(s));
sumfpowtyp = zeros(size(s));
sumfpow2typ = zeros(size(s));
fpowcvtyp = zeros(size(s));


%since doing one sound at a time, no edge effect issues
% % window the sound before doing the spectrogram to avoid edge
% % effects
% input = input .* w';

input = voc;

% Get the spectrogram
[s, fo, pg] = GaussianSpectrum(input, increment, winLength, samprate);
sabs = log(abs(s)+1.0);

 % Display the spectrogram
 if (debug)
   figure(100);
   imagesc(t, fo, sabs);
   axis xy;
   pause;    
 end
 % calculate the 2D fft
 fabs = fft2(sabs);
 fpow = real(fabs.*conj(fabs));
 sumfpow = fpow;


 
 % Reorganize for plotting
 sumfpow = fftshift(sumfpow);
 ReturnValue = log(sumfpow);
  
if(debug) 
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
figure(200);

 
fracpower = [0.5 0.6 0.7 0.8];
fracvalues_mps = zeros(1, length(fracpower));


% Plot mod spectrum
 imagesc(dwt,dwf*1000,log(sumfpow));
 %shading interp;
 %lighting phong;
 colormap(jet);
 axis xy;
 hold on;
    
 % Calculate and plot contour lines
 fracvalues_mps = calc_contour_values(sumfpow, fracpower);
 [C h] = contour(dwt,dwf*1000,sumfpow,fracvalues_mps,'k-');
 hold off;
    
      
 axis([-80 80 0 8]);
 xlabel('{\omega}_{t}(Hz)');
 ylabel('{\omega}_{x}(Cycles/kHz)');
   
end
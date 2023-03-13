function [y,d] = ex5_logmodsub
% [y,d] = ex5_logmodsub

% ------- ex5_logmodsub.m ----------------------------------
% Marios Athineos, marios@ee.columbia.edu
% http://www.ee.columbia.edu/~marios/
% Copyright (c) 2003 by Columbia University.
% All rights reserved.
% ----------------------------------------------------------

% Load the file to be processed
[x,sr] = wavread('neneh32.wav');

% Define the hop lengths in milliseconds
fhop1ms = 10;
fhop2ms = 500;

% Define the normalized frequency range in which
% we should zero out the modulation coefs
mlim = [0.05,1];

% Convert from ms to samples
fhop1 = round(fhop1ms*sr/1000);
fhop2 = round(fhop2ms*sr/(fhop1*1000));

% Do the base transform
[fxdm,fxdp,fpad1] = basetran(x,fhop1);

% Take the log for 'demultiplication'
fxdm = log(fxdm);

% Get the modulation spectrum
[fXc,fpad] = modspec(fxdm,fhop2);

% For the modulation frequencies we can zero out
fXc = modsub(fXc,mlim,'zero');

% Invert the modulation spectrum
fxdm = invmodspec(fXc,fpad);

% Invert the log
fxdm = exp(fxdm);

% Take the inverse base transform to reconstruct
y = invbasetran(fxdm,fxdp,fpad1);

% Make same length for taking the difference etc.
[x,y] = pad2longest(x,y);

% So the difference is just
d = y - x;

% Plot ...
subplot(311); specgram(x,1024,sr); c = caxis;
subplot(312); specgram(y,1024,sr); caxis(c);
subplot(313); specgram(d,1024,sr); caxis(c);
print('-djpeg','neneh32_logmodsub.jpg');
% ... and save wavs
wavwrite(y,sr,16,'neneh32_logmodsub_decode.wav');
% Normalize the error so it can be audible (else its very quiet)
wavwrite(d/max(abs(d)),sr,16,'neneh32_logmodsub_error.wav');
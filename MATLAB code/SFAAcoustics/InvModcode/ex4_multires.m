function [y,d] = ex4_multires
% [y,d] = ex4_multires

% ------- ex4_multires.m -----------------------------------
% Marios Athineos, marios@ee.columbia.edu
% http://www.ee.columbia.edu/~marios/
% Copyright (c) 2003 by Columbia University.
% All rights reserved.
% ----------------------------------------------------------

% Load the file to be processed
[x,sr] = wavread('neneh32.wav');

% Define the hop length in milliseconds
fhop1ms = 10;
% Convert from ms to samples
fhop1   = round(fhop1ms*sr/1000);

% Now define the linear multiresolution fhop2
% 500ms on low bins to 50ms on high bins
fhop2ms = linspace(500,50,fhop1);
fhop2   = round(fhop2ms*sr/(fhop1*1000));

% Define the normalized frequency range in which
% we should zero out the modulation coefs
mlim = [0.5,1];

% Do the base transform
[fxdm,fxdp,fpad1] = basetran(x,fhop1);

% Get the modulation spectrum
[fXc,fpad] = modspec(fxdm,fhop2);

% For the modulation frequencies we can zero out
fXc = modsub(fXc,mlim,'zero');

% Invert the modulation spectrum
fxdm = invmodspec(fXc,fpad);

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
print('-djpeg','neneh32_multires.jpg');
% ... and save wavs
wavwrite(y,sr,16,'neneh32_multires_decode.wav');
% Normalize the error so it can be audible (else its very quiet)
wavwrite(d/max(abs(d)),sr,16,'neneh32_multires_error.wav');
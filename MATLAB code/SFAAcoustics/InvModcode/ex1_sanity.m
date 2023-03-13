function [y,d] = ex1_sanity
% [y,d] = ex1_sanity

% ------- ex1_sanity.m -------------------------------------
% Marios Athineos, marios@ee.columbia.edu
% http://www.ee.columbia.edu/~marios/
% Copyright (c) 2003 by Columbia University.
% All rights reserved.
% ----------------------------------------------------------

% Load the file to be processed
[x,sr] = audioread('neneh32.wav');

% Define the hop lengths in milliseconds
fhop1ms = 10;
fhop2ms = 250;

% Convert from ms to samples
fhop1 = round(fhop1ms*sr/1000);
fhop2 = round(fhop2ms*sr/(fhop1*1000));

% Do the base transform
[fxdm,fxdp,fpad1] = basetran(x,fhop1);

% Get the modulation spectrum
[fXc,fpad] = modspec(fxdm,fhop2);

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
print('-djpeg','neneh32_sanity.jpg');
% ... and save wavs
filename1 = ['neneh32_sanity_decode.wav'];
audiowrite(filename1,y,sr,'BitsPerSample',16);
% Normalize the error so it can be audible (else its very quiet)
filename2 = ['neneh32_sanity_error.wav'];
audiowrite(filename2,d/max(abs(d)),sr,'BitsPerSample',16);
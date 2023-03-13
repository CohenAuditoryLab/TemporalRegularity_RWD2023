function [y,d] = modcodec(x,sr,mlim,plim,fhop1ms,fhop2ms)
% MODCODEC Full modulation spectrum-based codec
%   [y,d] = modcodec(x,sr,mlim,plim,fhop1ms,fhop2ms)
%
%   x:       input signal
%   sr:      sampling rate
%   mlim:    modulation spectrum limits for truncation
%   plim:    phase limits for substitution
%   fhop1ms: hop length in ms for base tranform
%   fhop2ms: hop length in ms for modulation spectrum transform (can be a
%            matrix in order to define multiresolution modspec)
%
%   y:       output signal
%   d:       difference signal for SNR's etc.

% ------- modcodec.m ---------------------------------------
% Marios Athineos, marios@ee.columbia.edu
% http://www.ee.columbia.edu/~marios/
% Copyright (c) 2003 by Columbia University.
% All rights reserved.
% ----------------------------------------------------------

% Convert from ms to samples
fhop1 = round(fhop1ms*sr/1000);
fhop2 = round(fhop2ms*sr/(fhop1*1000));

% Do the base transform
[fxdm,fxdp,fpad1] = basetran(x,fhop1);

% For the phase we can do noise substitution
fxdp = phasesub(fxdp,plim,'rand');

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
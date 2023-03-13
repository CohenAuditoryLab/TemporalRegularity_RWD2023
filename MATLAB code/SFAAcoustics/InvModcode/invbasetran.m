function y = invbasetran(fxdm,fxdp,fpad)
% INVBASETRAN Inverse base transform of the modulation codec
%   y = invbasetran(fxdm,fxdp,fpad)
%
%   fxdm:  magnitude part
%   fxdp:  phase part
%   fpad:  padding information needed for OLA
%   y:     output signal

% ------- invbasetran.m ------------------------------------
% Marios Athineos, marios@ee.columbia.edu
% http://www.ee.columbia.edu/~marios/
% Copyright (c) 2003 by Columbia University.
% All rights reserved.
% ----------------------------------------------------------

% Get transform frame length
[tlen,fnum] = size(fxdm);
fhop = tlen;
flen = 2*fhop;
win  = 'lowin';

% Recombine mag and phase and get ready to invert
ifxd = fxdm.*exp(i*fxdp);
clear fxdm fxdp;

% Get the sine and cosine parts
ifxs = imag(ifxd);
ifxc = real(ifxd);
clear ifxd;

% Make the inverse Cosine and Sine transformation matrices
iTs = mdxtmtx(flen,'b',@sin,0);
iTc = mdxtmtx(flen,'b',@cos,0);

% Invert each one
ifxo = iTs*ifxs;
ifxe = iTc*ifxc;
clear ifxs ifxc iTs iTc;

% Preallocate so that we can recombine
ifx = zeros(size(ifxo,1),2*size(ifxo,2));
ifx(:,2:2:end) = ifxo;
ifx(:,1:2:end) = ifxe;
clear ifxo ifxe;

% Rewindow
ifx = winit(ifx,win);

% OLA
y = linunframe(ifx,fhop,fpad);

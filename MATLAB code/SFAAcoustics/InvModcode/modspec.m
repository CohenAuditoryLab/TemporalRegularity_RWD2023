function [fXc,fpad] = modspec(fxdm,fhop)
% MODSPEC
%   [fXc,fpad] = modspec(fxdm,fhop)
%
%
%   fxdm:  the magnitude matrix (output of basetran)
%   fhop:  scalar or vector that specifies the hop per frequency bin
%          must be either scalar or have length size(fxdm,1)
%   fXc:   the modulation spectrum (either 3D matrix or cell array)
%   fpad:  needed for OLA of mod spec (either vector or cell array)

% ------- modspec.m ----------------------------------------
% Marios Athineos, marios@ee.columbia.edu
% http://www.ee.columbia.edu/~marios/
% Copyright (c) 2003 by Columbia University.
% All rights reserved.
% ----------------------------------------------------------

% Number of frequency bins x number of frames
[bnum,fnum] = size(fxdm);

% Window and its length
% (Other TDAC windows are 'lowin', 'rectwintdac', 'trapezwin')
win  = 'sinewin';
flen = 2*fhop;

if length(fhop) == 1;
    % Here we can do 3D matrix (faster than cell)
    
    % Make the transformation matrix
    Tc  = mdxtmtx(flen,'f',@cos,0.5);
    
    % Do a test linframe to get dimensions for preallocation of the cube
    [tmp,crap] = linframe(fxdm(1,:).',fhop,flen,'sym');
    
    % Ge the modulation frame length and the modulation frame count
    [mflen,mfnum] = size(tmp);
    
    % So now we can preallocate the following cube that represents the mod
    % spectrum (freq x time x mod freq)
    fXc = zeros(bnum,mfnum,mflen/2);
    
    % For each frequency bin
    for I = 1:bnum
        % Frame it with 50% overlap
        [fX,fpad] = linframe(fxdm(I,:).',fhop,flen,'sym');
        
        % Window it
        fX = winit(fX,win);
        
        % Take the MDCT (les2)
        fXc(I,:,:) = (Tc*fX).';
    end
    
else
    % Here we have to use cell arrays
    
    % For each frequency bin
    for I = 1:bnum
        % Make the transformation matrices
        Tc  = mdxtmtx(flen(I),'f',@cos,0.5);
        
        % Frame it with 50% overlap
        [fX,fpad{I}] = linframe(fxdm(I,:).',fhop(I),flen(I),'sym');
        
        % Window it
        fX = winit(fX,win);
        
        % Take the MDCT (les2)
        fXc{I} = Tc*fX;
    end    
end
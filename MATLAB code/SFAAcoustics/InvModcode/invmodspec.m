
function fxdm = invmodspec(fXc,fpad)
% INVMODSPEC
%   fxdm = invmodspec(fXc,fpad)
%
%   fXc:   the modulation spectrum (either 3D matrix or cell array)
%   fpad:  needed for OLA of mod spec (either vector or cell array)
%   fxdm:  the magnitude matrix (feed it to invbasetran)

% ------- invmodspec.m -------------------------------------
% Marios Athineos, marios@ee.columbia.edu
% http://www.ee.columbia.edu/~marios/
% Copyright (c) 2003 by Columbia University.
% All rights reserved.
% ----------------------------------------------------------

% Window (Other TDAC windows are 'lowin', 'rectwintdac', 'trapezwin')
win  = 'sinewin';

if iscell(fXc)
    % Get number of bins
    bnum = length(fXc);
    
    % Preallocate the output matrix
    fxdm = zeros(bnum,fpad{1}(3));
    
    for I = 1:bnum
        % Get number of mod bins and number of frames
        [mnum,fnum] = size(fXc{I});
        
        % Make the transformation matrices
        iTc  = mdxtmtx(2*mnum,'b',@cos,0.5);
        
        % Invert
        ifX = iTc*fXc{I};
        
        % Rewindow
        ifX = winit(ifX,win);
        
        % OLA
        fxdm(I,:) = linunframe(ifX,mnum,fpad{I}).';
    end
else
    % Get number of bins and number of frames
    [bnum,fnum,mnum] = size(fXc);

    % Preallocate the output matrix
    fxdm = zeros(bnum,fpad(3));
    
    % Make the transformation matrix
    iTc  = mdxtmtx(2*mnum,'b',@cos,0.5);
    
    for I = 1:bnum
        % Invert
        ifX = iTc*(squeeze(fXc(I,:,:)).');
        
        % Rewindow
        ifX = winit(ifX,win);
        
        % OLA
        fxdm(I,:) = linunframe(ifX,mnum,fpad).';
    end    
end

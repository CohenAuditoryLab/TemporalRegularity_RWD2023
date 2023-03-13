
function p = phasesub(p,lim,type)
% PHASESUB Phase substitution function
%   p = phasesub(p,lim,type)
%
%   Applies phase substitution to a phase matrix
%
%   p:    phase matrix (freq bins x #frames)
%   lim:  limits (e.g. (1/2,1) substitutes upper half)
%   type: substitution (e.g. 'rand', 'zero')

% ------- phasesub.m ---------------------------------------
% Marios Athineos, marios@ee.columbia.edu
% http://www.ee.columbia.edu/~marios/
% Copyright (c) 2003 by Columbia University.
% All rights reserved.
% ----------------------------------------------------------

if isempty(lim); return; end

% Get number of bins and number of frames
[bnum,fnum] = size(p);

% Adjust limits to integer indices
lim(find(lim==0)) = eps;
lim = ceil(bnum*lim);

switch lower(type)
    case 'rand'
        p(lim(1):lim(2),:) = 2*pi*rand(lim(2)-lim(1)+1,fnum) - pi;
    case 'zero'
        p(lim(1):lim(2),:) = 0;
    otherwise
        error('Unknown phase substitution type, try rand or zero');
end

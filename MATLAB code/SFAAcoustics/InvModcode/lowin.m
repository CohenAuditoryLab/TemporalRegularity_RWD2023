
function y = lowin(N)
% LOWIN Low-Overlap window.
%   y = lowin(N)
%
%   Used in MDCT transform for TDAC. Taken from the Low-Delay
%   specification for MPEG-4 Audio Version 2. Length must be
%   a multiple of 16.
%
%   N: length of window to create
%   y: the window in column

% ------- lowin.m ------------------------------------------
% Marios Athineos, marios@ee.columbia.edu
% http://www.ee.columbia.edu/~marios/
% Copyright (c) 2002 by Columbia University.
% All rights reserved.
% ----------------------------------------------------------

if (rem(N,16)~=0)
    error('Window length must be a multiple of 16')
end
N4  = N/4;
N16 = N/16;

% Preallocate window
y = zeros(N,1);

% Sine up
i = ((3*N16):(5*N16-1)).';
y(i+1) = sin(pi*(i-3*N16+0.5)/N4);

% Flat one
i = ((5*N16):(11*N16-1)).';
y(i+1) = 1;

% Sine down
i = ((11*N16):(13*N16-1)).';
y(i+1) = sin(pi*(i-9*N16+0.5)/N4);

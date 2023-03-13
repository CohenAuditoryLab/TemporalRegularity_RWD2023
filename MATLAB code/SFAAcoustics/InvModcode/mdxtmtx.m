
function T = mdxtmtx(N,fb,fun,c)
% MDXTMTX Creates various MDCT and MDST transformation matrices 
%   T = mdxtmtx(N,fb,fun,c)
%
%   Of course this is very slow since we transform using a full matrix
%   multiplication but on the other hand it's very clean programmatically.
%
%   N:    Frame length
%   fb:   'f' = forward, 'b' = backward
%   fun:  Either @cos or @sin for MDCT or MDST
%   c:    Offset for example 0 or 0.5

% ------- mdxtmtx.m ----------------------------------------
% Marios Athineos, marios@ee.columbia.edu
% http://www.ee.columbia.edu/~marios/
% Copyright (c) 2003 by Columbia University.
% All rights reserved.
% ----------------------------------------------------------

% Make sure N is even
if (rem(N,2)~=0)
    error('MDCT is defined only for even lengths.');
end
if N > 4096
    error('Out of memory in mdxtmtx');
end

% We need these for furmulas below
M  = N/2;       % Number of coefficients
N0 = (M+1)/2;
C  = 2/sqrt(N);

% Create the grid for the vectorization
[n,k] = meshgrid(0:(N-1),0:(M-1));

% If it is backward, transpose
if fb == 'b'
    n = n.'; k = k.';
end
    
% Create the transformation matrix
T = C * feval(fun, pi*(n+N0).*(k+c)/M);
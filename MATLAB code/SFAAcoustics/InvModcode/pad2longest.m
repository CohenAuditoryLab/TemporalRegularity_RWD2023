
function [x,y] = pad2longest(x,y)
% PAD2LONGEST Zero pads one or the other to the longest of the two
%   [x,y] = pad2longest(x,y)
%
%   x:  signal one
%   y:  signal two

% ------- pad2longest.m ------------------------------------
% Marios Athineos, marios@ee.columbia.edu
% http://www.ee.columbia.edu/~marios/
% Copyright (c) 2003 by Columbia University.
% All rights reserved.
% ----------------------------------------------------------

% Make same length for taking difference etc.
if length(x) > length(y)
    y(length(x)) = 0;
elseif length(x) < length(y)
    x(length(y)) = 0;
end
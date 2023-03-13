
% KBDWIN Kaiser-Bessel Derived window.
%   y = kbdwin(len,alpha)
%
%   Used in MDCT transform for TDAC
%
%   len:   length of window to create
%   alpha: alpha parameter for window shape
%   y:     the window in column

% Mex-file by:
% ------- kbdwin.m -----------------------------------------
% Marios Athineos, marios@ee.columbia.edu
% http://www.ee.columbia.edu/~marios/
% Copyright (c) 2002 by Columbia University.
% All rights reserved.
% ----------------------------------------------------------

% Original C code by:
% Programmer:    Craig Stuart Sapp <craig@ccrma.stanford.edu>
% Creation Date: Sat Jan 27 14:27:14 PST 2001
% Last Modified: Sat Jan 27 15:24:58 PST 2001
% Filename:      kbdwindow.cpp
% $Smake:        g++ -O6 -o %b %f
% Syntax:        C++; functions in ANSI C
%
% Description:   This is a sample program for generating
%                Kaiser-Bessel Derived Windows for audio signal
%                processing -- in particular for the Time-Domain
%                Alias Cancellation procedure which has the
%                overlap-add requirements:
%                   Window_m[N-1-n] + Window_m+1[n]^2 = 1.0;
%                which means: The squares of the overlapped
%                windows must add to the constant value 1.0
%                for time domain alias cancellation to work.
%
%                The two function necessary to create the KBD window are:
%                   KBDWindow -- calculates the window values.
%                   BesselI0  -- bessel function needed for KBDWindow.

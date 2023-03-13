
function m = modsub(m,lim,type)
% MODSUB Modulation spectrum subtitution function
%   m = modsub(m,lim,type)
%
%   m:    modulation spectrum (3D matrix or cell) (freq x time x mod freq))
%   lim:  limits (e.g. (1/2,1) substitutes upper half) (can be a matrix)
%         dim 1 is a function of bin number
%   type: substitution (e.g. 'zero')

% ------- modsub.m -----------------------------------------
% Marios Athineos, marios@ee.columbia.edu
% http://www.ee.columbia.edu/~marios/
% Copyright (c) 2003 by Columbia University.
% All rights reserved.
% ----------------------------------------------------------
 
if isempty(lim); return; end
lim(find(lim==0)) = eps;

if iscell(m)
    % Get number of bins
    bnum = length(m);

    if size(lim) == [1,2];
        % Expand for all bins
        lim = repmat(lim,bnum,1);
    end

    switch lower(type)
        case 'zero'
            for I = 1:bnum
                % Get number of mod bins and number of frames
                [mnum,fnum] = size(m{I});
                
                % Adjust limits to integer indices
                lim(I,:) = ceil(mnum*lim(I,:));
                
                % This is really cool indexing (perl-like)
                m{I}(lim(I,1):lim(I,2),:) = 0;
            end
        case 'mult'
        case 'add'
        case 'shift'
        case 'warp'
        otherwise
            error('Unknown modulation substitution type, try zero');
    end
else
    % Get number of bins and number of frames
    [bnum,fnum,mnum] = size(m);

    if size(lim) == [1,2];
        % Expand for all bins
        lim = repmat(lim,bnum,1);
    end
    
    % Adjust limits to integer indices
    lim = ceil(mnum*lim);
    
    switch lower(type)
        case 'zero'
            % I am not in the mood to make the linear index here :)
            for I = 1:bnum
                m(I,:,lim(I,1):lim(I,2)) = 0;
            end
        case 'mult'
        case 'add'
        case 'shift'
        case 'warp'
        otherwise
            error('Unknown modulation substitution type, try zero');
    end    
end

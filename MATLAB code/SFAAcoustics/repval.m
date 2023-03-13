function [RVNR, POS]=repval(v,mult);
%REPVAL Repeated Values
%   repval(X) finds all repeated values for input X, and their attributes. 
%   The input may be vector, matrix, char string, or cell of strings
%
%   mult = if you want the output to be only those with multiple repeats
%
%   Y=repval(X) returns the repeated values of X
%
%   [RV, NR, POS, IR]=repval(X) returns the following outputs
%      RV  : Repeated Values (sorted)
%      NR  : Number of times each RV was repeated
%      POS : Position vector of X of RV entries
%


B = unique(v); %B are the unique values of v (removed duplicates). 
    
for i = 1:length(B) %Did this because I had to fill empty spaces with '0'. If no empty cells then start from 1.   
    I = find(v == B(i)); 
    POS{i} = I;
    NR(i) = numel(I);
end
NR = NR';    
RV = B;

RVNR = [RV NR];

if nargin > 1
    [a,~] = find(NR == 1);
    RVNR(a,:) = [];
    if nargout > 1
        POS{a} = [];
        POS = POS(~cellfun(@isempty, POS));
    end
end




function [RVNR,POS]=repvalstr(v, mult);
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

B = unique(v); %B are the unique values of v (removed duplicates). B(J) = v, where length(J) = length(v) and values of J range 1:length(B)
for i = 1:length(B) %Started at 2 because I had to fill empty spaces with '0'. If no empty cells then start from 1.   
    I = find(strcmp(B{i},v)); 
    POS{i} = I;
    NR{i} = numel(I);
end
NR = NR';    
RV = B(1:end);

RVNR = [RV NR];

if nargin > 1
    [a,~] = find(NR == 1);
    RVNR{a,:} = [];
    POS{a} = [];
    POS = POS(~cellfun(@isempty, POS));
    RVNR = RVNR(~cellfun(@isempty,RVNR));
end

end
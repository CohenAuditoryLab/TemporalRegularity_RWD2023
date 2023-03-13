function [expanded_matrix] = RWD_quadexpand(matrix, nolin)
%RWD_QUADEXPAND Quadratically expand a matrix using similar implementation to what is  used in SFA code
%   Quadratically expand a matrix to generate a matrix containing all
%   linear components and all UNIQUE correlated components assumes that
%   matrix has "neuroscience configuration" instead of standard linear
%   algebra configuration, i.e. matrix is var x obs instead of obs x var
%   so in the case of frequency changing over time.  Each column is a
%   moment in time and each row is a frequency bin


if size(matrix,1) > size(matrix,2)
    
    warning('Rows exceed columns.  This code assumes var x obs structure')
    display('Did you enter data correctly?')
%     test = input(' ')
%     
%     if strcmp(test,'y')
%         
%     end

    
end

%preallocate with nan to ensure errors are caught
%Note formula for quadratic expansion: e.g. [x1, x2, x3, x1x1, x1x2, x1x3,
%x2,x2 ,x2x3...etc] NOT collapsed over time (i.e. can't just use cov)
%Consider matrix m x n
%Unique quadratic components is going to follow the formula for the sum of consecutive
%intergers: m*(m+1)/2
%Including the diagonal is simply + m
%Simplifying m*(m+1)/2 +m yields:  (m^2 + 3m)/2

expanded_matrix = nan((size(matrix,1)^2 + 3*size(matrix,1))/2, size(matrix,2));


%adding in normalization to see if that changes results.  Seems like it
%would explain why correlations seem to just replicate the original mod
%spec but with some changes in

matrix = matrix - mean(matrix,2);

matrix = matrix./sqrt(mean(matrix.^2)+1e-11);
%2022-05-21 getting nans so adapting to use Chetan's fix for rows with all zeros



expanded_matrix(1:size(matrix,1),:) = matrix; %First rows are simply the linear components

%Essentially borrowing what Chetan did 


count = size(matrix,1)+1;
for cur_var = 1:size(matrix,1)
    
    cur_x = matrix(cur_var,:); %grab variable currently using
    
    for cur_var2 = cur_var:size(matrix,1) %To get unique only, start from diagaonal and work down
        
        cur_x2 = matrix(cur_var2,:); %grab variable to pair with cur_var
        
        expanded_matrix(count,:) = cur_x .* cur_x2; %take product and add to expanded_matrix
        
        count = count +1; %simply keep iterating to fill in the rows properly.
        
    end
        
end

if nolin %toggle to remove linear components
    
    expanded_matrix = expanded_matrix(size(matrix,1)+1:end,:);
    
end



end


function [DisR, DisN, DisB, DisE, DisK, num] = Dendro(C,fig,num);
%To create the dendrogram of features measured using Praat.

%C = struct containing information from the stimuli calculated by
%Praat.
%fig = 0 will display the dendrograms, while fig = 1 (or any other number)
%will suppress the display of dendrograms.
%num = any number, important because if used for Grunts, there is no mean
%harmonicity.

for i = 1:length(C.Duration)
    for j = 1:length(C.Duration)
        L(i,j) = abs(C.Duration(i) - C.Duration(j));
        M(i,j) = abs(C.SDofInt(i) - C.SDofInt(j));
        A(i,j) = abs(C.CoG(i) - C.CoG(j));
        F(i,j) = abs(C.Entropy(i) - C.Entropy(j));
        if nargin < 3 %Must enter a number for analysis of grunts.
        H(i,j) = abs(C.MeanHarm(i) - C.MeanHarm(j));
        end
    end
end

%Scale the distances to be between 0 and 1.
for i = 1:length(C.Duration)
    for j = 1:length(C.Duration)       
        R(i,j) = (L(i,j) - min(min(L)))./(max(max(L)) - min(min(L)));
        N(i,j) = (M(i,j) - min(min(M)))./(max(max(M)) - min(min(M)));        
        B(i,j) = (A(i,j) - min(min(A)))./(max(max(A)) - min(min(A)));
        E(i,j) = (F(i,j) - min(min(F)))./(max(max(F)) - min(min(F)));
        if nargin < 3
        K(i,j) = (H(i,j) -  min(min(H)))./(max(max(H)) - min(min(H)));
        else K(i,j) = 0;
        end
    end
end 

%Add one to every value of above except the diagonal (this may be specific to Praat data because unnecessary for correlation values). This will be
%dissimilarity matrix for MDS function. For some reason MDS will not accept
%the above matrices, but adding 1 magically makes it work. Trust me!
DisR = 1 + R; DisR = DisR - diag(diag(DisR));
DisN = 1 + N; DisN = DisN - diag(diag(DisN));
DisB = 1 + B; DisB = DisB - diag(diag(DisB));
DisE = 1 + E; DisE = DisE - diag(diag(DisE));
if nargin < 3
    DisK = 1 + K; DisK = DisK - diag(diag(DisK));
else DisK = 0;
end
 
    tree = linkage(R); D1 = pdist(R);
    leaforder1 = optimalleaforder(tree,D1);   
    
    fern = linkage(N); D2 = pdist(N);
    leaforder2 = optimalleaforder(fern,D2);
    
    ivy = linkage(B); D3 = pdist(B);
    leaforder3 = optimalleaforder(ivy,D3);
    
    oak = linkage(E); D4 = pdist(E);
    leaforder4 = optimalleaforder(oak,D4);
    
    if nargin < 3
        holly = linkage(K); D5 = pdist(K);
        leaforder5 = optimalleaforder(holly,D5);
    else leaforder5 = 0;
    end
    
if nargout < 1 || fig == 0    
    figure(1); dendrogram(tree,0,'Reorder', leaforder1);
    title('Duration (s)')
    figure(2); dendrogram(fern,0,'Reorder', leaforder2);
    title('SD of Intensity (dB)')
    figure(3); dendrogram(ivy,0,'Reorder', leaforder3);
    title('Center of Gravity (Hz)')
    figure(4); dendrogram(oak,0,'Reorder', leaforder4);
    title('Entropy')
    
    if nargin < 3
        figure(5); dendrogram(holly,0,'Reorder',leaforder5);
        title('Mean Harmonicity (dB)')
    end
end
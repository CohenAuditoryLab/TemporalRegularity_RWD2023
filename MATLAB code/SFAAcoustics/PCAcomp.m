function [R] = PCAcomp(C,G,j);
% j = the principal component you want to look at.

[PCAC,~] = PlotPCA(C);
[PCAG,~] = PlotPCA(G,1);

CName = fieldnames(C);
CNames = {'Center of Gravity (Hz)';'Mean Harmonicity (dB)';'SD of Intensity (dB)'; ...
'Duration (s)'; 'Entropy'};  
R = struct('C',[],'G',[]);
for x = 1:numel(CName)
    figure; scatter(C.(CName{x}),PCAC(:,j), '+k')
    title('Coos','fontsize',14); ylabel(['PCA Component ' num2str(j)],'fontsize',14); xlabel(CNames{x},'fontsize',14)
    [r,q] = corrcoef(C.(CName{x}),PCAC(:,j));
    R.C(x,1) = r(2); R.C(x,2) = q(2);
    %print('-djpeg','-r400',['PCAC' num2str(x)]);
end
GName = fieldnames(G);
GNames = {'Center of Gravity (Hz)'; 'SD of Intensity (dB)';'Duration (s)'; 'Entropy'}; 
for x = 1:numel(GName)
    figure; scatter(G.(GName{x}),PCAG(:,j), '+k')
    title('Grunts','fontsize',14); ylabel(['PCA Component ' num2str(j)],'fontsize',14); xlabel(GNames{x},'fontsize',14)
    [r,q] = corrcoef(G.(GName{x}),PCAG(:,j));
    R.G(x,1) = r(2); R.G(x,2) = q(2);
    %print('-djpeg','-r400',['PCAG' num2str(x)]);
end
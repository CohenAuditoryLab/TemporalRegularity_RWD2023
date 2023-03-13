function [PCA,V] = PlotPCA(C,num);

%Scale the values to be between -1 and 1.
for i = 1:length(C.Duration) 
    Dur(i) = 2*(C.Duration(i) - min(C.Duration))./(max(C.Duration) - min(C.Duration))-1;
    Int(i) = 2*(C.SDofInt(i) - min(C.SDofInt))./(max(C.SDofInt) - min(C.SDofInt))-1;        
    Grav(i) = 2*(C.CoG(i) - min(C.CoG))./(max(C.CoG) - min(C.CoG))-1;
    Ent(i) = 2*(C.Entropy(i) - min(C.Entropy))./(max(C.Entropy) - min(C.Entropy))-1;
    if nargin < 2
        Harm(i) = 2*(C.MeanHarm(i) -  min(C.MeanHarm))./(max(C.MeanHarm) - min(C.MeanHarm))-1;
    else Harm(i) = 0;
    end
end 

if nargin < 2
    Cat1 = [Dur' Int' Grav' Ent' Harm'];
    Cat2 = [Dur;Int;Grav;Ent;Harm];
else Cat1 = [Dur' Int' Grav' Ent'];
    Cat2 = [Dur;Int;Grav;Ent];
end

Cov1 = cov(Cat1); [v,~] = eigs(Cov1,3);
Cov2 = cov(Cat2);
[PCA,V] = pcacov(Cov2);
mapcaplot(Cov2);
figure; imagesc(v); colorbar; box off; 
end
%[CooCluster, GruntCluster, C, G] = MDSCluster('PlotSimMat');

function [R] = ClusterComp(C, G, CooCluster, GruntCluster);
%CooCluster = MDS data of Coos for all Praat features and MS and LFS
%GruntCluster = MDS data of Grunts for all Praat features and MS and LFS
%C = Raw Praat data for Coos
%G = Raw Praat data for Grunts

if nargin > 2
%Want to plot against ascending values
[CDuration, CDord] = sort(C.Duration);
[CSDofInt, CIord] = sort(C.SDofInt);
[CCoG, CGord] = sort(C.CoG);
[CMeanHarm, CHord] = sort(C.MeanHarm);
[CEntropy, CEord] = sort(C.Entropy);

[GDuration, GDord] = sort(G.Duration);
[GSDofInt, GIord] = sort(G.SDofInt);
[GCoG, GGord] = sort(G.CoG);
[GEntropy, GEord] = sort(G.Entropy);

%For plotting, want the correct corresponding MDS values to match up with
%vocalization.
for i = 1:length(CDuration)
    CD(CooCluster.Duration(1,i)) = CooCluster.Duration(2,i);
    CH(CooCluster.MeanHarm(1,i)) = CooCluster.MeanHarm(2,i);
    CMS(CooCluster.MS(1,i)) = CooCluster.MS(2,i);
    CLFS(CooCluster.LFS(1,i)) = CooCluster.LFS(2,i);
    CI(CooCluster.SDofInt(1,i)) = CooCluster.SDofInt(2,i);
    CCOG(CooCluster.CoG(1,i)) = CooCluster.CoG(2,i);
    CE(CooCluster.Entropy(1,i)) = CooCluster.Entropy(2,i);
end

for i = 1:length(GDuration)
    GD(GruntCluster.Duration(1,i)) = GruntCluster.Duration(2,i);   
    GMS(GruntCluster.MS(1,i)) = GruntCluster.MS(2,i);
    GLFS(GruntCluster.LFS(1,i)) = GruntCluster.LFS(2,i);
    GI(GruntCluster.SDofInt(1,i)) = GruntCluster.SDofInt(2,i);
    GCOG(GruntCluster.CoG(1,i)) = GruntCluster.CoG(2,i);
    GE(GruntCluster.Entropy(1,i)) = GruntCluster.Entropy(2,i);
end
R = struct('RC',[],'PC',[],'RG',[],'PG',[]);
P1 = [CD(CEord); CI(CEord); CH(CEord); CMS(CEord); CLFS(CEord); CCOG(CEord); CE(CEord)];
%Correlation between the MDS measures
for i = 1:size(P1,1)
    for j = 1:size(P1,1)
    [rc, pc] = corrcoef(P1(i,:),P1(j,:));
    R.RC(i,j) = rc(2);
    R.PC(i,j) = pc(2);
    end
end

T1 = [GD(GEord); GI(GEord); GMS(GEord); GLFS(GEord); GCOG(GEord); GE(GEord)];
for i = 1:size(T1,1)
    for j = 1:size(T1,1)
        [rg,pg] = corrcoef(T1(i,:),T1(j,:));
        R.RG(i,j) = rg(2);
        R.PG(i,j) = pg(2);
    end
end

%Plot everything together
if nargout < 1

fig1 = figure; scatter(CDuration,CD(CDord))
hold on
scatter(CDuration,CH(CDord), 'x');scatter(CDuration,CI(CDord), '.');scatter(CDuration,CMS(CDord), '+');scatter(CDuration,CLFS(CDord), 'd');scatter(CDuration,CCOG(CDord), '*');scatter(CDuration,CE(CDord), 's');
xlabel('Ascending Duration (s)'); ylabel('MDS zDistance'); title('Coos')
set(fig1,'Position', [1 1 864 531]);
leg1 = legend('Duration', 'Mean Harm', 'SD of Intensity', 'Mod Spec', 'Log Freq Spec', 'Center of Gravity', 'Entropy', 'Location', 'Best');
set(leg1,'FontSize', 8);
%for i = 1:length(CDord)
%    text(CDuration(i),CD(CDord(i)),num2str(CDord(i)));
%end
hold off
print('-djpeg','-r400','CDuration');

fig2 = figure; scatter(CMeanHarm,CD(CHord))
hold on
scatter(CMeanHarm,CH(CHord), 'x');scatter(CMeanHarm,CI(CHord), '.');scatter(CMeanHarm,CMS(CHord), '+');scatter(CMeanHarm,CLFS(CHord), 'd');scatter(CMeanHarm,CCOG(CHord), '*');scatter(CMeanHarm,CE(CHord), 's');
xlabel('Ascending Mean Harmonicity (dB)');ylabel('MDS zDistance');title('Coos')
set(fig2,'Position',[1 1 864 531]);
leg2 = legend('Duration', 'Mean Harm', 'SD of Intensity', 'Mod Spec', 'Log Freq Spec', 'Center of Gravity', 'Entropy', 'Location', 'Best');
set(leg2,'FontSize', 8);
hold off
print('-djpeg','-r400','CMeanHarm');

fig3 = figure; scatter(CSDofInt,CD(CIord))
hold on
scatter(CSDofInt,CH(CIord), 'x');scatter(CSDofInt,CI(CIord), '.');scatter(CSDofInt,CMS(CIord), '+');scatter(CSDofInt,CLFS(CIord), 'd');scatter(CSDofInt,CCOG(CIord), '*');scatter(CSDofInt,CE(CIord), 's');
xlabel('Ascending SD of Intensity (dB)');ylabel('MDS zDistance');title('Coos')
set(fig3,'Position', [1 1 864 531]);
leg3 = legend('Duration', 'Mean Harm', 'SD of Intensity', 'Mod Spec', 'Log Freq Spec', 'Center of Gravity', 'Entropy', 'Location', 'Best');
set(leg3,'FontSize', 8);
hold off
print('-djpeg','-r400','CSDofInt');

fig4 = figure; scatter(CCoG,CD(CGord))
hold on
scatter(CCoG,CH(CGord), 'x');scatter(CCoG,CI(CGord), '.');scatter(CCoG,CMS(CGord), '+');scatter(CCoG,CLFS(CGord), 'd');scatter(CCoG,CCOG(CGord), '*');scatter(CCoG, CE(CGord), 's')
xlabel('Ascending Center of Gravity (Hz)');ylabel('MDS zDistance');title('Coos')
set(fig4,'Position', [1 1 864 531]);
leg4 = legend('Duration', 'Mean Harm', 'SD of Intensity', 'Mod Spec', 'Log Freq Spec', 'Center of Gravity', 'Entropy', 'Location', 'Best');
set(leg4,'FontSize', 8);
hold off
print('-djpeg','-r400','CCoG');

fig5 = figure; scatter(CEntropy, CD(CEord))
hold on
scatter(CEntropy,CH(CEord), 'x');scatter(CEntropy,CI(CEord), '.');scatter(CEntropy,CMS(CEord), '+');scatter(CEntropy,CLFS(CEord), 'd');scatter(CEntropy,CCOG(CEord), '*');scatter(CEntropy, CE(CEord), 's')
xlabel('Ascending Entropy');ylabel('MDS zDistance');title('Coos')
set(fig5,'Position', [1 1 864 531]);
leg5 = legend('Duration', 'Mean Harm', 'SD of Intensity', 'Mod Spec', 'Log Freq Spec', 'Center of Gravity', 'Entropy', 'Location', 'Best');
set(leg5,'FontSize', 8);
hold off
print('-djpeg', '-r400', 'CEnt');

fig6 = figure; scatter(GCoG,GD(GGord))
hold on
scatter(GCoG,GI(GGord), '.');scatter(GCoG,GMS(GGord), '+');scatter(GCoG,GLFS(GGord), 'd');scatter(GCoG,GCOG(GGord), '*');scatter(GCoG, GE(GGord), 's');
xlabel('Ascending Center of Gravity (Hz)');ylabel('MDS zDistance');title('Grunts')
set(fig6,'Position', [1 1 864 531]);
leg6 = legend('Duration', 'SD of Intensity', 'Mod Spec', 'Log Freq Spec', 'Center of Gravity', 'Entropy', 'Location', 'Best');
set(leg6,'FontSize', 8);
hold off
print('-djpeg','-r400','GCoG');

fig7 = figure; scatter(GDuration,GD(GDord))
hold on
scatter(GDuration,GI(GDord), '.');scatter(GDuration,GMS(GDord), '+');scatter(GDuration,GLFS(GDord), 'd');scatter(GDuration,GCOG(GDord), '*');scatter(GDuration,GE(GDord), 's');
xlabel('Ascending Duration (s)');ylabel('MDS zDistance');title('Grunts')
set(fig7,'Position', [1 1 864 531]);
leg7 = legend('Duration', 'SD of Intensity', 'Mod Spec', 'Log Freq Spec', 'Center of Gravity', 'Entropy', 'Location', 'Best');
set(leg7,'FontSize', 8);
hold off
print('-djpeg','-r400','GDuration');

fig8 = figure; scatter(GSDofInt,GD(GIord))
hold on
scatter(GSDofInt,GI(GIord), '.');scatter(GSDofInt,GMS(GIord), '+');scatter(GSDofInt,GLFS(GIord), 'd');scatter(GSDofInt,GCOG(GIord), '*');scatter(GSDofInt,GE(GIord), 's')
xlabel('Ascending SD of Intensity (dB)');ylabel('MDS zDistance');title('Grunts')
set(fig8,'Position', [1 1 864 531]);
leg8 = legend('Duration', 'SD of Intensity', 'Mod Spec', 'Log Freq Spec', 'Center of Gravity', 'Entropy', 'Location', 'Best');
set(leg8,'FontSize', 8);
hold off
print('-djpeg','-r400','GSDofInt');

fig9 = figure; scatter(GEntropy,GD(GEord))
hold on
scatter(GEntropy,GI(GEord), '.');scatter(GEntropy,GMS(GEord), '+');scatter(GEntropy,GLFS(GEord), 'd');scatter(GEntropy,GCOG(GEord), '*');scatter(GEntropy,GE(GEord), 's')
xlabel('Ascending Entropy');ylabel('MDS zDistance');title('Grunts')
set(fig9,'Position', [1 1 864 531]);
leg9 = legend('Duration', 'SD of Intensity', 'Mod Spec', 'Log Freq Spec', 'Center of Gravity', 'Entropy', 'Location', 'Best');
set(leg9,'FontSize', 8);
hold off
print('-djpeg','-r400','GEnt');
end 
    end
end

        
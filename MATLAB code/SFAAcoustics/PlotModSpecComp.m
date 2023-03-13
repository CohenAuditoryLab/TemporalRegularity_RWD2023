%cd 'C:\Users\Jennifer\Desktop\DTW stuff'
load('/Volumes/OroroBackup/Jennifer/Cohen Rotation/Data/PlotSimMat.mat')

figure; imagesc(RHOC);
xlabel('Voc #','fontsize', 14)
ylabel('Voc #','fontsize', 14)
title('Coos MS similarity matrix');
colormap(whirlpool)
colorbar; caxis([0.65 1]);
print('-djpeg','-r400','SimMatMSC')

figure; imagesc(RHOG);
xlabel('Voc #','fontsize', 14)
ylabel('Voc #','fontsize', 14)
title('Grunts MS similarity matrix');
colormap(whirlpool)
colorbar; caxis([0.65 1]);
print('-djpeg','-r400','SimMatMSG')

figure; MDS(DisRHOC, 3, 'Coos MS');
view(-29,-44);
print('-djpeg','-r400','MDSMSC')

figure; MDS(DisRHOG, 4, 'Grunts MS');
view(-29,-44);
print('-djpeg','-r400','MDSMSG')

figure; imagesc(RHO.Start);
xlabel('Voc #');
ylabel('Voc #');
title('Coos (1-31) vs Grunts(32-73) [Start]');
colormap Hot
colorbar; caxis([0.65 1]);

figure; imagesc(RHO.Middle);
xlabel('Voc #');
ylabel('Voc #');
title('Coos (1-31) vs Grunts(32-73) [Middle]');
colormap Hot
colorbar; caxis([0.65 1]);

figure; imagesc(RHO.End);
xlabel('Voc #');
ylabel('Voc #');
title('Coos (1-31) vs Grunts(32-73) [End]');
colormap Hot
colorbar; caxis([0.65 1]);



 

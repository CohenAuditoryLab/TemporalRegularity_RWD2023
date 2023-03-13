function [CooCluster, GruntCluster, C, G] = MDSCluster(file)

load(file);

[DisRC, DisNC, DisBC, DisEC, DisKC] = Dendro(C,1);
[DisRG, DisNG, DisBG, DisEG, ~, ~] = Dendro(G,1,1);

CooCluster = struct('MS', [], 'LFS', [], 'Duration', [], 'SDofInt', [], 'CoG', [], 'MeanHarm', [], 'Entropy', []);
GruntCluster = struct('MS', [], 'LFS', [], 'Duration', [], 'SDofInt', [], 'CoG', [], 'Entropy', []);

for i = 1%:12 %Change to 12 if you want to view all the plots

[zDistHist, Order, ~, ~] = MDS(DisRHOC, i, 'Coos MDS'); view(-29,-44); 
CooCluster.MS = [Order; zDistHist]; %pause;
[zDistHist, Order, ~, ~] = MDS(DisRHOG, i, 'Grunts MDS'); view(-29,-44); 
GruntCluster.MS = [Order; zDistHist]; %pause;

[zDistHist, Order, ~, ~] = MDS(logRHOS.DislogRHOC, i, 'Coos LFS'); view(-29,-44); 
CooCluster.LFS = [Order; zDistHist]; %pause;
[zDistHist, Order, ~, ~] = MDS(logRHOS.DislogRHOG, i, 'Grunts LFS'); view(-29,-44); 
GruntCluster.LFS = [Order; zDistHist]; %pause;

[zDistHist, Order, ~, ~] = MDS(DisRC, i, 'Coos Duration'); view(-29,-44); 
CooCluster.Duration = [Order;zDistHist]; %pause;
[zDistHist, Order, ~, ~] = MDS(DisNC, i, 'Coos SD of Intensity'); view(-29,-44);
CooCluster.SDofInt = [Order;zDistHist]; %pause;
[zDistHist, Order, ~, ~] = MDS(DisBC, i, 'Coos Center of Gravity'); view(-29,-44);
CooCluster.CoG = [Order;zDistHist]; %pause;
[zDistHist, Order, ~, ~] = MDS(DisKC, i, 'Mean Harmonicity'); view(-29,-44);
CooCluster.MeanHarm = [Order;zDistHist]; %pause;
[zDistHist, Order, ~, ~] = MDS(DisEC, i, 'Coos Entropy'); view(-29,-44);
CooCluster.Entropy = [Order;zDistHist]; %pause;

[zDistHist, Order, ~, ~] = MDS(DisRG, i, 'Grunts Duration'); view(-29,-44);
GruntCluster.Duration = [Order;zDistHist]; %pause;
[zDistHist, Order, ~, ~] = MDS(DisNG, i, 'Grunts SD of Intensity'); view(-29,-44);
GruntCluster.SDofInt = [Order;zDistHist]; %pause;
[zDistHist, Order, ~, ~] = MDS(DisBG, i, 'Grunts Center of Gravity'); view(-29,-44);
GruntCluster.CoG = [Order;zDistHist]; %pause;
[zDistHist, Order, ~, ~] = MDS(DisEG, i, 'Grunts Entropy'); view(-29,-44);
GruntCluster.Entropy = [Order;zDistHist];

end
end


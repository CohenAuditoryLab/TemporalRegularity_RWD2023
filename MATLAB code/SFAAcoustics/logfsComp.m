function [logRHO, DislogRHO, logRHOS, figures] = logfsComp(Coos, Grunts, logRHOS, logRHO, DislogRHO);
%Log-frequency for Coos and Grunts (note that this is just using the
%trimmed data.) and the similarity matrices along with MDS.
%Coos/Grunts are the structs containing the data extracted from wave files along with the
%data trimmed to the length of the shortest stimulus of its class.

%Output:
%   logRHO = struct containing between-type (Coo/Grunt) similarity matrices (Start,
%   Middle, and End);
%   DislogRHO = same as logRHO, but dissimilarity matrix for use in MDS.
%   logRHOS = struct containing within-type similarity AND dissimilarity matrices
%   figures = plots similarity matrices and then MDS.

if nargin < 3 %Gives the option of doing the calculations, or just displaying the figures.
logfCoos = struct('Y', [], 'MX', [], 'Unwrap', []);
logfGrunts = struct('Y', [], 'MX', [], 'Unwrap', []);

for v = 1:length(Coos)
    [Y,MX] = logfsgram(Coos(v).dataT,512,50000);
    logfCoos(v).Y = Y;
    logfCoos(v).MX = MX;
    Unwrap = reshape(logfCoos(v).Y,1,size(logfCoos(v).Y,1)*size(logfCoos(v).Y,2));
    logfCoos(v).Unwrap = Unwrap;
end

for u = 1:length(Grunts)
    [Y,MX] = logfsgram(Grunts(u).dataT,512,50000);
    logfGrunts(u).Y = Y;
    logfGrunts(u).MX = MX;
    Unwrap = reshape(logfGrunts(u).Y,1,size(logfGrunts(u).Y,1)*size(logfGrunts(u).Y,2));
    logfGrunts(u).Unwrap = Unwrap; 
end

logRHOC = zeros(size(logfCoos(1).Unwrap,1),size(logfCoos(1).Unwrap,1));
logRHOG = zeros(size(logfGrunts(1).Unwrap,1),size(logfGrunts(1).Unwrap,1));
RC = struct('r', []);
RG = struct('r', []);

DislogRHOC = zeros(length(Coos), length(Coos));
for i = 1:length(Coos)
    for j = 1:length(Coos)
        r = corrcoef(logfCoos(i).Unwrap,logfCoos(j).Unwrap);
        RC(i,j).r = r;
        logRHOC(i,j) = RC(i,j).r(2);
        DislogRHOC(i,j) = 1-logRHOC(i,j);
    end
end

DislogRHOG = zeros(length(Grunts), length(Grunts));
for n = 1:length(Grunts)
    for m = 1:length(Grunts)
        r = corrcoef(logfGrunts(n).Unwrap, logfGrunts(m).Unwrap);
        RG(n,m).r = r;
        logRHOG(n,m) = RG(n,m).r(2);
        DislogRHOG(n,m) = 1-logRHOG(n,m);
    end
end

L = length(Grunts(1).dataT);
CoosTrim = struct('StartLFS', [], 'MiddleLFS', [], 'EndLFS', []);
P = struct('StartData', [], 'MiddleData', [], 'EndData', []);
for n = 1:31
    StartData = Coos(n).data(1:L(1)); %First portion of the coo
    P(n).StartData = StartData;
    EndData = Coos(n).data(end - L(1) + 1:end); %End portion of the Coo
    P(n).EndData = EndData;
    MiddleData = Coos(n).data(end/2 - L(1)/2:end/2 + L(1)/2 - 1); %Middle portion of the Coo
    P(n).MiddleData = MiddleData;
end 

CoosTUnwrap = struct('UnwrapS', [], 'UnwrapM', [], 'UnwrapE', []);
Gruntmod = struct('UnwrapS', [], 'UnwrapM', [], 'UnwrapE', []);

for x = 1:length(Grunts) 
Gruntmod(x).UnwrapS = logfGrunts(x).Unwrap;
Gruntmod(x).UnwrapM = logfGrunts(x).Unwrap;
Gruntmod(x).UnwrapE = logfGrunts(x).Unwrap;
end

for n = 1:31
    StartLFS = logfsgram(P(n).StartData,512,50000);
    CoosTrim(n).StartLFS = StartLFS;
    MiddleLFS = logfsgram(P(n).MiddleData,512,50000);
    CoosTrim(n).MiddleLFS = MiddleLFS;
    EndLFS = logfsgram(P(n).EndData,512,50000);
    CoosTrim(n).EndLFS = EndLFS;
    UnwrapS = reshape(CoosTrim(n).StartLFS,1,size(CoosTrim(n).StartLFS,1)*size(CoosTrim(n).StartLFS,2));
    CoosTUnwrap(n).UnwrapS = UnwrapS;
    UnwrapM = reshape(CoosTrim(n).MiddleLFS,1,size(CoosTrim(n).MiddleLFS,1)*size(CoosTrim(n).MiddleLFS,2));
    CoosTUnwrap(n).UnwrapM = UnwrapM;
    UnwrapE = reshape(CoosTrim(n).EndLFS,1,size(CoosTrim(n).EndLFS,1)*size(CoosTrim(n).EndLFS,2));
    CoosTUnwrap(n).UnwrapE = UnwrapE;
end

CG = [CoosTUnwrap Gruntmod];

%Now for comparison of Grunts directly to Coos
logRHO = struct('Start', [], 'Middle', [], 'End', []);
DislogRHO = struct('Start', [], 'Middle', [], 'End', []);

R = struct('rS', [], 'rM', [], 'rE', []);
for i = 1:length(CG)
    for j = 1:length(CG)
        rS = corrcoef(CG(i).UnwrapS,CG(j).UnwrapS);
        R(i,j).rS = rS;
        logRHO.Start(i,j) = R(i,j).rS(2);
        rM = corrcoef(CG(i).UnwrapM, CG(j).UnwrapM);
        R(i,j).rM = rM;
        logRHO.Middle(i,j) = R(i,j).rM(2);
        rE = corrcoef(CG(i).UnwrapE, CG(j).UnwrapE);
        R(i,j).rE = rE;
        logRHO.End(i,j) = R(i,j).rE(2);
        
        DislogRHO.Start(i,j) = 1 - logRHO.Start(i,j);
        DislogRHO.Middle(i,j) = 1 - logRHO.Middle(i,j);
        DislogRHO.End(i,j) = 1 - logRHO.End(i,j);
    end
end

logRHOS = struct('logRHOC', [], 'logRHOG', [], 'DislogRHOC', [], 'DislogRHOG', []);
logRHOS.logRHOC = logRHOC;
logRHOS.logRHOG = logRHOG;
logRHOS.DislogRHOC = DislogRHOC;
logRHOS.DislogRHOG = DislogRHOG;
end

if nargout > 3 || nargout < 1 %lets me decide whether or not I want the figures to be displayed, or if I just want the data.
    figure; figures = imagesc(logRHOS.logRHOC);
    xlabel('Voc #','fontsize', 14)
    ylabel('Voc #','fontsize', 14)
    title('Coos log-frequency spectrogram similarity matrix')
    colormap Cool
    colorbar; caxis([0 1]);
    print('-djpeg','-r400','SimMatLFSC')
    
    figure(2); imagesc(logRHOS.logRHOG);
    xlabel('Voc #','fontsize', 14)
    ylabel('Voc #','fontsize', 14)
    title('Grunts log-frequency spectrogram similarity matrix')
    colormap Cool
    colorbar; caxis([0 1]);
    print('-djpeg','-r400','SimMatLFSG')
    %{
    figure(3); imagesc(logRHO.Start);
    xlabel('Coo Voc #','fontsize', 14)
    ylabel('Grunt Voc #','fontsize', 14)
    title('Coos vs Grunts (Start)');
    colormap Cool
    colorbar; caxis([0 1]);

    figure(4); imagesc(logRHO.Middle);
    xlabel('Coo Voc #','fontsize', 14)
    ylabel('Grunt Voc #','fontsize', 14)
    title('Coos vs Grunts (Middle)');
    colormap Cool
    colorbar; caxis([0 1]);

    figure(5); imagesc(logRHO.End);
    xlabel('Coo Voc #','fontsize', 14)
    ylabel('Grunt Voc #','fontsize', 14)
    title('Coos vs Grunts (End)');
    colormap Cool
    colorbar; caxis([0 1]);
    %}
    
    MDS(logRHOS.DislogRHOC, 6, 'Coos LFS');
    view(-29,-44);
    print('-djpeg','-r400','MDSLFSC')
    MDS(logRHOS.DislogRHOG, 7, 'Grunts LFS');
    view(-29,-44);
    print('-djpeg','-r400','MDSLFSG')
    %{
    MDS(DislogRHO.Start, 8, 'Grunts and Coos MDS (Start)');
    view(-29,-44);
    MDS(DislogRHO.Middle, 9, 'Grunts and Coos MDS (Middle)');
    view(-29,-44);
    MDS(DislogRHO.End, 10, 'Grunts and Coos MDS (End)');
    view(-29,-44);
    %}
end

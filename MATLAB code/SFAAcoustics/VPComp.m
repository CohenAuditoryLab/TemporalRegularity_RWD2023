%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  Calculate spike train distances
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%Extract the spike trains for the metrics.
Hold = cell(size(allspiketimes,2),size(allspiketimes,3));
Check = cell(size(allspiketimes,2),size(allspiketimes,3));
for i = 1:size(allspiketimes,2) %# of trials
    for j = 1:size(allspiketimes,3) %# of neurons
        a = allspiketimes(:,i,j); b = allspiketimes(:,i,j);
        len = find(isnan(a),1,'first');
        if FR(i,j) > 0
            a = a(1:len-1); b = b(1);
        elseif isnan(FR(i,j))
            a = NaN; b = NaN;
        else a = []; b = 'zero';
        end
        Hold{i,j} = a;
        Check{i,j} = b;
    end
end

%For some reason the units seem to switch from s to ms, so this is to
%convert those back to s
for j = 68:246 
    for i = 1:size(allspiketimes,2)
        Hold{i,j} = Hold{i,j}./1000;
    end
end

%Calculate the victor/purpura distance for each pair of spike trains for each
%neuron. This takes a while to run. (VP1000A includes comparison of those
%w/ FR = 0)

%VP1000 = cell(1,size(allspiketimes,3));
VP0 = cell(1,size(allspiketimes,3)); %Pure rate code

for j = 1:size(allspiketimes,3)
    %VP1000{1,j} = NaN*ones(size(allspiketimes,2),size(allspiketimes,2));
end

for j = 1:size(allspiketimes,3) %# of neurons
    for i = 1:size(allspiketimes,2) %# of trials
        for k = 1:size(allspiketimes,2) %# of trials
            if ~isnan(Check{i,j}) & ~ischar(Check{i,j}) %Use both constraints to consider comparison of empty spike trains to be NaN.
                if ~isnan(Check{k,j}) & ~ischar(Check{k,j}) %Use both constraints to consider comparison of empty spike trains to be NaN.
                    VP0{j}(i,k) = spkd(Hold{i,j}, Hold{k,j},0);
                    %VP1000{j}(i,k) = spkd(Hold{i,j}, Hold{k,j},1000);
                end
            end
        end
    end
end
 
%Organize the VP distances of the comparisons of identical test stimuli
%on different trials.
VPHold0 = cell(2,size(allspiketimes,3));
Store = cell(2,size(allspiketimes,3));
for j = 1:size(allspiketimes,3)
    %VPHold0{1,j} = NaN*ones(size(allspiketimes,2),size(allspiketimes,2));
    %VPHold0{2,j} = NaN*ones(size(allspiketimes,2),size(allspiketimes,2));
    %Store{1,j} = NaN*ones(size(allspiketimes,2),size(allspiketimes,2));
    Store{2,j} = NaN*ones(size(allspiketimes,2),size(allspiketimes,2));
end

for j = 1:size(allspiketimes,3)
    for n = 1:size(allspiketimes,2)
        for m = (n+1):size(allspiketimes,2)
            if ~isnan(Stimulus(n,j)) && Stimulus(n,j) == Stimulus(m,j)
                if floor(Stimulus(n,j)) == Stimulus(n,j)
                    VPHold0{1,j}(n,m) = VP0{1,j}(n,m);
                    Store{1,j}(n,m) = Stimulus(n,j);
                else VPHold0{2,j}(n,m) = VP0{1,j}(n,m);
                    Store{2,j}(n,m) = floor(Stimulus(n,j));
                end
            end
        end
    end
end

VPNull = struct('C', [], 'G', []);
VPNull.C = cell(length(C.Entropy),size(allspiketimes,3)); VPNull.G = cell(length(G.Entropy),size(allspiketimes,3));

for j = 1:size(allspiketimes,3)
    for k = 1:length(C.Entropy)
        indx = find(Store{1,j} == k);
        VPNull.C{k,j} = VPHold0{1,j}(indx);
    end
    for k = 1:length(G.Entropy)
        indx = find(Store{2,j} == k);
        VPNull.G{k,j} = VPHold0{2,j}(indx);
    end
end

%Mean of VP distance across vocalizations, then across neurons.
MeanVP1000A = struct('CMean',[], 'CSD', [], 'CMeanAll', [], 'CSDAll',[],'GMean',[],'GSD',[],'GMeanAll',[],'GSDAll',[]);
    %Across vocalizations
for j = 1:size(allspiketimes,3)
    for k = 1:length(C.Entropy)
        MeanVP1000A.CMean(k,j) = nanmean(VPNull.C{k,j});
        MeanVP1000A.CSD(k,j) = nanstd(VPNull.C{k,j});
    end
    for k = 1:length(G.Entropy)
        MeanVP1000A.GMean(k,j) = nanmean(VPNull.G{k,j});
        MeanVP1000A.GSD(k,j) = nanstd(VPNull.G{k,j});
    end
end
    %Across neurons
for k = 1:length(C.Entropy)
    MeanVP1000A.CMeanAll(k) = nanmean(MeanVP1000A.CMean(k,:));
    MeanVP1000A.CSDAll(k) = nanstd(MeanVP1000A.CMean(k,:));
end
for k = 1:length(G.Entropy)
    MeanVP1000A.GMeanAll(k) = nanmean(MeanVP1000A.GMean(k,:));
    MeanVP1000A.GSDAll(k) = nanstd(MeanVP1000A.GMean(k,:));
end
%%
%For comparison of different voc's (Cx,Cy/Gx,Cy where x~=y)
Ratio = cell(2,size(allspiketimes,3)); SHC = cell(557,557); SHG = cell(557,557);
for j = 1:size(allspiketimes,3)
    Ratio{1,j} = NaN*ones(size(allspiketimes,2),size(allspiketimes,2));
    Ratio{2,j} = NaN*ones(size(allspiketimes,2),size(allspiketimes,2));
end

for j = 1:size(allspiketimes,3)
    for n = 1:size(allspiketimes,2)
        for m = (n+1):size(allspiketimes,2)
            if ~isnan(Stimulus(n,j)) && ~isnan(Stimulus(m,j)) && Stimulus(n,j) ~= Stimulus(m,j)
                if floor(Stimulus(n,j)) == Stimulus(n,j) && floor(Stimulus(m,j)) == Stimulus(m,j)
                    Ratio{1,j}(n,m) = C.Duration(max(Stimulus(n,j),Stimulus(m,j)))./C.Duration(min(Stimulus(n,j),Stimulus(m,j))); %Look at max/min because want the ratio to be symmetric like the VP distance.
                elseif floor(Stimulus(n,j)) ~= Stimulus(n,j) && floor(Stimulus(m,j)) ~= Stimulus(m,j)
                    Ratio{2,j}(n,m) = G.Duration(max(floor(Stimulus(n,j)),floor(Stimulus(m,j))))./G.Duration(min(floor(Stimulus(n,j)),floor(Stimulus(m,j))));
                end
            end
        end
    end
end
D = cell(2,size(allspiketimes,3));
%For each neuron, take mean VP Distance for each unique stimulus pair.
for i = 1:2
    for j = 1:size(allspiketimes,3)
        SH = cell(size(allspiketimes,2),size(allspiketimes,2));
        if isnan(nanmean(nanmean(Ratio{i,j})))%If there were no Cx,Cy/Gx,Gy stimulus pairs for this neuron
            D{i,j} = [0 0 0 0]; 
        else %[RNVR, ~, POSCell] = repval(Ratio{i,j});
            for n = 1:size(allspiketimes,2)
                for m = 1:size(allspiketimes,2)
                    if ~isnan(Ratio{i,j}(n,m))
                        Stims = {[max(Stimulus(n,j), Stimulus(m,j)),min(Stimulus(n,j), Stimulus(m,j))]};
                        SH{n,m} = num2str(Stims{1});
                    else SH{n,m} = num2str(0);
                    end
                end
            end
            [RNVR, POSCell] = repvalstr(SH);
            for t = 1:size(RNVR,1)
                D{i,j}(t,1) = nanmean(VP0{1,j}(POSCell{t}));%Will get different answers depending on if you use VP1000 and VP1000A
                D{i,j}(t,2) = nanstd(VP0{1,j}(POSCell{t}));
                [a,b] = find(strcmp(SH,RNVR{t,1}),1,'first');
                D{i,j}(t,3) = max(Stimulus(a,j),Stimulus(b,j));
                D{i,j}(t,4) = min(Stimulus(a,j),Stimulus(b,j));
            end
        end
    end
end

for i = 1:2
    Cat{i} = [D{i,1}(:,3) D{i,1}(:,4)];
    S{i} = D{i,1}(:,1);
    for j = 2%:size(allspiketimes,3) 
        Cat{i} = [Cat{i}; [D{i,j}(:,3) D{i,j}(:,4)]];
        S{i} = [S{i}; D{i,j}(:,1)];
    end
end

for i = 1:2
    Store = cell(1,size(Cat{i},1));
    for z = 1:size(Cat{i},1)
        Stims = {[Cat{i}(z,1) Cat{i}(z,2)]};
        Store{z} = num2str(Stims{1});       
    end
    Store = Store'; %Make sure it's a column vector for repvalstr function.
    [RNVR,POS] = repvalstr(Store);
    for t = 1:size(RNVR,1)
        MeanVPA{i}(t,1) = nanmean(S{i}(POS{t}));
        MeanVPA{i}(t,2) = nanstd(S{i}(POS{t}));
        MeanVPA{i}(t,3) = floor(Cat{i}(POS{t}(1),1));
        MeanVPA{i}(t,4) = floor(Cat{i}(POS{t}(1),2));
    end
end
%For calculating the ratio (easier access)
CName = fieldnames(C); 
for i = 1:length(CName)
    for t = 1:size(MeanVPA{1},1)
        MeanVPA{1}(t,i+4) = max(C.(CName{i})(MeanVPA{1}(t,3)),C.(CName{i})(MeanVPA{1}(t,4)))./min(C.(CName{i})(MeanVPA{1}(t,3)),C.(CName{i})(MeanVPA{1}(t,4)));
        MeanVPA{1}(t,i+5) = RHOC(MeanVPA{1}(t,3),MeanVPA{1}(t,4));
        MeanVPA{1}(t,i+6) = logRHOS.logRHOC(MeanVPA{1}(t,3),MeanVPA{1}(t,4));
    end
end

GName = fieldnames(G);
for i = 1:length(GName)
    for t = 1:size(MeanVPA{2},1)
        MeanVPA{2}(t,i+4) = max(G.(GName{i})(MeanVPA{2}(t,3)),G.(GName{i})(MeanVPA{2}(t,4)))./min(G.(GName{i})(MeanVPA{2}(t,3)),G.(GName{i})(MeanVPA{2}(t,4)));
        MeanVPA{2}(t,i+5) = RHOG(MeanVPA{2}(t,3),MeanVPA{2}(t,4));
        MeanVPA{2}(t,i+6) = logRHOS.logRHOG(MeanVPA{2}(t,3),MeanVPA{2}(t,4));
    end
end


%%
%%%%%%%%%%%%%%%%%%%%%%%%
%  Plot VP Distances
%%%%%%%%%%%%%%%%%%%%%%%%
%%%%
%Plot for (Cx,Cx)/(Gx,Gx) comparison
%%%%
MeanVPA = []; %Don't actually pre-allocate. This is just so I can change the names throughout the script all at once.
figure;errorbar(MeanVP1000A.CMeanAll,MeanVP1000A.CSDAll,'.k');
title('Coos Mean VP Distance'); xlabel('Voc #'); ylabel('VP Distance')
figure;errorbar(MeanVP1000A.GMeanAll,MeanVP1000A.GSDAll,'.k');
title('Grunts Mean VP Distance'); xlabel('Voc #'); ylabel('VP Distance')
%%%%
%Plot for (Cx,Cy)/(Gx,Gy) comparison
%%%%
CName = fieldnames(C); GName = fieldnames(G);
CNames = {'Center of Gravity';'Mean Harmonicity';'SD of Intensity'; ...
'Duration'; 'Entropy'; 'Mod Spec'; 'Log Freq Spec'}; 
GNames = {'Center of Gravity'; 'SD of Intensity';'Duration'; 'Entropy';'Mod Spec';'Log Freq Spec'}; 
%3D plot with VP and then feature value for both stimuli
for x = 1:numel(CName)
    figure; %errorbar3( C.(CName{x})(MeanVP0{1}(:,3)), C.Duration(MeanVP0{1}(:,4)),MeanVP0{1}(:,1),MeanVP0{1}(:,2)); 
    plot(C.(CName{x})(MeanVPA{1}(:,3)), C.Duration(MeanVPA{1}(:,4)),MeanVPA{1}(:,1),'.k')
    title('Coos Mean VP Distance'); xlabel(CNames{x}); ylabel(CNames{x}); zlabel('VP Distance');
end
for x = 1:numel(GName)
    figure; %errorbar3( G.(GName{x})(MeanVP0{2}(:,3)), G.(GName{x})(MeanVP0{2}(:,4)),MeanVP0{2}(:,1),MeanVP0{2}(:,2));
    plot(G.(GName{x})(MeanVPA{2}(:,3)), G.(GName{x})(MeanVPA{2}(:,4)),MeanVPA{2}(:,1),'.k')
    title('Grunts Mean VP Distance'); xlabel(GNames{x}); ylabel(GNames{x}); zlabel('VP Distance');
end

%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   PLOT VP DISTANCE VERSUS RATIO VALUES. 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%(Cx,Cy)/(Gx,Gy) comparison versus ratios and calculate
%correlation.

CName = fieldnames(C); GName = fieldnames(G);
CNames = {'Center of Gravity';'Mean Harmonicity';'SD of Intensity'; ...
'Duration'; 'Entropy'; 'Mod Spec'; 'Log Freq Spec'}; 
GNames = {'Center of Gravity'; 'SD of Intensity';'Duration'; 'Entropy';'Mod Spec';'Log Freq Spec'}; 

%Use MeanVP for q = 1000 and MeanVP0 for q = 0.
%The ratio values are contained within the cells themselves.

R = struct('C',[],'G',[]);
for x = 1:numel(CNames)
    figure(3); 
    subplot(3,3,x); scatter(MeanVP{1}(:,x+4),MeanVP{1}(:,1),'+k')
    axis('square')
    %title('Coos'); ylabel('Mean VP Distance'); 
    xlabel(['Ratio of ' CNames{x}])  
    [a,~] = find(~isnan(MeanVP0{1}(:,1)));
    [r,q] = corrcoef(MeanVP0{1}(a,1),MeanVP0{1}(a,x+4));
    R.C(x,1) = r(2); R.C(x,2) = q(2);
end
%print('-djpeg','-r400','MeanVPC0')
for x = 1:numel(GNames)
    figure(4); 
    subplot(2,3,x);scatter(MeanVP{2}(:,x+4),MeanVP{2}(:,1),'+k')
    %title('Grunts','fontsize',16); ylabel('Mean VP Distance','fontsize',14); 
    xlabel(['Ratio of ' GNames{x}])
    [a,~] = find(~isnan(MeanVP0{2}(:,1)));
    [r,q] = corrcoef(MeanVP0{2}(a,1),MeanVP0{2}(a,x+4));
    R.G(x,1) = r(2); R.G(x,2) = q(2);
end
%print('-djpeg','-r400','MeanVPG0')

%%

%%%%
%Plot comparison matrix for the spike train distance
%%%%
PilotA = NaN*ones(73,73);
for i = 1:size(MeanVPA{1},1)
    PilotA(MeanVPA{1}(i,3),MeanVPA{1}(i,4)) = MeanVPA{1}(i,1);
    PilotA(MeanVPA{1}(i,4),MeanVPA{1}(i,3)) = MeanVPA{1}(i,1);
end
for i = 1:length(MeanVP1000A.CMeanAll)
    PilotA(i,i) = MeanVP1000A.CMeanAll(i);
end
for i = 1:length(MeanVP1000A.GMeanAll)
    PilotA(i+31,i+31) = MeanVP1000A.GMeanAll(i);
end
for i = 1:size(MeanVPA{2},1)
    PilotA(floor(MeanVPA{2}(i,3))+31,floor(MeanVPA{2}(i,4))+31) = MeanVPA{2}(i,1);
    PilotA(floor(MeanVPA{2}(i,4))+31,floor(MeanVPA{2}(i,3))+31) = MeanVPA{2}(i,1);
end
PilotA(isnan(PilotA)) = max(max(PilotA)) + 10; %To distinguish from 0. May need to change added number, but in this case 10 is enough. 

imagesc(PilotA); colormap(vortex); colorbar
xlabel('Voc # (Coos: 1 - 31; Grunts: 32 - 73)','fontsize',14); ylabel('Voc # (Coos: 1 - 31; Grunts: 32 - 73)','fontsize',14)
title('VP Distance q = 0','fontsize',16)


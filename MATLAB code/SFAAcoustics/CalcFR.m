%%%%%%%%%%%%%%%%%%%%%%%%%%
%  Calulate Firing Rate  
%%%%%%%%%%%%%%%%%%%%%%%%%%

FR = zeros(size(allspiketimes,2),size(allspiketimes,3));
%Count number of spikes per trial for each neuron
for i = 1:size(allspiketimes,2) %# of trials
    for j = 1:size(allspiketimes,3) %# of neurons
        a = allspiketimes(:,i,j);
        spike = find(~isnan(a)); %This is the number of spikes
        if isempty(spike) %If there are no spikes
            FR(i,j) = 0;
        else FR(i,j) = size(spike,1);
        end
    end
end


%Determine #, type, and duration of test stimulus for each trial for each
%neuron.
Duration = zeros(size(alltrialsindex,2),size(alltrialsindex,3));
Stimulus = zeros(size(alltrialsindex,2),size(alltrialsindex,3));

for i = 1:size(alltrialsindex,2) %# of trials
    for j = 1:size(alltrialsindex,3)%# of neurons
        b = alltrialsindex(1,i,j);
        c = alltrialsindex(2,i,j);
        if c == 66 %Test stimulus is voc #1 of stimulus type different than reference
            if b == 1 %Reference stimulus was a Coo
                Duration(i,j) = G.Duration(1);
                Stimulus(i,j) = 1.1; %Distinguish Coos from Grunts by multiplying Grunt voc # by making it a non-integer. 
            else Duration(i,j) = C.Duration(1); Stimulus(i,j) = 1;
            end
        elseif c ~= 66 && ~isnan(c) %Test stimulus was same type as reference stimulus
            if b == 1 %Reference stimulus was a Coo
                Duration(i,j) = C.Duration(c);
                Stimulus(i,j) = c;
            else Duration(i,j) = G.Duration(c); Stimulus(i,j) = c + 0.1; %Add 0.1 to distinguish grunts from coos
            end
        else Duration(i,j) = NaN; Stimulus(i,j) = NaN; %NaN implies there was no stimulus presented for that neuron on that trial.
        end
    end
end

%Calculate FR (spikes/second) for each trial for each neuron.
for i = 1:size(FR,1)
    for j = 1:size(FR,2)
        if ~isnan(Duration(i,j)) %If there was a stimulus for that trial for that neuron.
            FR(i,j) = FR(i,j)./Duration(i,j);
        else FR(i,j) = NaN;
        end
    end
end

%Pre-allocate cell array
FRC = cell(1,size(allspiketimes,3)); 
FRG = cell(1,size(allspiketimes,3));
F = cell(2,size(allspiketimes,3));

%Pre-allcate memory of cell array
for j = 1:size(allspiketimes,3)
%FRC{1,j} = NaN*ones(size(allspiketimes,2),2); FRG{1,j} = NaN*ones(size(allspiketimes,2),2); 
F{1,j} = NaN*ones(size(allspiketimes,2),3); F{2,j} = NaN*ones(size(allspiketimes,2),3);
end

for j = 1:size(allspiketimes,3) %# of neurons
    for i = 1:size(allspiketimes,2) %# of trials 
        if ~isnan(Stimulus(i,j))
            if floor(Stimulus(i,j)) == Stimulus(i,j)
                FRC{j}(i,1) = FR(i,j); FRC{j}(i,2) = Stimulus(i,j);                
            else FRG{j}(i,1) = FR(i,j); FRG{j}(i,2) = floor(Stimulus(i,j)); 
            end
        end
    end
end

%Trim matrix arrays (although this might not be necessary)
for j = 1:size(allspiketimes,3)
    [a,~] = find(isnan(FRC{j}(:,1)));
    FRC{j}(a,:) = [];
    [b,~] = find(isnan(FRG{j}(:,1)));
    FRG{j}(b,:) = [];
end

%Calculate average firing rate and z-score of firing rate for each neuron
%for each vocalization.
for j = 1:size(allspiketimes,3)
    %For Coos
    MeanC = mean(FRC{j}(:,1));
    sdC = std(FRC{j}(:,1));
    for y = 1:size(FRC{j},1)
        FRC{j}(y,1) = abs((FRC{j}(y,1) - MeanC)./sdC);
    end
    [RNVRC,POSC] = repval(FRC{1,j}(:,2));   
    for i = 1:size(RNVRC,1)
        F{1,j}(i,1) = mean(FRC{1,j}(POSC{i}));
        F{1,j}(i,2) = std(FRC{1,j}(POSC{i}));
        F{1,j}(i,3) = RNVRC(i,1);
    end
    %For Grunts
    MeanG = mean(FRG{j}(:,1));
    sdG = std(FRG{j}(:,1));
    for y = 1:size(FRG{j},1)
        FRG{j}(y,1) = abs((FRG{j}(y,1) - MeanG)./sdG);
    end
    [RNVRG,POSG] = repval(FRG{1,j}(:,2));
    for i = 1:size(RNVRG,1)
        F{2,j}(i,1) = mean(FRG{1,j}(POSG{i}));
        F{2,j}(i,2) = std(FRG{1,j}(POSG{i}));
        F{2,j}(i,3) = RNVRG(i,1);
    end
end

MatF = F{1,1}(:,3)'; MatG = F{2,1}(:,3)';
for j = 1:size(allspiketimes,3)-1   
    MatF = [MatF; F{1,j+1}(:,3)'];
    MatG = [MatG; F{2,j+1}(:,3)'];
end

[RNVRC, ~] = repval(MatF); [RNVRG, ~] = repval(MatG);
[a,~] = find(isnan(RNVRC),1,'first');
RNVRC(a:end,:) = [];
[b,~] = find(isnan(RNVRG),1,'first');
RNVRG(b:end,:) = [];

HC = cell(1,length(RNVRC)); HG = cell(1,length(RNVRG));
for i = 1:length(HC)
    HC{1,i} = NaN*ones(2,size(allspiketimes,3));
end
for i = 1:length(HG)
    HG{1,i} = NaN*ones(2,size(allspiketimes,3));
end

for i = 1:length(RNVRC)
    for j = 1:size(allspiketimes,3)
        for k = 1:size(allspiketimes,2)
            if F{1,j}(k,3) == RNVRC(i,1)
                HC{1,i}(1,j) = F{1,j}(k,1); %Average z-zcore for voc in index i of RNVRC for neuron j
            end
        end
    end
end

for i = 1:length(RNVRG)
    for j = 1:size(allspiketimes,3)
        for k = 1:size(allspiketimes,2)
            if F{2,j}(k,3) == RNVRG(i,1)
                HG{1,i}(1,j) = F{2,j}(k,1);
            end
        end
    end
end

% Has SD, Mean FR, and voc #
for i = 1:length(HC)
    MeanFRC(i,1) = nanmean(HC{1,i}(1,:));
    MeanFRC(i,2) = nanstd(HC{1,i}(1,:));
    MeanFRC(i,3) = RNVRC(i,1); %Voc #
end

for i = 1:length(HG)
    MeanFRG(i,1) = nanmean(HG{1,i}(1,:));
    MeanFRG(i,2) = nanstd(HG{1,i}(1,:));
    MeanFRG(i,3) = RNVRG(i,1);
end

%%

%%%%%%%%%%%%%%%%%%%%%%%
%  FR ANOVA
%%%%%%%%%%%%%%%%%%%%%%%
%function [CorrC, CorrG] = FRanov(FRC,FRG,C,G);
A = cell(2,size(allspiketimes,3));
for j = 1:size(allspiketimes,3)
    A{1,j} = NaN*ones(size(allspiketimes,2),length(C.Duration));
    A{2,j} = NaN*ones(size(allspiketimes,2),length(G.Duration));
end

for j = 1:size(allspiketimes,3)
    A{1,j}(1,1:length(unique(FRC{j}(:,2)))) = unique(FRC{j}(:,2)); %Identify the unique stimuli presented to the nueron over all stimuli.
    A{2,j}(1,1:length(unique(FRG{j}(:,2)))) = unique(FRG{j}(:,2));
    a = find(isnan(A{1,j}(1,:))); A{1,j}(:,a) = [];
    b = find(isnan(A{2,j}(1,:))); A{2,j}(:,b) = [];
    %For A, top row is stimulus #, below that are the different firing
    %rates for that stimulus.
    Here = FRC{j}(:,1);
    for t = 1:size(A{1,j},2)
       LOC = find(FRC{j}(:,2) == A{1,j}(1,t)); 
       A{1,j}(2:length(LOC)+1,t) = Here(LOC);
    end
    Here2 = FRG{j}(:,1);
    for t = 1:size(A{2,j},2)
       LOC2 = find(FRG{j}(:,2) == A{2,j}(1,t)); 
       A{2,j}(2:length(LOC2)+1,t) = Here2(LOC2);
    end
    a = find(isnan(A{1,j}),1,'first'); A{1,j}(a:end,:) = [];
    b = find(isnan(A{2,j}),1,'first'); A{2,j}(b:end,:) = [];
end
%Anova between all the firing rates of the different stimuli per neuron.
for j = 1:size(allspiketimes,3)
    p(1,j) = anova1(A{1,j}(2:end,:),[],'off');        
    p(2,j) = anova1(A{2,j}(2:end,:),[],'off');
end

CName = fieldnames(C); GName = fieldnames(G); 
Corr = struct('Cr', [], 'Cp', [], 'CR2',[],'Gr',[],'Gp', [],'GR2', []);
Corr.Cr = NaN*ones(numel(CName),length(p(1,:))); Corr.CR2 = Corr.Cr; Corr.Cp = Corr.Cr;
Corr.Gr = NaN*ones(numel(CName),length(p(2,:))); Corr.GR2 = Corr.Gr; Corr.Gp = Corr.Gr;

for j = 1:size(allspiketimes,3)
    if p(1,j) < 0.05 %If ANOVA finds a significant relationship, then calculate the correlation coefficient.
        Tmp = A{1,j}(2:end,:);
        for k = 1:numel(CName)
            Feat = C.(CName{k})(A{1,j}(1,:)); 
            [Feat,Ind] = sort(Feat);
            Sorted = Tmp(:,Ind);
            Feature = Sorted;
            for i = 1:size(A{1,j},2)
               Feature(~isnan(Feature(:,i)),i) = Feat(i);
            end
            x = Feature(:); x(isnan(x)) = [];
            y = Sorted(:); y(isnan(y)) = [];
            [r,q] = corrcoef(x,y); Corr.Cr(k,j) = r(2); Corr.Cp(k,j) = q(2);
            %R^2 value for linear regression
            coeff = polyfit(x,y,1);
            f = polyval(coeff,x);
            Corr.CR2(k,j) = max(0,1 - sum((y(:)-f(:)).^2)/sum((y(:)-mean(y(:))).^2));
        end
    end
end

for j = 1:size(allspiketimes,3)
    if p(2,j) < 0.05
        Tmp = A{2,j}(2:end,:);
        for k = 1:numel(GName)
            Feat = G.(GName{k})(A{2,j}(1,:)); 
            [Feat,Ind] = sort(Feat);
            Sorted = Tmp(:,Ind);
            Feature = Sorted;
            for i = 1:size(A{2,j},2)
               Feature(~isnan(Feature(:,i)),i) = Feat(i);
            end
            x = Feature(:); x(isnan(x)) = [];
            y = Sorted(:); y(isnan(y)) = [];
            [r,q] = corrcoef(x,y); Corr.Gr(k+1,j) = r(2); Corr.Gp(k+1,j) = q(2);
            %R^2 value for linear regression
            coeff = polyfit(x,y,1);
            f = polyval(coeff,x);
            Corr.GR2(k+1,j) = max(0,1 - sum((y(:)-f(:)).^2)/sum((y(:)-mean(y(:))).^2)); 
        end
    end
end
Corr.Gr(1,:) = Corr.Gr(2,:);
Corr.Gp(1,:) = Corr.Gp(2,:);
Corr.GR2(1,:) = Corr.GR2(2,:);
Corr.Gr(2,:) = NaN*ones(1,246); Corr.Gp(2,:) = NaN*ones(1,246);Corr.GR2(2,:) = NaN*ones(1,246);

%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  Firing Rate Comparison
%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Sham = cell(1,size(allspiketimes,3));
Wow = cell(1,size(allspiketimes,3));
for j = 1:size(allspiketimes,3)
    [RVNR,POS] = repval(Stimulus(:,j));
    RVNR(isnan(RVNR(:,1)),:) = [];
    for i = 1:size(RVNR,1)
        Sham{j}(i,1) = mean(FR(POS{i},j));
        Sham{j}(i,2) = RVNR(i,1);
    end
    for i = 1:length(Sham{j}) - 1
        for u = (i+1):length(Sham{j})
            o = size(Wow{j},1) + 1;
            Wow{j}(o,1) = abs((Sham{j}(i,1) - Sham{j}(u,1))./(Sham{j}(i,1) + Sham{j}(u,1)));
            Wow{j}(o,2) = max(RVNR(i,1),RVNR(u,1));
            Wow{j}(o,3) = min(RVNR(i,1),RVNR(u,1));
        end
    end
end

Mat = Wow{1}; 
for j = 1:size(allspiketimes,3)-1   
    Mat = [Mat; Wow{j+1}];
end
for i = 1:size(Mat,1)
    MatStr{i,1} = num2str(Mat(i,1));
    MatStr{i,2} = num2str([Mat(i,2),Mat(i,3)]);
end
[RVNR,POS] = repvalstr(MatStr(:,2));
for i = 1:size(RVNR,1)
    FRcomp(i,1) = nanmean(Mat(POS{i},1)); %In some cases where FR was 0 for both, then the Wow value will be Nan.
    FRcomp(i,2) = Mat(POS{i}(1),2);
    FRcomp(i,3) = Mat(POS{i}(1),3);
end

%Identify only the within-type comparisons (Coos vs Coos and Grunts vs
%Grunts)
test{1} = find(floor(FRcomp(:,2)) == FRcomp(:,2) & floor(FRcomp(:,3)) == FRcomp(:,3));
test{2} = find(floor(FRcomp(:,2)) ~= FRcomp(:,2) & floor(FRcomp(:,3)) ~= FRcomp(:,3));
for i = 1:2
    for j = 1:length(test{i})
        FRcmpPlot{i}(j,1) = FRcomp(test{i}(j),1);
        FRcmpPlot{i}(j,2) = floor(FRcomp(test{i}(j),2));
        FRcmpPlot{i}(j,3) = floor(FRcomp(test{i}(j),3));
    end
end

CName = fieldnames(C); 
for i = 1:length(CName)
    for t = 1:size(FRcmpPlot{1},1)
        FRcmpPlot{1}(t,i+3) = max(C.(CName{i})(FRcmpPlot{1}(t,2)),C.(CName{i})(FRcmpPlot{1}(t,3)))./min(C.(CName{i})(FRcmpPlot{1}(t,2)),C.(CName{i})(FRcmpPlot{1}(t,3)));
        FRcmpPlot{1}(t,i+4) = RHOC(FRcmpPlot{1}(t,2),FRcmpPlot{1}(t,3));
        FRcmpPlot{1}(t,i+5) = logRHOS.logRHOC(FRcmpPlot{1}(t,2),FRcmpPlot{1}(t,3));
    end
end

GName = fieldnames(G);
for i = 1:length(GName)
    for t = 1:size(FRcmpPlot{2},1)
        FRcmpPlot{2}(t,i+3) = max(G.(GName{i})(FRcmpPlot{2}(t,2)),G.(GName{i})(FRcmpPlot{2}(t,3)))./min(G.(GName{i})(FRcmpPlot{2}(t,2)),G.(GName{i})(FRcmpPlot{2}(t,3)));
        FRcmpPlot{2}(t,i+4) = RHOG(FRcmpPlot{2}(t,2),FRcmpPlot{2}(t,3));
        FRcmpPlot{2}(t,i+5) = logRHOS.logRHOG(FRcmpPlot{2}(t,2),FRcmpPlot{2}(t,3));
    end
end


%%

%%%%%%%%%%%%%%%%%%%%%%%%%%
%  Probability of Tuning
%%%%%%%%%%%%%%%%%%%%%%%%%%
%Probability that k neurons are selective for feature i.

P = 0.05; %P(i) = Probability that any one neuron is selective for feature i
n = size(allspiketimes,3);
for i = 1:size(Corr.Cp,1)
    neur{1,i} = find(Corr.Cp(i,:) < P);
    %k = numel(find(Corr.Cp(i,:) < P));  %Consider statistically significant correlation values.
    %EXP(1,i) = n*P; %Expected value (what we would expect by chance)
    %Prob(1,i) = nchoosek(n,k)*(P.^k)*((1 - P).^(n - k)); %Probability of having k neurons tuned to the stimulus feature by chance. (???) 
end
for i = 1:size(Corr.Gp,1)
    neur{2,i} = find(Corr.Gp(i,:) < P);
    %k = numel(find(Corr.Gp(i,:) < P));
    %EXP(2,i) = n*P;
    %Prob(2,i) = nchoosek(n,k)*(P.^k)*((1-P).^(n-k));
end
%Find which neurons share tuning for both Coos and Grunts
for i = 1:5
    Tune{i} = repval([neur{1,i} neur{2,i}]',0);
end

for i = 1:5
    if ~isempty(Tune{i})
        for k = 1:size(Tune{i},1)
            Tune{i}(k,3) = Corr.Cr(i,Tune{i}(k,1));
            Tune{i}(k,4) = Corr.Gr(i,Tune{i}(k,1));
            Tune{i}(k,5) = abs(Corr.Cr(i,Tune{i}(k,1)) - Corr.Gr(i,Tune{i}(k,1)));
        end
    end
end
%%

%%%%%%%%%%%%%%%%%%%%%%%%%%
%  Plot Firing Rate
%%%%%%%%%%%%%%%%%%%%%%%%%%  

CName = fieldnames(C); 
CNames = {'Center of Gravity (Hz)';'Mean Harmonicity (dB)';'SD of Intensity (dB)'; ...
    'Duration (s)'; 'Entropy'};    
R = struct('C',[],'G',[]);
for x = 1:numel(CName)
    figure; errorbar(C.(CName{x})(MeanFRC(:,3)),MeanFRC(:,1),MeanFRC(:,2),'dk','MarkerFaceColor','k')
    title('Coos','fontsize',16); ylabel('Firing Rate (spikes/s)','fontsize',14); xlabel(CNames{x},'fontsize',14)
    box off
    [r,q] = corrcoef(C.(CName{x})(MeanFRC(:,3)),MeanFRC(:,1));
    R.C(x,1) = r(2); R.C(x,2) = q(2);
    %print('-djpeg','-r400',['FRC' num2str(x)])
end

GName = fieldnames(G); 
GNames = {'Center of Gravity (Hz)'; 'SD of Intensity (dB)';'Duration (s)'; 'Entropy'}; 

for x = 1:numel(GName)
    figure; errorbar(G.(GName{x})(MeanFRG(:,3)), MeanFRG(:,1), MeanFRG(:,2), 'dk','MarkerFaceColor','k')
    title('Grunts','fontsize',16); ylabel('Firing Rate (spikes/s)','fontsize',14); xlabel(GNames{x},'fontsize',14)
    box off
    [r,q] = corrcoef(G.(GName{x})(MeanFRG(:,3)), MeanFRG(:,1));
    R.G(x,1) = r(2); R.G(x,2) = q(2);
    %print('-djpeg','-r400',['FRG' num2str(x)])
end

%For the FR comparison (a-b)/(a+b)
CNames = {'Center of Gravity';'Mean Harmonicity';'SD of Intensity'; ...
'Duration'; 'Entropy'; 'Mod Spec'; 'Log Freq Spec'}; 
GNames = {'Center of Gravity'; 'SD of Intensity';'Duration'; 'Entropy';'Mod Spec';'Log Freq Spec'}; 
R = struct('C',[],'G',[]);
for x = 1:numel(CNames)
    figure; scatter(FRcmpPlot{1}(:,x+3),FRcmpPlot{1}(:,1),'+k')
    title('Coos','fontsize',16); ylabel('Firing Rate Distance','fontsize',14); xlabel(['Ratio of ' CNames{x}],'fontsize',14)
    box off
    [r,q] = corrcoef(FRcmpPlot{1}(:,x+3),FRcmpPlot{1}(:,1));
    R.C(x,1) = r(2); R.C(x,2) = q(2);   
    %print('-djpeg','-r400',['CompFRC' num2str(x)])
end

for x = 1:numel(GNames)
    figure; scatter(FRcmpPlot{2}(:,x+3),FRcmpPlot{2}(:,1),'+k')
    title('Grunts','fontsize',16); ylabel('Firing Rate Distance','fontsize',14); xlabel(['Ratio of ' GNames{x}],'fontsize',14)
    box off
    [r,q] = corrcoef(FRcmpPlot{2}(:,x+3),FRcmpPlot{2}(:,1));
    R.G(x,1) = r(2); R.G(x,2) = q(2);
    %print('-djpeg','-r400',['CompFRG' num2str(x)])
end


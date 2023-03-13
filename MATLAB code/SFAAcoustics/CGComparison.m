function [RHO, DisRHO] = CGComparison(Grunts, Coos);
%Coos/Grunts are the structs containing the data extracted from wave files along with the
%data trimmed to the length of the shortest stimulus of its class.
%Three possible ways for Grunts and Coos comparisons because they aren't
%the same size

%Trim the Coos to be the same length as the shortest Grunt
L = length(Grunts(1).dataT);
CoosTrim = struct('StartMS', [], 'MiddleMS', [], 'EndMS', []);
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

for n = 1:31
    StartMS = MakeModSpec(P(n).StartData,0);
    CoosTrim(n).StartMS = StartMS;
    MiddleMS = MakeModSpec(P(n).MiddleData,0);
    CoosTrim(n).MiddleMS = MiddleMS;
    EndMS = MakeModSpec(P(n).EndData,0);
    CoosTrim(n).EndMS = EndMS;
    UnwrapS = reshape(CoosTrim(n).StartMS,1,size(CoosTrim(n).StartMS,1)*size(CoosTrim(n).StartMS,2));
    CoosTUnwrap(n).UnwrapS = UnwrapS;
    UnwrapM = reshape(CoosTrim(n).MiddleMS,1,size(CoosTrim(n).MiddleMS,1)*size(CoosTrim(n).MiddleMS,2));
    CoosTUnwrap(n).UnwrapM = UnwrapM;
    UnwrapE = reshape(CoosTrim(n).EndMS,1,size(CoosTrim(n).EndMS,1)*size(CoosTrim(n).EndMS,2));
    CoosTUnwrap(n).UnwrapE = UnwrapE;
end

%In order to concatenate Coos with Grunts the two structs must have the
%same fields.
Gruntmod = struct('UnwrapS', [], 'UnwrapM', [], 'UnwrapE', []);
for i = 1:length(Grunts) 
Gruntmod(i).UnwrapS = Grunts(i).Unwrap;
Gruntmod(i).UnwrapM = Grunts(i).Unwrap;
Gruntmod(i).UnwrapE = Grunts(i).Unwrap;
end

CG = [CoosTUnwrap Gruntmod]; %Concatenate CoosTUnwrap with the Grunts in order to make a square matrix for comparison.

RHO = struct('Start', [], 'Middle', [], 'End', []);
DisRHO = struct('Start', [], 'Middle', [], 'End', []);

R = struct('rS', [], 'rM', [], 'rE', []);
for x = 1:length(CG)
    for y = 1:length(CG)
        rS = corrcoef(CG(x).UnwrapS,CG(y).UnwrapS);
        R(x,y).rS = rS;
        RHO.Start(x,y) = R(x,y).rS(2);
        rM = corrcoef(CG(x).UnwrapM,CG(y).UnwrapM);
        R(x,y).rM = rM;
        RHO.Middle(x,y) = R(x,y).rM(2);
        rE = corrcoef(CG(x).UnwrapE,CG(y).UnwrapE);
        R(x,y).rE = rE;
        RHO.End(x,y) = R(x,y).rE(2);
        
        %Also make dissimilarity matrix for further MDS analysis
        DisRHO.Start(x,y) = 1 - RHO.Start(x,y);
        DisRHO.Middle(x,y) = 1 - RHO.Middle(x,y);
        DisRHO.End(x,y) = 1 - RHO.End(x,y);
    end
end
        
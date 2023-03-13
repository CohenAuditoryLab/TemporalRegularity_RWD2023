%Make a struct containing feature of vocalizations calculated in praat.
C = struct('CoG', [],'MeanHarm', [], 'SDofInt', [], 'Duration', [], 'Entropy', []);

%CoG in Hertz
%SDofInt and MeanHarm in dB
%Duration in s

path(path,'/Volumes/OroroBackup/Jennifer/Cohen Rotation/Code')

%cd 'C:\Users\Jennifer\Desktop\DTW stuff\C'
cd /Volumes/OroroBackup/Jennifer/DTW_stuff
Coos = struct('data', [], 'fs', [], 'dataT', [], 'ModSpec',[],'ModSpecT',[], 'Unwrap', []);

kc = dir('*.wav');
for v = 1:length(kc)
    [data,fs] = wavread(kc(v).name);    
    Coos(v).data = data;
    Coos(v).fs = fs;
    ModSpec = MakeModSpec(data,0);
    Coos(v).ModSpec = ModSpec;    
end

L = zeros(1,31);
for v = 1:31
L(v) = length(Coos(v).data);
end
[c,d] = min(L);
Length = length(Coos(d).data);
for v = 1:length(kc)
    dataT = Coos(v).data(1:Length);
    Coos(v).dataT = dataT;
    ModSpecT = MakeModSpec(dataT,0);
    Coos(v).ModSpecT = ModSpecT;
    Unwrap = reshape(Coos(v).ModSpecT,1,size(Coos(v).ModSpecT,1)*size(Coos(v).ModSpecT,2));
    Coos(v).Unwrap = Unwrap;
end

RHOC = zeros(size(Coos(1).Unwrap,1),size(Coos(1).Unwrap,1));
R = struct('r', []);
for i = 1:length(Coos)
    for j = 1:length(Coos)        
        r = corrcoef(Coos(i).Unwrap,Coos(j).Unwrap);
        R(i,j).r = r;
        RHOC(i,j) = R(i,j).r(2);
    end
end

DisRHOC = 1 - RHOC;

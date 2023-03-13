%Make a struct containing feature of vocalizations calculated in praat.
G = struct('CoG', [], 'SDofInt', [], 'Duration', [], 'Entropy', []);

%CoG in Hertz
%SDofInt in dB
%Duration in s

cd 'C:\Users\Jennifer\Desktop\DTW stuff\G'
Grunts = struct('data', [], 'fs', [], 'dataT', [], 'ModSpec', [], 'ModSpecT', [], 'Unwrap', []);

kg = dir('*.wav');
for v = 1:length(kg)
    [data,fs] = wavread(kg(v).name);    
    Grunts(v).data = data;
    Grunts(v).fs = fs;
    ModSpec = MakeModSpec(data,0);
    Grunts(v).ModSpec = ModSpec;    
end

L = zeros(1,length(kg));
for v = 1:length(kg)
L(v) = length(Grunts(v).data);
end
[c,d] = min(L);
Length = length(Grunts(d).data);
for v = 1:length(kg)
    dataT = Grunts(v).data(1:Length);
    Grunts(v).dataT = dataT;
    ModSpecT = MakeModSpec(dataT,0);
    Grunts(v).ModSpecT = ModSpecT;
    Unwrap = reshape(Grunts(v).ModSpecT,1,size(Grunts(v).ModSpecT,1)*size(Grunts(v).ModSpecT,2));
    Grunts(v).Unwrap = Unwrap;
end

RHOG = zeros(size(Grunts(1).Unwrap,1),size(Grunts(1).Unwrap,1));
R = struct('r', []);
for i = 1:length(Grunts)
    for j = 1:length(Grunts)
        r = corrcoef(Grunts(i).Unwrap,Grunts(j).Unwrap);
        R(i,j).r = r;
        RHOG(i,j) = R(i,j).r(2);
    end
end

DisRHOG = 1 - RHOG;

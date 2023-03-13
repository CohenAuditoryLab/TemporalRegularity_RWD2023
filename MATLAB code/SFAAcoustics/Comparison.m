function [num] = Comparison(C,titlename,num)

for i = 1:length(C.Duration)
    L = C.Duration;
    M = C.SDofInt; 
    A = C.CoG;
    F = C.Entropy;    
    if nargin < 3 %Must enter a number for analysis of grunts.
        H = C.MeanHarm;
    end
end

%Scale the data to be between -1 (smallest) and 1 (largest).
for i = 1:length(C.Duration)       
    R(i) = 2*(L(i) -  min(min(L)))./(max(max(L)) - min(min(L))) - 1;
    N(i) = 2*(M(i) -  min(min(M)))./(max(max(M)) - min(min(M))) - 1;        
    B(i) = 2*(A(i) -  min(min(A)))./(max(max(A)) - min(min(A))) - 1;
    E(i) = 2*(F(i) -  min(min(F)))./(max(max(F)) - min(min(F))) - 1;    
    if nargin < 3
    K(i) = 2*(H(i) -  min(min(H)))./(max(max(H)) - min(min(H))) - 1;
    else K(i) = 0;
    end
end 


figure; scatter(B,N);
xlabel('Center of Gravity (Hz)'); ylabel('SD of Intensity (dB)'); title(titlename)
figure; scatter(B,R);
xlabel('Center of Gravity (Hz)'); ylabel('Duration (s)');title(titlename)
figure; scatter(N,R);
xlabel('SD of Intensity (dB)'); ylabel('Duration (s)');title(titlename)
figure; scatter(E,B);
xlabel('Entropy'); ylabel('Center of Gravity (Hz)');title(titlename)
figure; scatter(E,N);
xlabel('Entropy'); ylabel('SD of Intensity (dB)');title(titlename)
figure; scatter(E,R);
xlabel('Entropy'); ylabel('Duration (s)');title(titlename)

if nargin < 3
    figure; scatter(K,B);
    xlabel('Mean Harmonicity (dB)'); ylabel('Center of Gravity (Hz)');title(titlename)
    figure; scatter(K,N);
    xlabel('Mean Harmonicity (dB)'); ylabel('SD of Intensity (dB)');title(titlename)
    figure; scatter(K,R);
    xlabel('Mean Harmonicity (dB)'); ylabel('Duration (s)');title(titlename)
    figure; scatter(K,E);
    xlabel('Mean Harmonicity (dB)'); ylabel('Entropy');title(titlename)
end
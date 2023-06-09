function cmap = spec_cmap()
% Makes a color map that looks good for spectrograms

cmap = zeros(64,3);
cmap(1,3) = 1.0;

for ib=1:21
    cmap(ib+1,1) = (31+(ib-1)*(12/20))/60;
    cmap(ib+1,2) = ib/21;
    cmap(ib+1,3) = 1.0;
end
for ig=1:21
    cmap(ig+ib+1,1) = (21-(ig-1)*(12/20))/60;
    cmap(ig+ib+1,2) = 1.0;
    cmap(ig+ib+1,3) = 0.5+(ig-1)*(0.3/20);
end
for ir=1:21
    cmap(ir+ig+ib+1,1) = (8-(ir-1)*(7/20))/60;
    cmap(ir+ig+ib+1,2) = 0.5 + (ir-1)*(0.5/20);
    cmap(ir+ig+ib+1,3) = 1;
end
    cmap = hsv2rgb(cmap);
return
function [zDistHist, Order, zOneDHist, OneDOrder] = MDS(EtaSquared,FigNum,TitleName)

D = EtaSquared;
N = size(EtaSquared,1);
     
%%% ----- MDS -----

A       = -0.5*D.^2;
dim     = size( D,1 );
H       = eye(dim) - 1/dim * ones(dim,1) * ones(1,dim);
B       = H * A * H;
[v,d]   = eig(B);
diags   = diag(d);
[val,ind] = sort(diags);
ind     = flipud(ind);
val     = flipud(val);


figure(FigNum);clf
proj=[];
%%% ----- CREATE LOW-DIMENSIONAL (2) REPRESENTATION ----
for k = 1 : 3
   proj(k,:) = v(:,ind(k))' * sqrt(diags(ind(k)));
end

%plot( proj(1,1:N), proj(2,1:N), 'bo' ); hold on;        % GROUP 1
%plot( proj(1,N+1:N2), proj(2,N+1:N2), 'ro' ); hold off; % GROUP 2

hold on; %fit oriented ellipse

x = proj(1,1:N);
y = proj(2,1:N);
z = proj(3,1:N);

%% COMPUTE PRINCIPLE AXIS
mux = mean(x);
muy = mean(y);
muz = mean(z);
x   = x - mux; % zero-mean
y   = y - muy; % zero-mean
z   = z - muz; % zero-mean
P   = [x ; y ; z];
Cov   = P * P'; % covariance matrix
[V,D] = eig(Cov);

[val,ind] = sort( diag(D) );
ind = flipud( ind ); % sort largest to smallest

e1 = V(:,ind(1)); % maximum variance
e1 = e1 / norm(e1); % unit-length
v1 = 1.96*std(P'*e1); % standard deviation % standard deviation 95% CI

e2 = V(:,ind(2)); % minimum variance
e2 = e2 / norm(e2); % unit-length
v2 = 1.96*std(P'*e2); % standard deviation 95% CI

e3 = V(:,ind(3)); % minimum variance
e3 = e3 / norm(e3); % unit-length
v3 = 1.96*std(P'*e3); % standard deviation 95% CI


%%% MAKE THE ORIENTED ELLIPSOID
[cx,cy,cz] = sphere(40);
cx = v1*cx(:); % scale
cy = v2*cy(:); % scale
cz = v3*cz(:); % scale
Pc = [e1 e2 e3] * [cx' ; cy' ; cz']; % rotate


%%% PLOT DATA AND BOUDNING ELLIPSOID
plot3( P(1,:), P(2,:), P(3,:), 'r.' );

for i=1:size(P,2)
  text(P(1,i), P(2,i), P(3,i),num2str(i))    
end

hold on;
plot3( Pc(1,:), Pc(2,:), Pc(3,:) );
axis equal; 
hold off;
box on;

title(TitleName)

%Plots the first dimension (gives hierarchy seen in dendrogram)
%qwerty = zeros(1,length(P(1,:)));
%figure(2); scatter(P(1,:), qwerty, '.');
%for i = 1:size(P,2)
%    text(P(1,i), qwerty(i),num2str(i))
%end


%%%%%%%%%%%%%%%%%%%%%
% do this here b/c the data is here...
% calculate center of data and then distance from center
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

x = real(mean(P(1,:)));
y = real(mean(P(2,:)));
z = real(mean(P(3,:)));

DistHist = [];
for i=1:size(P,2)
  DistHist(i) = sign(P(1,i))*sqrt((x-P(1,i))^2 + (y-P(2,i))^2 + (z-P(3,i))^2 );
  FirstDistHist(i) = x - P(1,i); %Just looking at distance in the first dimension.
end

%normalize distances so can compare across analyses
zDistHist = (DistHist-mean(DistHist))/std(DistHist);
zFirstDistHist = (FirstDistHist - mean(FirstDistHist))/std(FirstDistHist);

%take absolute value since don't care about direction just distance
%zDistHist = abs(zDistHist);
%zFirstDistHist = abs(FirstDistHist);

[OneDHist, OneDOrd] = sort(FirstDistHist);
[zOneDHist, OneDOrder] = sort(zFirstDistHist);
[DistHist, Ord] = sort(DistHist);
[zDistHist,Order] = sort(zDistHist);
end


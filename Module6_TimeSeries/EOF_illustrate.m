close all
clear all

%Put your data into a matrix so that the rows indicate temporal development and the columns are corresponding to spatial data points

%make a synthetic spatio-temporal data set
%on a regular grid
x = linspace(-200,200,20);
y = linspace(-200,200,20);
[Xcoord,Ycoord] = meshgrid(x,y);
Xcoord = Xcoord(:);
Ycoord = Ycoord(:);

%regular time intervals
N = 100; %number of times
dt = 1;  %time intervals
t = dt:dt:N*dt;


%noise level
Noise = 0.5;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Quadratic trend
%sinusoidal spatial wavelengths
L1x = 200;
L1y = 300;
L2x = 100;
L2y = 100;


%make spatial pattern
D = rand(1)*cos(2*pi*Xcoord/L1x) + rand(1)*sin(2*pi*Xcoord/L2x);
D = D + rand(1)*cos(2*pi*Ycoord/L1y) + rand(1)*sin(2*pi*Ycoord/L2y);


ts = (t/max(t)).^2;
Dt = (D*ts)';

t1 = ts;
D1 = D; 
D1t = Dt;

%plot spatial and temporal patters for quadratic trend
figure
subplot(121)
imagesc(reshape(D1,20,20))
title('spatial pattern for quadratic trend','fontsize',20)
c=colorbar;
c.FontSize = 20;
ax = gca;
ax.XAxis.FontSize = 24;
ax.YAxis.FontSize = 24;

subplot(122)
imagesc(1:400,t,D1t)
title('spatial-temporal pattern for quadratic trend','fontsize',20)
xlabel('cell number')
ylabel('time')
c=colorbar;
c.FontSize = 20;
ax = gca;
ax.XAxis.FontSize = 20;
ax.YAxis.FontSize = 20;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% periodic trend
%sinusoidal spatial terms
L1x = 400;
L1y = 600;
L2x = 200;
L2y = 200;

D = rand(1)*cos(2*pi*Xcoord/L1x) + rand(1)*sin(2*pi*Xcoord/L2x);
D = D + rand(1)*cos(2*pi*Ycoord/L1y) + rand(1)*sin(2*pi*Ycoord/L2y);

T = 30; %period of harmonic oscillation
ts = rand(1)*cos(2*pi*t/T); 
Dt = (D*ts)';

t2 = ts;
D2 = D; 
D2t = Dt;

figure
subplot(121)
imagesc(reshape(D2,20,20))
title('spatial pattern for quadratic trend','fontsize',20)
c=colorbar;
c.FontSize = 20;
ax = gca;
ax.XAxis.FontSize = 24;
ax.YAxis.FontSize = 24;

subplot(122)
imagesc(1:400,t,D2t)
title('spatial-temporal pattern for quadratic trend','fontsize',20)
xlabel('cell number')
ylabel('time')
c=colorbar;
c.FontSize = 20;
ax = gca;
ax.XAxis.FontSize = 20;
ax.YAxis.FontSize = 20;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%total signal
D = D1t + D2t;
Dtrue = D;

figure
subplot(121)
imagesc(1:400,t,D)
title('spatial-temporal pattern for total signal','fontsize',20)
xlabel('cell number')
ylabel('time')
c=colorbar;
c.FontSize = 20;
ax = gca;
ax.XAxis.FontSize = 20;
ax.YAxis.FontSize = 20;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%add noise
D = D + Noise*randn(size(D));

subplot(122)
imagesc(1:400,t,D)
title('with noise added','fontsize',20)
xlabel('cell number')
ylabel('time')
c=colorbar;
c.FontSize = 20;
ax = gca;
ax.XAxis.FontSize = 20;
ax.YAxis.FontSize = 20;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%plot some time series, randomly sampled
i = randsample(size(D,2),9);
figure
for k=1:9
    subplot(3,3,k)
    plot(t,Dtrue(:,i(k)),'.')
    hold on
    plot(t,D(:,i(k)),'.')
    ax = gca;
    ax.XAxis.FontSize = 15;
    ax.YAxis.FontSize = 15;

end




%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%compute SVD
% Remove the time mean of each column
%D = detrend(D,'constant');

[U,S,V] = svd(D);
EC = U*S;

%EOFs = principal component loading patterns (i.e., principal components)
%ECs = EOF time series (i.e., expansion coefficient time series)
%EOFs = V
%ECs = U*S


figure
imagesc(V)
c=colorbar;
c.FontSize = 20;
title('V = EOFs, columns are spatial basis','fontsize',20)
ax = gca;
ax.XAxis.FontSize = 20;
ax.YAxis.FontSize = 20;

figure
imagesc(EC)
c=colorbar;
c.FontSize = 20;
title('U*S = EOF time series','fontsize',20)
ax = gca;
ax.XAxis.FontSize = 20;
ax.YAxis.FontSize = 20;


%plot first two EOFs (spatial basis) and associated EOF time series

figure
subplot(1,2,1)
imagesc(reshape(V(:,1),20,20))
axis equal
c=colorbar;
c.FontSize = 20;
title('V(:,1) = first spatial basis','fontsize',20)
ax = gca;
ax.XAxis.FontSize = 20;
ax.YAxis.FontSize = 20;

subplot(1,2,2)
plot(t,EC(:,1))
title('EC(:,1) = first EOF time series','fontsize',20)
ax = gca;
ax.XAxis.FontSize = 20;
ax.YAxis.FontSize = 20;



figure
subplot(1,2,1)
imagesc(reshape(V(:,2),20,20))
axis equal
c=colorbar;
c.FontSize = 20;
title('V(:,2) = second spatial basis','fontsize',20)
ax = gca;
ax.XAxis.FontSize = 20;
ax.YAxis.FontSize = 20;

subplot(1,2,2)
plot(t,EC(:,2))
title('EC(:,2) = second EOF time series')
ax = gca;
ax.XAxis.FontSize = 20;
ax.YAxis.FontSize = 20;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% reconstruct data using first p spatial EOFs
p = 50;
Sp = S(1:p,1:p);
Up = U(:,1:p); 
Vp = V(:,1:p);
D_hat = Up*Sp*Vp';

%plot some time series, randomly sampled
i = randsample(size(D,2),9);
figure
for k=1:9
    subplot(3,3,k)
    plot(t,D(:,i(k)),'*')
    hold on
    plot(t,D_hat(:,i(k)),'r-o')
    ax = gca;
    ax.XAxis.FontSize = 15;
    ax.YAxis.FontSize = 15;

    
end
title('Data (blue) and fit (red)')


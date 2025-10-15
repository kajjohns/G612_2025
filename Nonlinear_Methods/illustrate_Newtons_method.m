% Newton's method example
% invert for m1, m2 using simple forward model
% d = sin(w0*m1*x) + m1*m2
% given w0 and x


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%Make synthetic data
N=40;
xmin=0;
xmax=1.0;
Dx=(xmax-xmin)/(N-1);
x = Dx*[0:N-1]';

% true model parameters
mtrue = [1.21, 1.54]';

w0=20;
dtrue = sin(w0*mtrue(1)*x) + mtrue(1)*mtrue(2);
sd=0.4; %data standard deviation
dobs = dtrue + random('Normal',0,sd,N,1);




%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 2D grid, for plotting residuals
L = 101;
Dm = 0.02;
m1min=0;
m2min=0;
m1a = m1min+Dm*[0:L-1]';
m2a = m2min+Dm*[0:L-1]';
m1max = m1a(L);
m2max = m2a(L);

% compute error (sum of squared residuals), E, on grid for plotting 
E = zeros(L,L);
for j = [1:L]
for k = [1:L]
    dpre = sin(w0*m1a(j)*x) + m1a(j)*m2a(k);
    E(j,k) = (dobs-dpre)'*(dobs-dpre);
end
end

figure(2);
hold on;
axis( [m2min, m2max, m1min, m1max] );
axis ij;
imagesc( [m2min, m2max], [m1min, m1max], E);
c=colorbar;
c.FontSize = 15;
xlabel('m2');
ylabel('m1');
plot( mtrue(2), mtrue(1), 'go', 'LineWidth', 3 );
ax = gca;
ax.XAxis.FontSize = 15;
ax.YAxis.FontSize = 15;
title('Sum of Squared Residuals','fontsize',15)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Newton's method, calculate derivatives
% y = sin( w0 m1  x) + m1 m2;
% dy/dm1 = w0 x cos( w0 m1 x) + m2
% dy/dm2 = m2

% initial guess and corresponding error
m0=[1.15,1]';
dhat = sin(w0*m0(1)*x) + m0(1)*m0(2);
Eg = (dobs-dhat)'*(dobs-dhat);
plot( m0(2), m0(1), 'ws', 'LineWidth', 2 , 'markersize', 15);

% save solution and minimum error as a function of iteration number
Niter=20;
Ehis=zeros(Niter+1,1);  %history of errors
m1his=zeros(Niter+1,1); %history of m1 values
m2his=zeros(Niter+1,1); %history of m2 values
%store initial guess
Ehis(1)=Eg;
m1his(1)=m0(1);  
m2his(1)=m0(2);

% iterate to improve initial guess
M = length(mtrue);
G = zeros(N,M);
mg = m0;
for k = [1:Niter]

    dhat = sin( w0*mg(1)*x) + mg(1)*mg(2);
    dd = dobs-dhat;
    Eg=dd'*dd;
    Ehis(k+1)=Eg;
    
    %derivatie matrix
    G = zeros(N,2);
    G(:,1) = w0 * x .* cos( w0 * mg(1) * x ) + mg(2);
    G(:,2) = mg(2)*ones(N,1);
    
    % least squares solution for incremental improvement, dm
    dm = (G'*G)\(G'*dd);
    
    % update
    mg = mg+dm;
    
    %show update on the plot
    plot( mg(2), mg(1), 'wo', 'LineWidth', 2 );
    
    m1his(k+1)=mg(1);
    m2his(k+1)=mg(2);
    
    %show step as a line on the plot
    plot( [m2his(1+k-1), m2his(1+k) ], [m1his(1+k-1), m1his(1+k) ], 'r', 'LineWidth', 2 );
    
end

%final estimate for m1, m2
m1hat = m1his(Niter+1);
m2hat = m2his(Niter+1);

plot( mtrue(2), mtrue(1), 'go', 'LineWidth', 2 );
plot( mg(2), mg(1), 'ro', 'LineWidth', 2 );
    
%plot history of iterations
figure(3);
clf;
subplot(3,1,1);
set(gca,'LineWidth',2);
hold on;
plot( [0:Niter], Ehis, 'k-', 'LineWidth', 2 );
xlabel('iteration');
ylabel('E');
subplot(3,1,2)
set(gca,'LineWidth',2);
hold on;
plot( [0, Niter], [mtrue(1), mtrue(1)], 'r', 'LineWidth', 2 );
plot( [0:Niter], m1his, 'k-', 'LineWidth', 2 );
xlabel('iteration');
ylabel('m_1');
subplot(3,1,3);
set(gca,'LineWidth',2);
hold on;
plot( [0, Niter], [mtrue(2), mtrue(2)], 'r', 'LineWidth', 2 );
plot( [0:Niter], m2his, 'k-', 'LineWidth', 2 );
xlabel('iteration');
ylabel('m_2');
    
% evaluate prediction and plot it


% plot data and fit
figure(1);
hold on;
axis( [0, xmax, 0, 4 ] );
plot(x,dtrue,'b','LineWidth',2);
plot(x,dobs,'ko','LineWidth',2);
xlabel('x');
ylabel('d');

dpre = sin(w0*m1hat*x) + m1hat*m2hat;
plot(x,dpre,'r-','LineWidth',3);
legend('true','data with errors','best-fitting','fontsize',15)


ax = gca;
ax.XAxis.FontSize = 15;
ax.YAxis.FontSize = 15;
grid on
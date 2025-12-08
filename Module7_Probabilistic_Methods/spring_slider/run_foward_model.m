% Simulate evolution of slip after sudden stress change 
% Single spring and slider


% INPUT PARAMETERS
   d_c = 10^-4;                % critical displacement (m)
   A =   0.01;
   B =   0.006;
   sigma = 20;    % normal stress (MPa)
   L=25000;  %radius of slip patch, meters
   v_inf  =   0.03;           % load-point velocity in m/yr
   deltau= 2;   %%instantaneous stress change, MPa
   
   %paramater vector
   X = [d_c,A,B,sigma,L,v_inf,deltau];

   %observation times (years)
   times = linspace(0,10/365/24);


   
  
   u = forward_model(X,times);


%PLOT OUTPUT
plot(times*24*365,u)
xlabel('time, hours');
ylabel('fault slip, meters')
grid on
set(gca,'fontsize',15)

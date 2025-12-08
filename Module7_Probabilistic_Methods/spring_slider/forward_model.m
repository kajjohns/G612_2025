function u = forward_model(X,times)

%return displacement at specified times

  %unwrap X
  d_c = X(1);
  A = X(2);
  B = X(3);
  sigma = X(4);
  L = X(5);
  v_inf = X(6);
  deltau = X(7);


 
   k=3*10^4*.9/L;   %L is radius of circular slip patch in m
   
   mu_0 = 0.6;                % nominal friction coefficient
  
     
% SEISMIC RADIATION DAMPING TERM
   eta = 1.0e-7;       % units of MPa-yr/m
       
   
   % INITIAL CONDITIONS
   %steady state conditions before stress change
   v0      =   v_inf; %initial slip rate
   tau0    =    sigma*(mu_0 + (A-B)*log(v0));
   theta0 = d_c/v0;


   %impose stress change
   tau = tau0+deltau; %new inital stress immediately after stress change
 
    %instantaneous velocity change -- note: theta does not change instantaneosly
   v0=v0*exp(deltau/(A*sigma));
   


  
   %  RUN ODE SOLVER
  
   x0      =   [v0; theta0; tau0; 0*tau];
   const = [d_c; A; B; sigma; k; v_inf; eta];
   opt = [];
 

  t0 = 0;
  tf = times(end);
 [t,x] = ode23('rate_state', [t0 tf],x0,opt,const);

ut = x(:,end);

%interpolate to times
u = interp1(t,ut,times);


  
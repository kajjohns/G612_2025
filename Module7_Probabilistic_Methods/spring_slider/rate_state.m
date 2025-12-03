function [out1, out2, out3] = rate_state(t,x,flag,const)
%%       [out1, out2, out3] = rate_state(t,x,flag,const)
%
% S-file version of fault-constitutive law
%       t = times
%       x = model vector%
%	x(1) is velocity
%	x(2) is state  variable
%	x(3) is shear stress
%
%  unwrap constants
	d_c = const(1);
	A = const(2); 
	B = const(3); 
	sigma = const(4);
	k = const(5); 
	v_inf = const(6);
	eta = const(7);

    
    
 if nargin < 3 | isempty(flag)

    v     = x(1);
	theta = x(2);
	tau   = x(3);

     %return rates
     out1(2) = 1-theta*v/d_c; %Aging Law
     %out1(2) = -v*theta/d_c*log(v*theta/d_c);  %Slip Law
    
     out1(3) = k*(v_inf - v);
     out1(1) =  (out1(3)/sigma - B*out1(2)/theta ) /(eta/sigma + A/v);

     out1(4) = v;  %du/dt = v
    
      
     out1 = out1';
else
      switch(flag)

      case 'init'                       % Return default [tspan,y0,options].
        out1 = [0 200]; %Default tspan
        out2 = [1e-4, 2.15, 172]; % Default y0 here
        out3 = [];              % Default options

        
      otherwise
                error(['Unknown flag ''' flag '''.']);
      end
    end




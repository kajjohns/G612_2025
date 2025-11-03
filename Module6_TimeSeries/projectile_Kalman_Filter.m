
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Generate synthetic data
% Define the system
N = 1000; % number of time steps
dt = 0.001; % Sampling time (s)
t = dt*(1:N); % time vector (s)

include_process_noise = true;
generate_new_data = true;
include_random_walk_noise = true;

if include_process_noise
    

    %% with process noise
    F = [1, dt, 0; 0, 1, 0;0, 0, 1]; % system matrix - state
    G = [-1/2*dt^2; -dt; 0]; % system matrix - input
    H = [1 0 1]; % observation matrix
    tau = 10;
    Q = [0, 0, 0; 0, 0, 0;0, 0, tau^2*dt]; % process noise covariance
    u = 9.80665; % input = acceleration due to gravity (m/s^2)
    I = eye(3); % identity matrix

    %Initialize the state vector (true state)
    y0 = 100; % m  % Define the initial position and velocity
    v0 = 0; % m/s
    xt = zeros(3, N); % True state vector
    xt(:, 1) = [y0; v0; 0]; % True intial state


else
    
        %% without process noise
    F = [1, dt; 0, 1]; % system matrix - state
    G = [-1/2*dt^2; -dt]; % system matrix - input
    H = [1 0]; % observation matrix
    Q = [0, 0; 0, 0]; % process noise covariance
    u = 9.80665; % input = acceleration due to gravity (m/s^2)
    I = eye(2); % identity matrix

    % %Initialize the state vector (true state)
    y0 = 100; % m  % Define the initial position and velocity
    v0 = 0; % m/s
    xt = zeros(2, N); % True state vector
    xt(:, 1) = [y0; v0]; % True intial state



end




% Loop through and calculate the state
for k = 2:N
    % Propagate the states through the prediction equations
    xt(:, k) = F*xt(:, k-1) + G*u;
end


if generate_new_data
% Generate the noisy measurement from the true state
R = .2; % m^2/s^2
v = sqrt(R)*randn(1, N); % measurement noise
z = H*xt + v;% + 3*t; % noisy measurement

if include_random_walk_noise
    %add integrated random walk
    Riw = .002;
    v_iw = cumsum(sqrt(Riw)*randn(1, N));  %integrated white noise (Brownian motion)
    z = z + v_iw;
end

end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Perform the Kalman filter estimation
% Initialize 

if include_process_noise


    %% with process noise
    x = zeros(3, N); % Estimated state vector
    x(:, 1) = [105; -0.02; 0]; % Guess for initial state
    P = [10, 0, 0; 0, 0.01, 0;0 0 0]; % Covariance for initial state error

else
    
    %% without process noise
    x = zeros(2, N); % Estimated state vector
    x(:, 1) = [105; -0.02]; % Guess for initial state
    % Initialize the covariance matrix
    P = [10, 0; 0, 0.01]; % Covariance for initial state error

end


% Loop through and perform the Kalman filter equations recursively
for k = 2:N
    % Predict the state vector
    x(:, k) = F*x(:, k-1) + G*u;
    % Predict the covariance
    P = F*P*F' + Q;
    % Calculate the Kalman gain matrix
    K = P*H'*inv(H*P*H' + R);
    % Update the state vector
    x(:,k) = x(:,k) + K*(z(k) - H*x(:,k));
    % Update the covariance
    P = (I - K*H)*P;
end



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Plot the results
% Plot the states
figure;
subplot(211);
plot(t, z, 'g-', t, x(1,:), 'b--', 'LineWidth', 2);
hold on; plot(t, xt(1,:), 'r:', 'LineWidth', 1.5)
xlabel('t (s)'); ylabel('x_1 = h (m)'); grid on;
legend('Measured','Estimated','True');

if include_process_noise
    %including process noise contribution
    plot(t,x(1,:)+x(3,:),'k')
    legend('Measured','Estimated (projectile)','True','Estimate with process noise');
end



subplot(212);
plot(t, x(2,:), 'b--', 'LineWidth', 2);
hold on; plot(t, xt(2,:), 'r:', 'LineWidth', 1.5)
xlabel('t (s)'); ylabel('x_2 = v (m/s)'); grid on;
legend('Estimated','True');



% Plot the estimation errors
figure;
subplot(211);
plot(t, x(1,:)-xt(1,:), 'm', 'LineWidth', 2)
xlabel('t (s)'); ylabel('\Deltax_1 (m)'); grid on;
subplot(212);
plot(t, x(2,:)-xt(2,:), 'm', 'LineWidth', 2)
xlabel('t (s)'); ylabel('\Deltax_2 (m/s)'); grid on;
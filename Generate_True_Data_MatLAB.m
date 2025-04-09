% Constants
g = 9.81;  % Acceleration due to gravity (m/s^2)
L = 1.0;     % Length of the pendulum (meters)
m = 1.0;     % Mass of the pendulum (kg)
    
% Initial conditions
theta0 = 165 * (pi/180);  % Initial angle (radians)
p0 = 0;        % Initial angular momentum (kg*m^2/s)

% Time span
tspan = [0 10];  % Simulation for 6 seconds
    
% Initial state vector
initial_conditions = [theta0; p0];
    
% Solver options
options = odeset('MaxStep', 0.0004, 'InitialStep', 0.0004);  % Set maximum and initial step sizes
   
% Solve the ODE
[t, y] = ode45(@(t, y) pendulum_ode(t, y, m, g, L), tspan, initial_conditions, options);
    
% Extracting theta and p from the results
theta = y(:, 1);
p = y(:, 2);
    
% Energy calculations
KE = 0.5 * (p.^2) / (m * L.^2);  % Kinetic Energy
PE = m * g * L * (1 - cos(theta));  % Potential Energy
TE = KE + PE;  % Total Energy
    
% Plot the results
figure;
subplot(3, 1, 1);
plot(t, theta);
title('Pendulum Angle vs. Time');
xlabel('Time (s)');
ylabel('Angle (rad)');
    
subplot(3, 1, 2);
plot(t, p);
title('Angular Momentum vs. Time');
xlabel('Time (s)');
ylabel('Angular Momentum (kg*m^2/s)');

subplot(3, 1, 3);
plot(t, TE);
title('Total Energy vs. Time');
xlabel('Time (s)');
ylabel('Total Energy (J)');

% Create a table to save data
data = table(t, theta, p, 'VariableNames', ...
    {'Time', 'Pendulum Angle', 'Angular Momentum'});

% Save data
writetable(data, 'pendulum_simulation_data.csv');

function dydt = pendulum_ode(t, y, m, g, L)
    theta = y(1);
    p = y(2);
    
    % Derivatives
    dtheta_dt = p / (m * L^2);
    dp_dt = -m * g * L * sin(theta);
    
    dydt = [dtheta_dt; dp_dt];
end

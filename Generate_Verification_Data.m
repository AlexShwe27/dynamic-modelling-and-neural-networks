%% Constants

g = 9.81;    % Acceleration due to gravity (m/s^2)
mc = 5.0;    % Mass of the cart (kg)
mp = 2.0;    % Mass of the pendulum (kg)
l = 0.75;    % Length of the pendulum arm (m)

%% Initial Conditions

theta0 = 15 * (pi/180);     % Initial angle (radians)
x0 = 0.0;                   % Initial cart distance (m)
ptheta0 = 0.0;              % Initial angular momentum (kg*m^2/s)
px0 = 0.0;                  % Initial linear momentum (kg*m/s)

% initial state vector
initial_conditions = [theta0; x0; ptheta0; px0];

%% Simulation

% time span
tspan = [0 100];  % Simulation for 20 seconds

% Solver options
options = odeset('MaxStep', 0.0005, 'InitialStep', 0.0005);  % Set maximum and initial step sizes

% Solve the ODE
[t, y] = ode45(@(t, y) CartPole_ode(t, y, mc, mp, g, l), tspan, initial_conditions, options);

%% Results

% Extracting results
theta = y(:, 1);
x = y(:, 2);
ptheta = y(:, 3);
px = y(:, 4);

%% Energy calculations

% Kinetic Energy
KE = (0.5*((mc+mp)/(mc.^2))*(px.^2)) - ((1/(mc*l))*px.*ptheta.*cos(theta)) + (0.5*(1/(mp*(l.^2)))*(ptheta.^2));
% Potential Energy
PE = mp*g*l*cos(theta);
% Total Energy
TE = KE + PE;

%% Plots
figure;
subplot(3, 1, 1);
plot(t, theta);
title('Pendulum Angle vs. Time');
xlabel('Time (s)');
ylabel('Angle (rad)');
    
subplot(3, 1, 2);
plot(t, x);
title('Cart Position vs. Time');
xlabel('Time (s)');
ylabel('Cart Position (m)');

subplot(3, 1, 3);
plot(t, TE);
title('Total Energy vs. Time');
xlabel('Time (s)');
ylabel('Total Energy (J)');

%% Save the simulation data

% Create a table to save data
data = table(t, theta, x, ptheta, px, KE, PE, TE, 'VariableNames', ...
    {'Time', 'Pendulum Angle', 'Cart Position', 'Angular Momentum', 'Linear Momentum', 'Kinetic Energy', 'Potential Energy', 'Total Energy'});

% Save data
writetable(data, 'cartpole_verification_data.csv');

%% ODE for CartPole
function dydt = CartPole_ode(t, y, mc, mp, g, l)
    theta   = y(1);
    x       = y(2);
    ptheta  = y(3);
    px      = y(4);
    
    % Derivatives
    dtheta_dt   = ((1/(mp*(l.^2)))*ptheta) - (1/(mc*l)*cos(theta)*px);
    dx_dt       = (((mc+mp)/(mc.^2))*px) - (1/(mc*l)*cos(theta)*ptheta);
    dptheta_dt  = (mp*g*l*sin(theta)) - ((1/(mc*l))*ptheta*px*sin(theta));
    dpx_dt      = 0;
    
    dydt = [dtheta_dt; dx_dt; dptheta_dt; dpx_dt];
end

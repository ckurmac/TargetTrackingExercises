% First load the data, then plot it 
load("trueTarget.mat");
timeIndices = trueTarget(1,:);
true_XPos = trueTarget(2,:);
true_YPos = trueTarget(3,:);

figure
plot(true_XPos,true_YPos,"Marker","o","MarkerSize",5);
title("True Trajectory of The Target");
ylabel("Y Position(m)");
xlabel("X Position(m)");
xlim([800,2800]);
ylim([200,2200]);
grid on;
%%
% Generate noisy measurements 
sigma_measurement = 100;
noisy_XPos = sigma_measurement*randn(size(true_XPos,1),size(true_XPos,2))+true_XPos;
noisy_YPos = sigma_measurement*randn(size(true_YPos,1),size(true_YPos,2))+true_YPos;

% Plot on top of the true trajectory.
figure 
plot(true_XPos,true_YPos,"Marker","o","MarkerSize",5,'DisplayName','True Trajectory');
hold on
plot(noisy_XPos,noisy_YPos,"LineStyle","none","Marker","x","MarkerSize",8,'DisplayName','Noisy Measurements');
ylabel("Y Position(m)");
xlabel("X Position(m)");
xlim([800,2800]);
ylim([200,2200]);
grid on;

%%
% Implement the Kalman Filter.

x0 = [1000,1000,0,0]; % x y vx vy
P0 = diag([100^2,100^2,10^2,10^2]);
Q = eye(2,2); % Process Noise
R = 100^2*eye(2,2); % Measurement Noise
H = [1,0,0,0;...
     0,1,0,0]; % Measurement matrix
A = @(dt) ([1,0,dt,0;...
           0,1,0,dt;...
           0,0,1,0; ...
           0,0,0,1]); % State update matrix as a function of time.
B = @(dt) ()

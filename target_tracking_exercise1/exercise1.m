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
measured_XPos = sigma_measurement*randn(size(true_XPos,1),size(true_XPos,2))+true_XPos;
measured_YPos = sigma_measurement*randn(size(true_YPos,1),size(true_YPos,2))+true_YPos;

% Plot on top of the true trajectory.
figure
plot(true_XPos,true_YPos,"Marker","o","MarkerSize",5,'DisplayName','True Trajectory');
hold on
plot(measured_XPos,measured_YPos,"LineStyle","none","Marker","x","MarkerSize",8,'DisplayName','Noisy Measurements');
ylabel("Y Position(m)");
xlabel("X Position(m)");
xlim([800,2800]);
ylim([200,2200]);
title("True Trajectory of The Target and Noisy Measurements");
legend("show");
grid on;

%%
% Implement the Kalman Filter.

x0 = [1000;1000;0;0]; % x y vx vy
P0 = diag([100^2,100^2,10^2,10^2]);
Q = eye(2,2); % Process Noise
R = 100^2*eye(2,2); % Measurement Noise
C = [1,0,0,0;...
  0,1,0,0]; % Measurement matrix
A = @(dt) ([1,0,dt,0;...
  0,1,0,dt;...
  0,0,1,0; ...
  0,0,0,1]); % State update matrix as a function of time.
B = @(dt) ([dt^2/2,0;...
  0,dt^2/2;...
  dt,0;...
  0,dt]); % Noise gain matrix as a function of time.
estimated_states = zeros(4,size(timeIndices,2));
predicted_states = zeros(4,size(timeIndices,2));
% initialize the estimator
prev_state = x0;
prev_state_covariance = P0;
prev_time = 0;

for i = 1:size(timeIndices,2)
  current_time = timeIndices(i);
  dt = current_time - prev_time;
  A_k = A(dt);
  B_k = B(dt);
  z_k = [measured_XPos(i);...
    measured_YPos(i)];
  %predict the state and covariance.
  predicted_states(:,i) = A_k*prev_state;
  predicted_covariance = A_k*prev_state_covariance*A_k' + B_k*Q*B_k';
  %measurement update.
  K_k = predicted_covariance*C'*(C*predicted_covariance*C'+R)^-1;% Kalman Gain
  estimated_states(:,i) = predicted_states(:,i)+K_k*(z_k-C*predicted_states(:,i));
  I = eye(4,4);
  estimated_covariance = (I-K_k*C)*predicted_covariance;
  % update k-1|k-1 state and covariance for next step
  prev_state = estimated_states(:,i);
  prev_state_covariance = estimated_covariance;
  prev_time = current_time;
end

% plot the estimated trajectory.
figure
plot(true_XPos,true_YPos,"Marker","o","MarkerSize",5,'DisplayName','True Trajectory');
hold on
plot(measured_XPos,measured_YPos,"LineStyle","none","Marker","x","MarkerSize",8,'DisplayName','Noisy Measurements');
plot(estimated_states(1,:),estimated_states(2,:),"LineStyle","--","Marker","diamond","MarkerSize",5,'DisplayName','Estimated Trajectory');
title("True Trajectory of The Target, Noisy Measurements and Estimated Trajectory");
ylabel("Y Position(m)");
xlabel("X Position(m)");
xlim([800,2800]);
ylim([200,2200]);
legend("show");
grid on;


%  calculate the estimation and prediction errors and plot w.r.t time.
pos_estimation_error = zeros(1,size(timeIndices,2));
pos_prediction_error = zeros(1,size(timeIndices,2));

for i = 1:size(timeIndices,2)
  pos_estimation_error(i) = sqrt((true_XPos(i)-estimated_states(1,i))^2+(true_YPos(i)-estimated_states(2,i))^2);
  pos_prediction_error(i) = sqrt((true_XPos(i)-predicted_states(1,i))^2+(true_YPos(i)-predicted_states(2,i))^2);
end

figure 
plot(timeIndices,pos_estimation_error,"LineStyle","--","Marker","+","MarkerSize",5,'DisplayName','Estimation Error');
hold on
plot(timeIndices,pos_prediction_error,"LineStyle","--","Marker","+","MarkerSize",5,'DisplayName','Prediction Error');
title("Estimation and Prediction Errors vs. Time");
ylabel("Error");
xlabel("Time (s)");
legend("show");
grid on;

% calculate RMS error.

RMS_estimated = sqrt(sum(pos_estimation_error.^2)/length(pos_estimation_error));
RMS_predicted = sqrt(sum(pos_prediction_error.^2)/length(pos_prediction_error));

fprintf("Estimated RMS is: %.4f\n",RMS_estimated);
fprintf("Predicted RMS is: %.4f\n",RMS_predicted);

% Re-run the kalman filter example with fixed process noise, but x100 /100
% measurement noises.

% R' =100*R

R_times100 = 100*R; % Measurement Noise

estimated_states_Rtimes100 = zeros(4,size(timeIndices,2));
prev_state = x0;
prev_state_covariance = P0;
prev_time = 0;

for i = 1:size(timeIndices,2)
  current_time = timeIndices(i);
  dt = current_time - prev_time;
  A_k = A(dt);
  B_k = B(dt);
  z_k = [measured_XPos(i);...
    measured_YPos(i)];
  %predict the state and covariance.
  predicted_state = A_k*prev_state;
  predicted_covariance = A_k*prev_state_covariance*A_k' + B_k*Q*B_k';
  %measurement update.
  K_k = predicted_covariance*C'*(C*predicted_covariance*C'+R_times100)^-1;% Kalman Gain
  estimated_states_Rtimes100(:,i) = predicted_state+K_k*(z_k-C*predicted_state);
  I = eye(4,4);
  estimated_covariance = (I-K_k*C)*predicted_covariance;
  % update k-1|k-1 state and covariance for next step
  prev_state = estimated_states_Rtimes100(:,i);
  prev_state_covariance = estimated_covariance;
  prev_time = current_time;
end

% plot the R' = 100*R case, measurements, true trajectory and estimated
% trajectory.

figure
plot(true_XPos,true_YPos,"Marker","o","MarkerSize",5,'DisplayName','True Trajectory');
hold on
plot(measured_XPos,measured_YPos,"LineStyle","none","Marker","x","MarkerSize",8,'DisplayName','Noisy Measurements');
plot(estimated_states_Rtimes100(1,:),estimated_states_Rtimes100(2,:),"LineStyle","--","Marker","diamond","MarkerSize",5,'DisplayName','Estimated Trajectory');
title("True Trajectory of The Target, Noisy Measurements and Estimated Trajectory, R = 100*R");
ylabel("Y Position(m)");
xlabel("X Position(m)");
xlim([800,2800]);
ylim([200,2200]);
legend("show");
grid on;

% R' = R/100

R_div100 = 0.01*R; % Measurement Noise

estimated_states_Rdiv100 = zeros(4,size(timeIndices,2));
prev_state = x0;
prev_state_covariance = P0;
prev_time = 0;

for i = 1:size(timeIndices,2)
  current_time = timeIndices(i);
  dt = current_time - prev_time;
  A_k = A(dt);
  B_k = B(dt);
  z_k = [measured_XPos(i);...
    measured_YPos(i)];
  %predict the state and covariance.
  predicted_state = A_k*prev_state;
  predicted_covariance = A_k*prev_state_covariance*A_k' + B_k*Q*B_k';
  %measurement update.
  K_k = predicted_covariance*C'*(C*predicted_covariance*C'+R_div100)^-1;% Kalman Gain
  estimated_states_Rdiv100(:,i) = predicted_state+K_k*(z_k-C*predicted_state);
  I = eye(4,4);
  estimated_covariance = (I-K_k*C)*predicted_covariance;
  % update k-1|k-1 state and covariance for next step
  prev_state = estimated_states_Rdiv100(:,i);
  prev_state_covariance = estimated_covariance;
  prev_time = current_time;
end

% plot the R' = R/100 case, measurements, true trajectory and estimated
% trajectory.

figure
plot(true_XPos,true_YPos,"Marker","o","MarkerSize",5,'DisplayName','True Trajectory');
hold on
plot(measured_XPos,measured_YPos,"LineStyle","none","Marker","x","MarkerSize",8,'DisplayName','Noisy Measurements');
plot(estimated_states_Rdiv100(1,:),estimated_states_Rdiv100(2,:),"LineStyle","--","Marker","diamond","MarkerSize",5,'DisplayName','Estimated Trajectory');
title("True Trajectory of The Target, Noisy Measurements and Estimated Trajectory, R' = R/100");
ylabel("Y Position(m)");
xlabel("X Position(m)");
xlim([800,2800]);
ylim([200,2200]);
legend("show");
grid on;

% Re-run the kalman filter example with fixed measurement noise, but x100 /100
% process noises.

% Q' = 100*Q

Q_times100 = 100*Q; % Measurement Noise

estimated_states_Qtimes100 = zeros(4,size(timeIndices,2));
prev_state = x0;
prev_state_covariance = P0;
prev_time = 0;

for i = 1:size(timeIndices,2)
  current_time = timeIndices(i);
  dt = current_time - prev_time;
  A_k = A(dt);
  B_k = B(dt);
  z_k = [measured_XPos(i);...
    measured_YPos(i)];
  %predict the state and covariance.
  predicted_state = A_k*prev_state;
  predicted_covariance = A_k*prev_state_covariance*A_k' + B_k*Q_times100*B_k';
  %measurement update.
  K_k = predicted_covariance*C'*(C*predicted_covariance*C'+R)^-1;% Kalman Gain
  estimated_states_Qtimes100(:,i) = predicted_state+K_k*(z_k-C*predicted_state);
  I = eye(4,4);
  estimated_covariance = (I-K_k*C)*predicted_covariance;
  % update k-1|k-1 state and covariance for next step
  prev_state = estimated_states_Qtimes100(:,i);
  prev_state_covariance = estimated_covariance;
  prev_time = current_time;
end

% plot the Q' = 100*Q case, measurements, true trajectory and estimated
% trajectory.

figure
plot(true_XPos,true_YPos,"Marker","o","MarkerSize",5,'DisplayName','True Trajectory');
hold on
plot(measured_XPos,measured_YPos,"LineStyle","none","Marker","x","MarkerSize",8,'DisplayName','Noisy Measurements');
plot(estimated_states_Qtimes100(1,:),estimated_states_Qtimes100(2,:),"LineStyle","--","Marker","diamond","MarkerSize",5,'DisplayName','Estimated Trajectory');
title("True Trajectory of The Target, Noisy Measurements and Estimated Trajectory, Q' = 100*Q");
ylabel("Y Position(m)");
xlabel("X Position(m)");
xlim([800,2800]);
ylim([200,2200]);
legend("show");
grid on;

% Q' = Q/100

Q_div100 = 0.01*Q; % Measurement Noise

estimated_states_Qdiv100 = zeros(4,size(timeIndices,2));
prev_state = x0;
prev_state_covariance = P0;
prev_time = 0;

for i = 1:size(timeIndices,2)
  current_time = timeIndices(i);
  dt = current_time - prev_time;
  A_k = A(dt);
  B_k = B(dt);
  z_k = [measured_XPos(i);...
    measured_YPos(i)];
  %predict the state and covariance.
  predicted_state = A_k*prev_state;
  predicted_covariance = A_k*prev_state_covariance*A_k' + B_k*Q_div100*B_k';
  %measurement update.
  K_k = predicted_covariance*C'*(C*predicted_covariance*C'+R)^-1;% Kalman Gain
  estimated_states_Qdiv100(:,i) = predicted_state+K_k*(z_k-C*predicted_state);
  I = eye(4,4);
  estimated_covariance = (I-K_k*C)*predicted_covariance;
  % update k-1|k-1 state and covariance for next step
  prev_state = estimated_states_Qdiv100(:,i);
  prev_state_covariance = estimated_covariance;
  prev_time = current_time;
end

% plot the Q' = Q/100 case, measurements, true trajectory and estimated
% trajectory.

figure
plot(true_XPos,true_YPos,"Marker","o","MarkerSize",5,'DisplayName','True Trajectory');
hold on
plot(measured_XPos,measured_YPos,"LineStyle","none","Marker","x","MarkerSize",8,'DisplayName','Noisy Measurements');
plot(estimated_states_Qdiv100(1,:),estimated_states_Qdiv100(2,:),"LineStyle","--","Marker","diamond","MarkerSize",5,'DisplayName','Estimated Trajectory');
title("True Trajectory of The Target, Noisy Measurements and Estimated Trajectory, Q' = Q/100");
ylabel("Y Position(m)");
xlabel("X Position(m)");
xlim([800,2800]);
ylim([200,2200]);
legend("show");
grid on;

% Range and bearing measurements. The method assumes our sensor is in
% origin.

meas_Range = zeros(1,size(timeIndices,2));
meas_Bearing = zeros(1,size(timeIndices,2));
sigma_Rng = 100; % meters
sigma_Bearing = 5; %degrees

for i = 1:size(timeIndices,2)
  meas_Range(i) = sqrt(true_XPos(i)^2 + true_YPos(i)^2) + sigma_Rng*randn;
  meas_Bearing(i) = atan2d(true_YPos(i),true_XPos(i)) + sigma_Bearing*randn;
end

R_polar = [sigma_Rng^2,0;...
            0,sigma_Bearing^2];
% Implement the ekf approach. The change would be to use first order Taylor
% series expansion of the measurement function. 

measFunc = @(x,y) ([sqrt(x^2+y^2);...
                    atan2d(y,x)]);

% dr/dx = x/sqrt(x^2+y^2)
% dr/dy = y/sqrt(x^2+y^2)
% dTheta/dx = -y/(x^2+y^2)
% dTheta/dy = x/(x^2+y^2)
C_jac = @(x,y) ([x/sqrt(x^2+y^2),y/sqrt(x^2+y^2),0,0;...
                 -y/(x^2+y^2),x/(x^2+y^2),0,0]);

% Estimate using EKF

estimated_states_EKF = zeros(4,size(timeIndices,2));
predicted_states_EKF = zeros(4,size(timeIndices,2));
% initialize the estimator
prev_state = x0;
prev_state_covariance = P0;
prev_time = 0;

for i = 1:size(timeIndices,2)
  current_time = timeIndices(i);
  dt = current_time - prev_time;
  A_k = A(dt);
  B_k = B(dt);
  z_k = [meas_Range(i);...
    meas_Bearing(i)];
  %predict the state and covariance.
  predicted_states_EKF(:,i) = A_k*prev_state;
  C_k = C_jac(predicted_states_EKF(1,i),predicted_states_EKF(2,i));
  predicted_covariance = A_k*prev_state_covariance*A_k' + B_k*Q*B_k';
  %measurement update.
  K_k = predicted_covariance*C_k'*(C_k*predicted_covariance*C_k'+R_polar)^-1;% Kalman Gain
  meas_prediction = measFunc(predicted_states_EKF(1,i),predicted_states_EKF(2,i));
  estimated_states_EKF(:,i) = predicted_states_EKF(:,i)+K_k*(z_k-meas_prediction);
  I = eye(4,4);
  estimated_covariance = (I-K_k*C_k)*predicted_covariance;
  % update k-1|k-1 state and covariance for next step
  prev_state = estimated_states_EKF(:,i);
  prev_state_covariance = estimated_covariance;
  prev_time = current_time;
end

%  calculate the estimation and prediction errors and plot w.r.t time.
pos_estimation_error_EKF = zeros(1,size(timeIndices,2));
pos_prediction_error_EKF = zeros(1,size(timeIndices,2));

for i = 1:size(timeIndices,2)
  pos_estimation_error_EKF(i) = sqrt((true_XPos(i)-estimated_states_EKF(1,i))^2+(true_YPos(i)-estimated_states_EKF(2,i))^2);
  pos_prediction_error_EKF(i) = sqrt((true_XPos(i)-predicted_states_EKF(1,i))^2+(true_YPos(i)-predicted_states_EKF(2,i))^2);
end

% calculate RMS error.

RMS_estimated_EKF = sqrt(sum(pos_estimation_error_EKF.^2)/length(pos_estimation_error_EKF));
RMS_predicted_EKF = sqrt(sum(pos_prediction_error_EKF.^2)/length(pos_prediction_error_EKF));

fprintf("EKF Estimated RMS is: %.4f\n",RMS_estimated_EKF);
fprintf("EKF Predicted RMS is: %.4f\n",RMS_predicted_EKF);

% Estimate using UKF

estimated_states_UKF = zeros(4,size(timeIndices,2));
predicted_states_UKF = zeros(4,size(timeIndices,2));
% initialize the estimator
prev_state = x0;
prev_state_covariance = P0;
prev_time = 0;

for i = 1:size(timeIndices,2)
  current_time = timeIndices(i);
  dt = current_time - prev_time;
  A_k = A(dt);
  B_k = B(dt);
  z_k = [meas_Range(i);...
    meas_Bearing(i)];
  %predict the state and covariance.
  predicted_states_UKF(:,i) = A_k*prev_state;
  predicted_covariance = A_k*prev_state_covariance*A_k' + B_k*Q*B_k';
  %measurement update.
  % Calculate sigma points & weigths.
  n_x = 4;
  n_z = 2;
  weights = 1/9*ones(1,2*n_x+1);
  sigmas = zeros(4,2*n_x+1);
  mean = predicted_states_UKF(:,i);
  sigmas(:,1) = mean;
  sqrt_cov = sqrtm(predicted_covariance);
  for j = 1:n_x
    sigmas(:,j+1) = mean + sqrt(n_x/(1-1/9))* sqrt_cov(:,j);
    sigmas(:,j+1+n_x) = mean - sqrt(n_x/(1-1/9)) * sqrt_cov(:,j);
  end
  sigma_measured = zeros(2,2*n_x+1);
  for j=1:2*n_x+1
    sigma_measured(:,j) = measFunc(sigmas(1,j),sigmas(2,j));
  end
  meas_prediction = zeros(2,1);
  for j = 1:2*n_x+1
    meas_prediction = meas_prediction + weights(j) * sigma_measured(:,j);
  end
  inov_cov = zeros(2,2);
  for j = 1:2*n_x+1
    inov_cov = inov_cov + weights(j)*(sigma_measured(:,j)-meas_prediction)*(sigma_measured(:,j)-meas_prediction)';
  end
  inov_cov = inov_cov + R_polar;
  sigma_xz = zeros(n_x,n_z); 
  for j = 1:2*n_x+1
    sigma_xz = sigma_xz + weights(j) * (sigmas(:,j)-mean) * (sigma_measured(:,j)-meas_prediction)';
  end
  K_k = sigma_xz * (inov_cov)^-1;% Kalman Gain
  estimated_states_UKF(:,i) = predicted_states_UKF(:,i)+K_k*(z_k-meas_prediction);
  estimated_covariance = predicted_covariance-K_k*inov_cov*K_k';
  % update k-1|k-1 state and covariance for next step
  prev_state = estimated_states_UKF(:,i);
  prev_state_covariance = estimated_covariance;
  prev_time = current_time;
end

%  calculate the estimation and prediction errors and plot w.r.t time.
pos_estimation_error_UKF = zeros(1,size(timeIndices,2));
pos_prediction_error_UKF = zeros(1,size(timeIndices,2));

for i = 1:size(timeIndices,2)
  pos_estimation_error_UKF(i) = sqrt((true_XPos(i)-estimated_states_UKF(1,i))^2+(true_YPos(i)-estimated_states_UKF(2,i))^2);
  pos_prediction_error_UKF(i) = sqrt((true_XPos(i)-predicted_states_UKF(1,i))^2+(true_YPos(i)-predicted_states_UKF(2,i))^2);
end

% calculate RMS error.

RMS_estimated_UKF = sqrt(sum(pos_estimation_error_UKF.^2)/length(pos_estimation_error_UKF));
RMS_predicted_UKF = sqrt(sum(pos_prediction_error_UKF.^2)/length(pos_prediction_error_UKF));

fprintf("UKF Estimated RMS is: %.4f\n",RMS_estimated_UKF);
fprintf("UKF Predicted RMS is: %.4f\n",RMS_predicted_UKF);

% plot EKF and UKF estimation / prediction erros.


figure 
plot(timeIndices,pos_estimation_error_EKF,"LineStyle","--","Marker","+","MarkerSize",5,'DisplayName','EKF Estimation Error');
hold on
plot(timeIndices,pos_prediction_error_EKF,"LineStyle","--","Marker","+","MarkerSize",5,'DisplayName','EKF Prediction Error');
plot(timeIndices,pos_estimation_error_UKF,"LineStyle","--","Marker","+","MarkerSize",5,'DisplayName','UKF Estimation Error');
plot(timeIndices,pos_prediction_error_UKF,"LineStyle","--","Marker","+","MarkerSize",5,'DisplayName','UKF Prediction Error');
title("Estimation and Prediction Errors vs. Time");
ylabel("Error");
xlabel("Time (s)");
legend("show");
grid on;
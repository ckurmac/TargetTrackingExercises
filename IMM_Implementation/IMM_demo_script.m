clc; clear all;
%% Part a
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

%% Part b
% Generate noisy measurements
sigma_measurement = 20;
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

%% Part c
% Create 3 Kalman Filters with different noise Covariances. 
Q1 = 0.1^2*eye(2);
Q2 = eye(2);
Q3 = 100*eye(2);

R = sigma_measurement^2*eye(2);

x0 = [measured_XPos(1)-25;measured_YPos(1)-25;0;0];
xP0 = [2500,0,0,0;... %Vmax*t^2
       0,2500,0,0;... %Vmax*t^2 
       0,0,2500,0;... %Vmax^2 
       0,0,0,2500];   %Vmax^2

KF1 = CV_KF(Q1,x0,xP0);
KF2 = CV_KF(Q2,x0,xP0);
KF3 = CV_KF(Q3,x0,xP0);

KF1_estimation_error = zeros(1,length(timeIndices));
KF2_estimation_error = zeros(1,length(timeIndices));
KF3_estimation_error = zeros(1,length(timeIndices));

KF1_estimated_states = zeros(4,length(timeIndices));
KF2_estimated_states = zeros(4,length(timeIndices));
KF3_estimated_states = zeros(4,length(timeIndices));

KF1_gate_sizes = zeros(1,length(timeIndices));
KF2_gate_sizes = zeros(1,length(timeIndices));
KF3_gate_sizes = zeros(1,length(timeIndices));


% Q1 
prevTime = 0;
isEllipseLegendInitiated = false;

gamma_G = 9.2; % Chi squared distribution, 0.99 area for 2DOF.

figure
plot(true_XPos,true_YPos,"Marker","o","MarkerSize",5,'DisplayName','True Trajectory');
hold on
plot(measured_XPos,measured_YPos,"LineStyle","none","Marker","x","MarkerSize",8,'DisplayName','Noisy Measurements');
for i = 1:length(timeIndices)
    currTime = timeIndices(i);
    dt = currTime-prevTime;

    y_k = [measured_XPos(i);measured_YPos(i)];
    
    [KF1_predicted_x,KF1_predicted_xP] = KF1.predict(dt);
    
    Sk_KF1 = KF1.C * KF1_predicted_xP * KF1.C' + R;
    
    
    [KF1_estimated_x,~] = KF1.update(y_k,R);
    KF1_estimated_states(:,i) = KF1_estimated_x;
    prevTime = currTime;
        
    if(~isEllipseLegendInitiated)
       toLegend = true;
       isEllipseLegendInitiated = true;
    else
       toLegend = false;
    end
    draw_ellipse([KF1_predicted_x(1),KF1_predicted_x(2)],Sk_KF1,gamma_G,toLegend);
    KF1_gate_sizes(i) = pi*gamma_G*sqrt(det(Sk_KF1));
end
plot(KF1_estimated_states(1,:),KF1_estimated_states(2,:),"LineStyle","--","Marker","diamond","MarkerSize",5,'DisplayName','Estimated Trajectory');
title("Q1, True Trajectory of The Target, Noisy Measurements, Covariance Ellipses and Estimated Trajectory");
ylabel("Y Position(m)");
xlabel("X Position(m)");
xlim([800,2800]);
ylim([200,2200]);
legend("show");
grid on;

for i = 1:length(timeIndices)
    KF1_estimation_error(i) = sqrt((true_XPos(i)-KF1_estimated_states(1,i))^2+(true_YPos(i)-KF1_estimated_states(2,i))^2);
end

figure 
plot(timeIndices,KF1_estimation_error,"LineStyle","--","Marker","+","MarkerSize",5,'DisplayName','Estimation Error');
hold on
title("Q1 Estimation Error vs. Time");
ylabel("Error");
xlabel("Time (s)");
legend("show");
grid on;

figure 
plot(timeIndices(2:end),KF1_gate_sizes(2:end),"LineStyle","--","Marker","+","MarkerSize",5,'DisplayName','Gate Size');
hold on
title("Q1 Gate Sizes vs. Time");
ylabel("Gate Size");
xlabel("Time (s)");
legend("show");
grid on;

% Q2

prevTime = 0;
isEllipseLegendInitiated = false;

figure
plot(true_XPos,true_YPos,"Marker","o","MarkerSize",5,'DisplayName','True Trajectory');
hold on
plot(measured_XPos,measured_YPos,"LineStyle","none","Marker","x","MarkerSize",8,'DisplayName','Noisy Measurements');
for i = 1:length(timeIndices)
    currTime = timeIndices(i);
    dt = currTime-prevTime;

    y_k = [measured_XPos(i);measured_YPos(i)];

    [KF2_predicted_x,KF2_predicted_xP] = KF2.predict(dt);

    Sk_KF2 = KF2.C * KF2_predicted_xP * KF2.C' + R;

    [KF2_estimated_x,~] = KF2.update(y_k,R);
    KF2_estimated_states(:,i) = KF2_estimated_x;
    prevTime = currTime;

    if(~isEllipseLegendInitiated)
        toLegend = true;
        isEllipseLegendInitiated = true;
    else
        toLegend = false;
    end
    draw_ellipse([KF2_predicted_x(1),KF2_predicted_x(2)],Sk_KF2,gamma_G,toLegend);
    KF2_gate_sizes(i) = pi*gamma_G*sqrt(det(Sk_KF2));

end
plot(KF2_estimated_states(1,:),KF2_estimated_states(2,:),"LineStyle","--","Marker","diamond","MarkerSize",5,'DisplayName','Estimated Trajectory');
title("Q2, True Trajectory of The Target, Noisy Measurements, Covariance Ellipses and Estimated Trajectory");
ylabel("Y Position(m)");
xlabel("X Position(m)");
xlim([800,2800]);
ylim([200,2200]);
legend("show");
grid on;

for i = 1:length(timeIndices)
    KF2_estimation_error(i) = sqrt((true_XPos(i)-KF2_estimated_states(1,i))^2+(true_YPos(i)-KF2_estimated_states(2,i))^2);
end

figure 
plot(timeIndices,KF2_estimation_error,"LineStyle","--","Marker","+","MarkerSize",5,'DisplayName','Estimation Error');
hold on
title("Q2 Estimation Error vs. Time");
ylabel("Error");
xlabel("Time (s)");
legend("show");
grid on;

figure 
plot(timeIndices(2:end),KF2_gate_sizes(2:end),"LineStyle","--","Marker","+","MarkerSize",5,'DisplayName','Gate Size');
hold on
title("Q2 Gate Sizes vs. Time");
ylabel("Gate Size");
xlabel("Time (s)");
legend("show");
grid on;

% Q3 
prevTime = 0;
isEllipseLegendInitiated = false;

figure
plot(true_XPos,true_YPos,"Marker","o","MarkerSize",5,'DisplayName','True Trajectory');
hold on
plot(measured_XPos,measured_YPos,"LineStyle","none","Marker","x","MarkerSize",8,'DisplayName','Noisy Measurements');
for i = 1:length(timeIndices)
    currTime = timeIndices(i);
    dt = currTime-prevTime;

    y_k = [measured_XPos(i);measured_YPos(i)];

    [KF3_predicted_x,KF3_predicted_xP] = KF3.predict(dt);

    Sk_KF3 = KF3.C * KF3_predicted_xP * KF3.C' + R;

    [KF3_estimated_x,~] = KF3.update(y_k,R);
    KF3_estimated_states(:,i) = KF3_estimated_x;
    prevTime = currTime;

    if(~isEllipseLegendInitiated)
        toLegend = true;
        isEllipseLegendInitiated = true;
    else
        toLegend = false;
    end
    draw_ellipse([KF3_predicted_x(1),KF3_predicted_x(2)],Sk_KF3,gamma_G,toLegend);
    KF3_gate_sizes(i) = pi*gamma_G*sqrt(det(Sk_KF3));
end
plot(KF3_estimated_states(1,:),KF3_estimated_states(2,:),"LineStyle","--","Marker","diamond","MarkerSize",5,'DisplayName','Estimated Trajectory');
title("Q3, True Trajectory of The Target, Noisy Measurements, Covariance Ellipses and Estimated Trajectory");
ylabel("Y Position(m)");
xlabel("X Position(m)");
xlim([800,2800]);
ylim([200,2200]);
legend("show");
grid on;

for i = 1:length(timeIndices)
    KF3_estimation_error(i) = sqrt((true_XPos(i)-KF3_estimated_states(1,i))^2+(true_YPos(i)-KF3_estimated_states(2,i))^2);
end

figure 
plot(timeIndices,KF3_estimation_error,"LineStyle","--","Marker","+","MarkerSize",5,'DisplayName','Estimation Error');
hold on
title("Q3 Estimation Error vs. Time");
ylabel("Error");
xlabel("Time (s)");
legend("show");
grid on;

figure 
plot(timeIndices(2:end),KF3_gate_sizes(2:end),"LineStyle","--","Marker","+","MarkerSize",5,'DisplayName','Gate Size');
hold on
title("Q3 Gate Sizes vs. Time");
ylabel("Gate Size");
xlabel("Time (s)");
legend("show");
grid on;

%% Part d 
% IMM Filter 1

KF_Low_Q = CV_KF(Q1,x0,xP0);
KF_High_Q = CV_KF(Q3,x0,xP0);
TPM = [0.99,0.01;...
       0.01,0.99];

IMM1 = IMMFilter(x0,xP0,{KF_Low_Q,KF_High_Q},TPM);


IMM1_estimation_error = zeros(1,length(timeIndices));

IMM1_estimated_states = zeros(4,length(timeIndices));

IMM1_gate_sizes = zeros(1,length(timeIndices));

IMM1_model_probabilities = zeros(2,length(timeIndices));

figure
plot(true_XPos,true_YPos,"Marker","o","MarkerSize",5,'DisplayName','True Trajectory');
hold on
plot(measured_XPos,measured_YPos,"LineStyle","none","Marker","x","MarkerSize",8,'DisplayName','Noisy Measurements');
for i = 1:length(timeIndices)
    currTime = timeIndices(i);
    dt = currTime-prevTime;

    y_k = [measured_XPos(i);measured_YPos(i)];

    [IMM1_estimated_states(:,i),IMM1_estimated_xP,IMM1_model_prob,IMM1_y_k_predict,IMM1_S_k] = IMM1.update(y_k,R,dt);

    prevTime = currTime;
    
    IMM1_model_probabilities(:,i) = IMM1_model_prob';

    if(~isEllipseLegendInitiated)
        toLegend = true;
        isEllipseLegendInitiated = true;
    else
        toLegend = false;
    end
    draw_ellipse([IMM1_y_k_predict(1),IMM1_y_k_predict(2)],IMM1_S_k,gamma_G,toLegend);
    IMM1_gate_sizes(i) = pi*gamma_G*sqrt(det(IMM1_S_k));
end
plot(IMM1_estimated_states(1,:),IMM1_estimated_states(2,:),"LineStyle","--","Marker","diamond","MarkerSize",5,'DisplayName','Estimated Trajectory');
title("Part D IMM, True Trajectory of The Target, Noisy Measurements, Covariance Ellipses and Estimated Trajectory");
ylabel("Y Position(m)");
xlabel("X Position(m)");
xlim([800,2800]);
ylim([200,2200]);
legend("show");
grid on;

for i = 1:length(timeIndices)
    IMM1_estimation_error(i) = sqrt((true_XPos(i)-IMM1_estimated_states(1,i))^2+(true_YPos(i)-IMM1_estimated_states(2,i))^2);
end

figure 
plot(timeIndices,IMM1_estimation_error,"LineStyle","--","Marker","+","MarkerSize",5,'DisplayName','Estimation Error');
hold on
title("Part D IMM, Estimation Error vs. Time");
ylabel("Error");
xlabel("Time (s)");
legend("show");
grid on;

figure 
plot(timeIndices(2:end),IMM1_gate_sizes(2:end),"LineStyle","--","Marker","+","MarkerSize",5,'DisplayName','Gate Size');
hold on
title("Part D IMM, Gate Sizes vs. Time");
ylabel("Gate Size");
xlabel("Time (s)");
legend("show");
grid on;

figure 
plot(timeIndices,IMM1_model_probabilities(1,:),"LineStyle","--","Marker","+","MarkerSize",5,'DisplayName','Model 1 Probability');
hold on
plot(timeIndices,IMM1_model_probabilities(2,:),"LineStyle","--","Marker","+","MarkerSize",5,'DisplayName','Model 2 Probability');
title("Part D IMM, Model Probabilities vs. Time");
ylabel("Model Probablility");
xlabel("Time (s)");
legend("show");
grid on;

%% Part E
% IMM Filter 2

KF_Low_Q = CV_KF(Q1,x0,xP0);
KF_High_Q = CV_KF(Q3,x0,xP0);
TPM = [0.999,0.001;...
    0.001,0.999];

IMM2 = IMMFilter(x0,xP0,{KF_Low_Q,KF_High_Q},TPM);


IMM2_estimation_error = zeros(1,length(timeIndices));

IMM2_estimated_states = zeros(4,length(timeIndices));

IMM2_gate_sizes = zeros(1,length(timeIndices));

IMM2_model_probabilities = zeros(2,length(timeIndices));

figure
plot(true_XPos,true_YPos,"Marker","o","MarkerSize",5,'DisplayName','True Trajectory');
hold on
plot(measured_XPos,measured_YPos,"LineStyle","none","Marker","x","MarkerSize",8,'DisplayName','Noisy Measurements');
for i = 1:length(timeIndices)
    currTime = timeIndices(i);
    dt = currTime-prevTime;

    y_k = [measured_XPos(i);measured_YPos(i)];

    [IMM2_estimated_states(:,i),IMM2_estimated_xP,IMM2_model_prob,IMM2_y_k_predict,IMM2_S_k] = IMM2.update(y_k,R,dt);

    prevTime = currTime;

    IMM2_model_probabilities(:,i) = IMM2_model_prob';

    if(~isEllipseLegendInitiated)
        toLegend = true;
        isEllipseLegendInitiated = true;
    else
        toLegend = false;
    end
    draw_ellipse([IMM2_y_k_predict(1),IMM2_y_k_predict(2)],IMM2_S_k,gamma_G,toLegend);
    IMM2_gate_sizes(i) = pi*gamma_G*sqrt(det(IMM2_S_k));
end
plot(IMM2_estimated_states(1,:),IMM2_estimated_states(2,:),"LineStyle","--","Marker","diamond","MarkerSize",5,'DisplayName','Estimated Trajectory');
title("Part E IMM1 (0.999,0.001), True Trajectory of The Target, Noisy Measurements, Covariance Ellipses and Estimated Trajectory");
ylabel("Y Position(m)");
xlabel("X Position(m)");
xlim([800,2800]);
ylim([200,2200]);
legend("show");
grid on;

for i = 1:length(timeIndices)
    IMM2_estimation_error(i) = sqrt((true_XPos(i)-IMM2_estimated_states(1,i))^2+(true_YPos(i)-IMM2_estimated_states(2,i))^2);
end

figure 
plot(timeIndices,IMM2_estimation_error,"LineStyle","--","Marker","+","MarkerSize",5,'DisplayName','Estimation Error');
hold on
title("Part E IMM1 (0.999,0.001), Estimation Error vs. Time");
ylabel("Error");
xlabel("Time (s)");
legend("show");
grid on;

figure 
plot(timeIndices(2:end),IMM2_gate_sizes(2:end),"LineStyle","--","Marker","+","MarkerSize",5,'DisplayName','Gate Size');
hold on
title("Part E IMM1 (0.999,0.001), Gate Sizes vs. Time");
ylabel("Gate Size");
xlabel("Time (s)");
legend("show");
grid on;

figure 
plot(timeIndices,IMM2_model_probabilities(1,:),"LineStyle","--","Marker","+","MarkerSize",5,'DisplayName','Model 1 Probability');
hold on
plot(timeIndices,IMM2_model_probabilities(2,:),"LineStyle","--","Marker","+","MarkerSize",5,'DisplayName','Model 2 Probability');
title("Part E IMM1 (0.999,0.001), Model Probabilities vs. Time");
ylabel("Model Probablility");
xlabel("Time (s)");
legend("show");
grid on;

% IMM Filter 3

KF_Low_Q = CV_KF(Q1,x0,xP0);
KF_High_Q = CV_KF(Q3,x0,xP0);
TPM = [0.5,0.5;...
    0.5,0.5];

IMM3 = IMMFilter(x0,xP0,{KF_Low_Q,KF_High_Q},TPM);


IMM3_estimation_error = zeros(1,length(timeIndices));

IMM3_estimated_states = zeros(4,length(timeIndices));

IMM3_gate_sizes = zeros(1,length(timeIndices));

IMM3_model_probabilities = zeros(2,length(timeIndices));

figure
plot(true_XPos,true_YPos,"Marker","o","MarkerSize",5,'DisplayName','True Trajectory');
hold on
plot(measured_XPos,measured_YPos,"LineStyle","none","Marker","x","MarkerSize",8,'DisplayName','Noisy Measurements');
for i = 1:length(timeIndices)
    currTime = timeIndices(i);
    dt = currTime-prevTime;

    y_k = [measured_XPos(i);measured_YPos(i)];

    [IMM3_estimated_states(:,i),IMM3_estimated_xP,IMM3_model_prob,IMM3_y_k_predict,IMM3_S_k] = IMM3.update(y_k,R,dt);

    prevTime = currTime;

    IMM3_model_probabilities(:,i) = IMM3_model_prob';

    if(~isEllipseLegendInitiated)
        toLegend = true;
        isEllipseLegendInitiated = true;
    else
        toLegend = false;
    end
    draw_ellipse([IMM3_y_k_predict(1),IMM3_y_k_predict(2)],IMM3_S_k,gamma_G,toLegend);
    IMM3_gate_sizes(i) = pi*gamma_G*sqrt(det(IMM3_S_k));
end
plot(IMM3_estimated_states(1,:),IMM3_estimated_states(2,:),"LineStyle","--","Marker","diamond","MarkerSize",5,'DisplayName','Estimated Trajectory');
title("Part E IMM2 (0.5,0.5) True Trajectory of The Target, Noisy Measurements, Covariance Ellipses and Estimated Trajectory");
ylabel("Y Position(m)");
xlabel("X Position(m)");
xlim([800,2800]);
ylim([200,2200]);
legend("show");
grid on;

for i = 1:length(timeIndices)
    IMM3_estimation_error(i) = sqrt((true_XPos(i)-IMM3_estimated_states(1,i))^2+(true_YPos(i)-IMM3_estimated_states(2,i))^2);
end

figure 
plot(timeIndices,IMM3_estimation_error,"LineStyle","--","Marker","+","MarkerSize",5,'DisplayName','Estimation Error');
hold on
title("Part E IMM2 (0.5,0.5), Estimation Error vs. Time");
ylabel("Error");
xlabel("Time (s)");
legend("show");
grid on;

figure 
plot(timeIndices(2:end),IMM3_gate_sizes(2:end),"LineStyle","--","Marker","+","MarkerSize",5,'DisplayName','Gate Size');
hold on
title("Part E IMM2 (0.5,0.5), Gate Sizes vs. Time");
ylabel("Gate Size");
xlabel("Time (s)");
legend("show");
grid on;

figure 
plot(timeIndices,IMM3_model_probabilities(1,:),"LineStyle","--","Marker","+","MarkerSize",5,'DisplayName','Model 1 Probability');
hold on
plot(timeIndices,IMM3_model_probabilities(2,:),"LineStyle","--","Marker","+","MarkerSize",5,'DisplayName','Model 2 Probability');
title("Part E IMM2 (0.5,0.5), Model Probabilities vs. Time");
ylabel("Model Probablility");
xlabel("Time (s)");
legend("show");
grid on;
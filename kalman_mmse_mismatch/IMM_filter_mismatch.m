% random CV target generation

x_0 = [0;0;25;25]; % [x,y,vx,vy]
P_0 = eye(4);

process_noise_std = 1;
measurement_noise_std = 1;

N_iter = 1000;
time = 150;
T = 1;
N_step = time/T;

x_cv = zeros(4,N_step+1);
x_est = zeros(4,N_step+1); 
y_cv = zeros(2,N_step+1);

C = [1,0,0,0;0,1,0,0];

Qlow = 0.1^2*eye(2);
Qhigh = 100*eye(2);

MMSE = zeros(1,N_step+1);
kalmanMMSE = zeros(1,N_step+1);

covariances = cell(1,N_step+1);

isNoised = false;

for i = 1:N_iter
    x_cv(:,1) = x_0;
    y_cv(:,1) = C * x_0 + randn(2,1)*measurement_noise_std;
    % generate random data for current iteration
    for j = 2:N_step+1
        A_j = getCVStateTransitionMtx(T);
        B_j = getCVNoiseGainMtx(T);
        noiseTerm = B_j*randn(2,1)*process_noise_std;
        if(isNoised)
            x_cv(:,j) = A_j*x_cv(:,j-1) + noiseTerm;
        else
            x_cv(:,j) = A_j*x_cv(:,j-1);
        end
        y_cv(:,j) = C*x_cv(:,j) + randn(2,1)*measurement_noise_std;
    end
    % IMM Filtering
    % first step
    measurementCovariance = eye(2) * measurement_noise_std^2;

    KF_Low_Q = CV_KF(Qlow,x_0,P_0);
    KF_High_Q = CV_KF(Qhigh,x_0,P_0);
    TPM = [0.5,0.5;...
        0.5,0.5];

    IMM = IMMFilter(x_0,P_0,{KF_Low_Q,KF_High_Q},TPM);
    [x_est(:,1),covariances{1,1},~,~,~] = IMM.update(y_cv(:,1),measurementCovariance,0);

    for j = 2:N_step+1
        [x_est(:,j),covariances{1,j},~,~,~] = IMM.update(y_cv(:,j),measurementCovariance,T);
    end
    % MMSE calculation
    currMSE = zeros(1,N_step+1);
    currKalmanMSE = zeros(1,N_step+1);
    for j=1:N_step+1
        currMSE(j) = (x_est([1,2],j)-x_cv([1,2],j))'*(x_est([1,2],j)-x_cv([1,2],j));
        covMtx = covariances{1,j};
        currKalmanMSE(j) = trace(covMtx([1,2],[1,2]));
    end
    MMSE = MMSE + currMSE;
    kalmanMMSE = kalmanMMSE + currKalmanMSE;
end

MMSE = MMSE/N_iter;
kalmanMMSE = kalmanMMSE/N_iter;

% Plot the difference. 
t = 1:1:(N_step+1);
figure 
plot(t,MMSE);
hold on
avgMMSE = mean(MMSE);
label = sprintf("Avg: %.2f",avgMMSE);
yline(avgMMSE,'LineStyle',"--","Label",label);
plot(t,kalmanMMSE);

title("Kalman filter MMSE estimate vs. Actual MMSE");
legend("MMSE","Average MMSE","Kalman Estimate of MMSE");

%% True trajectory from the data

load("trueTarget.mat")

x_0 = [trueTarget(2,1);trueTarget(3,1);17;17];
P_0 = eye(4);

process_noise_std = 1;
measurement_noise_std = 1;

N_iter = 1000;
time = 150;
times = trueTarget(1,:);
T = times(2)-times(1);
N_step = time/T;


x_est = zeros(4,N_step+1); 
y_cv = zeros(2,N_step+1);

C = [1,0,0,0;0,1,0,0];
Qlow = 0.1^2*eye(2);
Qhigh = 100*eye(2);

MMSE = zeros(1,N_step+1);
kalmanMMSE = zeros(1,N_step+1);

covariances = cell(1,N_step+1);

for i = 1:N_iter
    % generate measurements
    for j = 1:size(trueTarget,2)
        y_cv(:,j) = [trueTarget(2,j);...
                     trueTarget(3,j)] + randn(2,1)*measurement_noise_std;
    end
    % Kalman Filtering
    % first step
    measurementCovariance = eye(2) * measurement_noise_std^2;

    KF_Low_Q = CV_KF(Qlow,x_0,P_0);
    KF_High_Q = CV_KF(Qhigh,x_0,P_0);
    TPM = [0.5,0.5;...
        0.5,0.5];

    IMM = IMMFilter(x_0,P_0,{KF_Low_Q,KF_High_Q},TPM);
    [x_est(:,1),covariances{1,1},~,~,~] = IMM.update(y_cv(:,1),measurementCovariance,0);

    for j = 2:N_step+1
        [x_est(:,j),covariances{1,j},~,~,~] = IMM.update(y_cv(:,j),measurementCovariance,T);
    end
    % MMSE calculation
    currMSE = zeros(1,N_step+1);
    currKalmanMSE = zeros(1,N_step+1);
    for j=1:N_step+1
        currMSE(j) = (x_est([1,2],j)-trueTarget([2,3],j))'*(x_est([1,2],j)-trueTarget([2,3],j));
        covMtx = covariances{1,j};
        currKalmanMSE(j) = trace(covMtx([1,2],[1,2]));
    end
    MMSE = MMSE + currMSE;
    kalmanMMSE = kalmanMMSE + currKalmanMSE;
end

MMSE = MMSE/N_iter;
kalmanMMSE = kalmanMMSE/N_iter;

% Plot the difference. 
figure 
plot(times,MMSE);
hold on
avgMMSE = mean(MMSE);
label = sprintf("Avg: %.2f",avgMMSE);
yline(avgMMSE,'LineStyle',"--","Label",label);
plot(times,kalmanMMSE);

title("Kalman filter MMSE estimate vs. Actual MMSE");
legend("MMSE","Average MMSE","Kalman Estimate of MMSE");
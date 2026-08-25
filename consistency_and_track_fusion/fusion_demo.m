% 1. Consistency 
% a) 
x_0 = [5000;5000;25;25];
sigma_noise = 2;
P_0 = diag(x_0/10.*x_0/10);
duration = 99;
T = 1;
trueTraj = GenerateCVData(x_0,duration,sigma_noise,T);
times = 0:T:duration;
%b)
S1_measurements = zeros(length(trueTraj),2);
S2_measurements = zeros(length(trueTraj),2);

S1_sigma_meas = 20;
S2_sigma_meas = 20;
for i = 1:length(trueTraj)
    S1_measurements(i,1) = trueTraj(i,1) + S1_sigma_meas*randn;
    S1_measurements(i,2) = trueTraj(i,2) + S1_sigma_meas*randn;
    S2_measurements(i,1) = trueTraj(i,1) + S2_sigma_meas*randn;
    S2_measurements(i,2) = trueTraj(i,2) + S2_sigma_meas*randn;
end

%c)
states = cell(1,100);
stateCovariances = cell(1,100);

processNoise = sigma_noise^2*eye(2,2);
R1 = S1_sigma_meas^2 * eye(2,2);
R2 = S2_sigma_meas^2 * eye(2,2);
C1 = [1,0,0,0;...
    0,1,0,0];
C2 = [1,0,0,0;...
    0,1,0,0];
C_combined = [C1;C2];
R_combined = [R1,zeros(2,2);...
              zeros(2,2),R2];

prevState = x_0;
prevStateCov = P_0;
prevTime = times(1);

for i = 1:length(trueTraj)
    dt = times(i)-prevTime;
    
    y_k = [S1_measurements(i,:)';...
           S2_measurements(i,:)'];
    A_k = getCVStateTransitionMtx(dt);
    B_k = getCVNoiseGainMtx(dt);
    x_predict = A_k*prevState;
    xP_predict = A_k*prevStateCov*A_k' + B_k*processNoise*B_k';
    
    S_k = C_combined*xP_predict*C_combined'+R_combined;
    K_k = xP_predict*C_combined'*S_k^-1;
    state = x_predict + K_k*(y_k-C_combined*x_predict);
    stateCov = xP_predict - K_k*S_k*K_k';

    states{i} = state;
    stateCovariances{i} = stateCov;

    prevState = state;
    prevStateCov = stateCov;
    prevTime = times(i);
end

%d

states = cell(1,100);
stateCovariances = cell(1,100);

processNoise = sigma_noise^2*eye(2,2);

R1 = S1_sigma_meas^2 * eye(2,2);
R2 = S2_sigma_meas^2 * eye(2,2);

KF_1 = CV_KF(processNoise,x_0,P_0);
KF_2 = CV_KF(processNoise,x_0,P_0);


prevTime = times(1);

for i = 1:length(trueTraj)
    dT = times(i)-prevTime;
    
    [state1, stateCov1] = KF_1.update(S1_measurements(i,:)',R1,dT);
    [state2, stateCov2] = KF_2.update(S2_measurements(i,:)',R2,dT);
    
    if(mod(times(i),2)==0)
        inv_SC1 = inv(stateCov1);
        inv_SC2 = inv(stateCov2);
        inv_Pf =  inv_SC1 + inv_SC2;
        Pf = inv(inv_Pf);
        xf = Pf*(inv_SC1*state1 + inv_SC2*state2);
        states{i} = xf;
        stateCovariances{i} = Pf;
    else
        states{i} = state2;
        stateCovariances{i} = stateCov2;
    end

    prevTime = times(i);
end

% e) 

states = cell(1,100);
stateCovariances = cell(1,100);

processNoise = sigma_noise^2*eye(2,2);

R1 = S1_sigma_meas^2 * eye(2,2);
R2 = S2_sigma_meas^2 * eye(2,2);

KF_1 = CV_KF(processNoise,x_0,P_0);
KF_2 = CV_KF(processNoise,x_0,P_0);


prevTime = times(1);

prev_fused_x = x_0;
prev_fused_xP = P_0;
prev_fused_time = times(1);

for i = 1:length(trueTraj)
    dT = times(i)-prevTime;

    [state1, stateCov1] = KF_1.update(S1_measurements(i,:)',R1,dT);
    [state2, stateCov2] = KF_2.update(S2_measurements(i,:)',R2,dT);

    if(mod(times(i),2)==0)
        dt_fuse = times(i) - prev_fused_time;
        A_k = getCVStateTransitionMtx(dt_fuse);
        B_k = getCVNoiseGainMtx(dt_fuse);
        fused_x_pred = A_k*prev_fused_x;
        fused_xP_pred = A_k*prev_fused_xP*A_k' + B_k*processNoise*B_k';
        inv_SC1 = inv(stateCov1);
        inv_SC2 = inv(stateCov2);
        inv_prev_pred = inv(fused_xP_pred);
        inv_Pf =  inv_SC1 + inv_SC2 - inv_prev_pred;
        Pf = inv(inv_Pf);
        xf = Pf*(inv_SC1*state1 + inv_SC2*state2 - inv_prev_pred*fused_x_pred);
        states{i} = xf;
        stateCovariances{i} = Pf;
        prev_fused_x = xf;
        prev_fused_xP = Pf;
        prev_fused_time = times(i);
    else
        states{i} = state2;
        stateCovariances{i} = stateCov2;
    end
    prevTime = times(i);
end

%f)

states = cell(1,100);
stateCovariances = cell(1,100);

processNoise = sigma_noise^2*eye(2,2);

R1 = S1_sigma_meas^2 * eye(2,2);
R2 = S2_sigma_meas^2 * eye(2,2);

KF_1 = CV_KF(processNoise,x_0,P_0);
KF_2 = CV_KF(processNoise,x_0,P_0);


prevTime = times(1);


for i = 1:length(trueTraj)
    dT = times(i)-prevTime;

    [state1, stateCov1] = KF_1.update(S1_measurements(i,:)',R1,dT);
    [state2, stateCov2] = KF_2.update(S2_measurements(i,:)',R2,dT);

    if(mod(times(i),2)==0)
        [U1,S1,~] = svd(stateCov1);
        T1 = S1^(-0.5)*U1';
        P2 = T1*stateCov2*T1';
        [U2,~,~] = svd(P2);
        T2 = U2'*T1;
    
        z1 = T2*state1;
        z2 = T2*state2;
        PZ1 = T2*stateCov1*T2';
        PZ2 = T2*stateCov2*T2';
        zf = 0*x_0;
        PZf = 0*P_0;
        for j = 1:length(x_0)
            if(PZ2(j,j)<1)
                zf(j) = z2(j);
                PZf(j,j) = PZ2(j,j);
            else
                zf(j) = z1(j);
                PZf(j,j) = PZ1(j,j);
            end
        end
        xf = inv(T2)*zf; 
        Pf = inv(T2)*PZf*inv(T2)';
        states{i} = xf;
        stateCovariances{i} = Pf;
    else
        states{i} = state2;
        stateCovariances{i} = stateCov2;
    end
    prevTime = times(i);
end

%g)

% Centralized monte carlo

Nmc = 100;
NEES = zeros(1,length(times));
SqrErr = zeros(1,length(times));
for j = 1:Nmc
    trueTraj = GenerateCVData(x_0,duration,sigma_noise,T);
    for i = 1:length(trueTraj)
        S1_measurements(i,1) = trueTraj(i,1) + S1_sigma_meas*randn;
        S1_measurements(i,2) = trueTraj(i,2) + S1_sigma_meas*randn;
        S2_measurements(i,1) = trueTraj(i,1) + S2_sigma_meas*randn;
        S2_measurements(i,2) = trueTraj(i,2) + S2_sigma_meas*randn;
    end
    
    prevState = x_0;
    prevStateCov = P_0;
    prevTime = times(1);

    for i = 1:length(trueTraj)
        dt = times(i)-prevTime;

        y_k = [S1_measurements(i,:)';...
            S2_measurements(i,:)'];
        A_k = getCVStateTransitionMtx(dt);
        B_k = getCVNoiseGainMtx(dt);
        x_predict = A_k*prevState;
        xP_predict = A_k*prevStateCov*A_k' + B_k*processNoise*B_k';

        S_k = C_combined*xP_predict*C_combined'+R_combined;
        K_k = xP_predict*C_combined'*S_k^-1;
        state = x_predict + K_k*(y_k-C_combined*x_predict);
        stateCov = xP_predict - K_k*S_k*K_k';

        states{i} = state;
        stateCovariances{i} = stateCov;

        prevState = state;
        prevStateCov = stateCov;
        prevTime = times(i);
    end

    NEES = NEES + calcNEES(states,stateCovariances,trueTraj);
    SqrErr = SqrErr + calcSqrErr(states,trueTraj);
end


NEES = NEES/Nmc;
chi_order = Nmc*length(x_0);
lowerBound = chi2inv(0.005,chi_order)/Nmc;
upperBound = chi2inv(1-0.005,chi_order)/Nmc;
figure
plot(times,NEES);
hold on
yline(lowerBound,'LineStyle','--','LineWidth',2,'Color','r');
yline(upperBound,'LineStyle','--','LineWidth',2,'Color','r');
xlabel("Time (s)");
ylabel("NEES");
title("NEES of Centralized Fusion");

RMSE = sqrt(SqrErr/Nmc);
avg = mean(RMSE);
figure
plot(times,RMSE);
hold on
avgStr = sprintf("%.4f",avg);
yline(avg,'--',['Average: ',avgStr ],'LineWidth',2,'Color','r');
xlabel("Time (s)");
ylabel("RMSE");
title("Position RMSE of Centralized Fusion");

% Naive Fusion Monte-Carlo

Nmc = 100;
NEES = zeros(1,length(times));
SqrErr = zeros(1,length(times));
for j = 1:Nmc
    trueTraj = GenerateCVData(x_0,duration,sigma_noise,T);
    for i = 1:length(trueTraj)
        S1_measurements(i,1) = trueTraj(i,1) + S1_sigma_meas*randn;
        S1_measurements(i,2) = trueTraj(i,2) + S1_sigma_meas*randn;
        S2_measurements(i,1) = trueTraj(i,1) + S2_sigma_meas*randn;
        S2_measurements(i,2) = trueTraj(i,2) + S2_sigma_meas*randn;
    end

    prevTime = times(1);

    KF_1 = CV_KF(processNoise,x_0,P_0);
    KF_2 = CV_KF(processNoise,x_0,P_0);
    
    for i = 1:length(trueTraj)
        dT = times(i)-prevTime;
        
        [state1, stateCov1] = KF_1.update(S1_measurements(i,:)',R1,dT);
        [state2, stateCov2] = KF_2.update(S2_measurements(i,:)',R2,dT);
        
        if(mod(times(i),2)==0)
            inv_SC1 = inv(stateCov1);
            inv_SC2 = inv(stateCov2);
            inv_Pf =  inv_SC1 + inv_SC2;
            Pf = inv(inv_Pf);
            xf = Pf*(inv_SC1*state1 + inv_SC2*state2);
            states{i} = xf;
            stateCovariances{i} = Pf;
        else
            states{i} = state2;
            stateCovariances{i} = stateCov2;
        end
    
        prevTime = times(i);
    end

    NEES = NEES + calcNEES(states,stateCovariances,trueTraj);
    SqrErr = SqrErr + calcSqrErr(states,trueTraj);
end


NEES = NEES/Nmc;
chi_order = Nmc*length(x_0);
lowerBound = chi2inv(0.005,chi_order)/Nmc;
upperBound = chi2inv(1-0.005,chi_order)/Nmc;
figure
plot(times,NEES);
hold on
yline(lowerBound,'LineStyle','--','LineWidth',2,'Color','r');
yline(upperBound,'LineStyle','--','LineWidth',2,'Color','r');
xlabel("Time (s)");
ylabel("NEES");
title("NEES of Naive Fusion");

RMSE = sqrt(SqrErr/Nmc);
avg = mean(RMSE);
figure
plot(times,RMSE);
hold on
avgStr = sprintf("%.4f",avg);
yline(avg,'--',['Average: ',avgStr ],'LineWidth',2,'Color','r');
xlabel("Time (s)");
ylabel("RMSE");
title("Position RMSE of Naive Fusion");

% Channel Filter Monte-Carlo


Nmc = 100;
NEES = zeros(1,length(times));
SqrErr = zeros(1,length(times));
for j = 1:Nmc
    trueTraj = GenerateCVData(x_0,duration,sigma_noise,T);
    for i = 1:length(trueTraj)
        S1_measurements(i,1) = trueTraj(i,1) + S1_sigma_meas*randn;
        S1_measurements(i,2) = trueTraj(i,2) + S1_sigma_meas*randn;
        S2_measurements(i,1) = trueTraj(i,1) + S2_sigma_meas*randn;
        S2_measurements(i,2) = trueTraj(i,2) + S2_sigma_meas*randn;
    end

    KF_1 = CV_KF(processNoise,x_0,P_0);
    KF_2 = CV_KF(processNoise,x_0,P_0);


    prevTime = times(1);

    prev_fused_x = x_0;
    prev_fused_xP = P_0;
    prev_fused_time = times(1);

    for i = 1:length(trueTraj)
        dT = times(i)-prevTime;

        [state1, stateCov1] = KF_1.update(S1_measurements(i,:)',R1,dT);
        [state2, stateCov2] = KF_2.update(S2_measurements(i,:)',R2,dT);

        if(mod(times(i),2)==0)
            dt_fuse = times(i) - prev_fused_time;
            A_k = getCVStateTransitionMtx(dt_fuse);
            B_k = getCVNoiseGainMtx(dt_fuse);
            fused_x_pred = A_k*prev_fused_x;
            fused_xP_pred = A_k*prev_fused_xP*A_k' + B_k*processNoise*B_k';
            inv_SC1 = inv(stateCov1);
            inv_SC2 = inv(stateCov2);
            inv_prev_pred = inv(fused_xP_pred);
            inv_Pf =  inv_SC1 + inv_SC2 - inv_prev_pred;
            Pf = inv(inv_Pf);
            xf = Pf*(inv_SC1*state1 + inv_SC2*state2 - inv_prev_pred*fused_x_pred);
            states{i} = xf;
            stateCovariances{i} = Pf;
            prev_fused_x = xf;
            prev_fused_xP = Pf;
            prev_fused_time = times(i);
        else
            states{i} = state2;
            stateCovariances{i} = stateCov2;
        end
        prevTime = times(i);
    end


    NEES = NEES + calcNEES(states,stateCovariances,trueTraj);
    SqrErr = SqrErr + calcSqrErr(states,trueTraj);
end


NEES = NEES/Nmc;
chi_order = Nmc*length(x_0);
lowerBound = chi2inv(0.005,chi_order)/Nmc;
upperBound = chi2inv(1-0.005,chi_order)/Nmc;
figure
plot(times,NEES);
hold on
yline(lowerBound,'LineStyle','--','LineWidth',2,'Color','r');
yline(upperBound,'LineStyle','--','LineWidth',2,'Color','r');
xlabel("Time (s)");
ylabel("NEES");
title("NEES of Channel Filter Fusion");

RMSE = sqrt(SqrErr/Nmc);
avg = mean(RMSE);
figure
plot(times,RMSE);
hold on
avgStr = sprintf("%.4f",avg);
yline(avg,'--',['Average: ',avgStr ],'LineWidth',2,'Color','r');
xlabel("Time (s)");
ylabel("RMSE");
title("Position RMSE of Channel Filter Fusion");

% LEA Monte Carlo

Nmc = 100;
NEES = zeros(1,length(times));
SqrErr = zeros(1,length(times));
for j = 1:Nmc
    trueTraj = GenerateCVData(x_0,duration,sigma_noise,T);
    for i = 1:length(trueTraj)
        S1_measurements(i,1) = trueTraj(i,1) + S1_sigma_meas*randn;
        S1_measurements(i,2) = trueTraj(i,2) + S1_sigma_meas*randn;
        S2_measurements(i,1) = trueTraj(i,1) + S2_sigma_meas*randn;
        S2_measurements(i,2) = trueTraj(i,2) + S2_sigma_meas*randn;
    end

    KF_1 = CV_KF(processNoise,x_0,P_0);
    KF_2 = CV_KF(processNoise,x_0,P_0);


    prevTime = times(1);

    prev_fused_x = x_0;
    prev_fused_xP = P_0;
    prev_fused_time = times(1);

    for i = 1:length(trueTraj)
        dT = times(i)-prevTime;

        [state1, stateCov1] = KF_1.update(S1_measurements(i,:)',R1,dT);
        [state2, stateCov2] = KF_2.update(S2_measurements(i,:)',R2,dT);

        if(mod(times(i),2)==0)
            dt_fuse = times(i) - prev_fused_time;
            A_k = getCVStateTransitionMtx(dt_fuse);
            B_k = getCVNoiseGainMtx(dt_fuse);
            fused_x_pred = A_k*prev_fused_x;
            fused_xP_pred = A_k*prev_fused_xP*A_k' + B_k*processNoise*B_k';
            inv_SC1 = inv(stateCov1);
            inv_SC2 = inv(stateCov2);
            inv_prev_pred = inv(fused_xP_pred);
            inv_Pf =  inv_SC1 + inv_SC2 - inv_prev_pred;
            Pf = inv(inv_Pf);
            xf = Pf*(inv_SC1*state1 + inv_SC2*state2 - inv_prev_pred*fused_x_pred);
            states{i} = xf;
            stateCovariances{i} = Pf;
        else
            states{i} = state2;
            stateCovariances{i} = stateCov2;
        end
        prevTime = times(i);
    end

    NEES = NEES + calcNEES(states,stateCovariances,trueTraj);
    SqrErr = SqrErr + calcSqrErr(states,trueTraj);
end


NEES = NEES/Nmc;
chi_order = Nmc*length(x_0);
lowerBound = chi2inv(0.005,chi_order)/Nmc;
upperBound = chi2inv(1-0.005,chi_order)/Nmc;
figure
plot(times,NEES);
hold on
yline(lowerBound,'LineStyle','--','LineWidth',2,'Color','r');
yline(upperBound,'LineStyle','--','LineWidth',2,'Color','r');
xlabel("Time (s)");
ylabel("NEES");
title("NEES of Largest Ellipsoid Fusion");

RMSE = sqrt(SqrErr/Nmc);
avg = mean(RMSE);
figure
plot(times,RMSE);
hold on
avgStr = sprintf("%.4f",avg);
yline(avg,'--',['Average: ',avgStr ],'LineWidth',2,'Color','r');
xlabel("Time (s)");
ylabel("RMSE");
title("Position RMSE of Largest Ellipsoid Fusion");
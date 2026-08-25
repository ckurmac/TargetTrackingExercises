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
measurements = zeros(length(trueTraj),2);
sigma_meas = 20;

for i = 1:length(trueTraj)
    measurements(i,1) = trueTraj(i,1) + sigma_meas*randn;
    measurements(i,2) = trueTraj(i,2) + sigma_meas*randn;
end

%c)
states = cell(1,100);
stateCovariances = cell(1,100);

processNoise = sigma_noise^2*eye(2,2);
R = sigma_meas^2 * eye(2,2);

KF_1 = CV_KF(processNoise,x_0,P_0);


prevTime = times(1);

for i = 1:length(trueTraj)
    dT = times(i)-prevTime;

    [state, stateCov] = KF_1.update(measurements(i,:)',R,dT);

    states{i} = state;
    stateCovariances{i} = stateCov;

    prevTime = times(i);
end

%d,e,f)

Nmc = 100;
NEES = zeros(1,length(times));
for i = 1:Nmc
    trueTraj = GenerateCVData(x_0,duration,sigma_noise,T);
    for j = 1:length(trueTraj)
        measurements(j,1) = trueTraj(j,1) + sigma_meas*randn;
        measurements(j,2) = trueTraj(j,2) + sigma_meas*randn;
    end
    
    KF_1 = CV_KF(processNoise,x_0,P_0);

    prevTime = times(1);

    for j = 1:length(trueTraj)
        dT = times(j)-prevTime;

        [state, stateCov] = KF_1.update(measurements(j,:)',R,dT);

        states{j} = state;
        stateCovariances{j} = stateCov;

        prevTime = times(j);
    end
    NEES = NEES + calcNEES(states,stateCovariances,trueTraj);
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
title("NEES of Matching Process Noise Kalman Filter");

%g)

% 0.5 sigma
Nmc = 100;
NEES = zeros(1,length(times));
for i = 1:Nmc
    trueTraj = GenerateCVData(x_0,duration,sigma_noise,T);
    for j = 1:length(trueTraj)
        measurements(j,1) = trueTraj(j,1) + sigma_meas*randn;
        measurements(j,2) = trueTraj(j,2) + sigma_meas*randn;
    end

    KF_1 = CV_KF(0.5^2*processNoise,x_0,P_0);

    prevTime = times(1);

    for j = 1:length(trueTraj)
        dT = times(j)-prevTime;

        [state, stateCov] = KF_1.update(measurements(j,:)',R,dT);

        states{j} = state;
        stateCovariances{j} = stateCov;

        prevTime = times(j);
    end
    NEES = NEES + calcNEES(states,stateCovariances,trueTraj);
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
title("NEES of $\sigma$ = 0.5*$\sigma$ Process Noise Kalman Filter",'Interpreter', 'latex');

% 2 sigma

Nmc = 100;
NEES = zeros(1,length(times));
for i = 1:Nmc
    trueTraj = GenerateCVData(x_0,duration,sigma_noise,T);
    for j = 1:length(trueTraj)
        measurements(j,1) = trueTraj(j,1) + sigma_meas*randn;
        measurements(j,2) = trueTraj(j,2) + sigma_meas*randn;
    end

    KF_1 = CV_KF(2^2*processNoise,x_0,P_0);

    prevTime = times(1);

    for j = 1:length(trueTraj)
        dT = times(j)-prevTime;

        [state, stateCov] = KF_1.update(measurements(j,:)',R,dT);

        states{j} = state;
        stateCovariances{j} = stateCov;

        prevTime = times(j);
    end
    NEES = NEES + calcNEES(states,stateCovariances,trueTraj);
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
title("NEES of $\sigma$ = 2*$\sigma$  Process Noise Kalman Filter",'Interpreter', 'latex');

% 0.5 sigma, Measurement Noise
Nmc = 100;
NEES = zeros(1,length(times));
for i = 1:Nmc
    trueTraj = GenerateCVData(x_0,duration,sigma_noise,T);
    for j = 1:length(trueTraj)
        measurements(j,1) = trueTraj(j,1) + sigma_meas*randn;
        measurements(j,2) = trueTraj(j,2) + sigma_meas*randn;
    end

    KF_1 = CV_KF(processNoise,x_0,P_0);

    prevTime = times(1);

    for j = 1:length(trueTraj)
        dT = times(j)-prevTime;

        [state, stateCov] = KF_1.update(measurements(j,:)',0.5^2*R,dT);

        states{j} = state;
        stateCovariances{j} = stateCov;

        prevTime = times(j);
    end
    NEES = NEES + calcNEES(states,stateCovariances,trueTraj);
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
title("NEES of $\sigma$ = 0.5*$\sigma$ Measurement Noise Kalman Filter",'Interpreter', 'latex');

% 2 sigma, measurement noise

Nmc = 100;
NEES = zeros(1,length(times));
for i = 1:Nmc
    trueTraj = GenerateCVData(x_0,duration,sigma_noise,T);
    for j = 1:length(trueTraj)
        measurements(j,1) = trueTraj(j,1) + sigma_meas*randn;
        measurements(j,2) = trueTraj(j,2) + sigma_meas*randn;
    end

    KF_1 = CV_KF(processNoise,x_0,P_0);

    prevTime = times(1);

    for j = 1:length(trueTraj)
        dT = times(j)-prevTime;

        [state, stateCov] = KF_1.update(measurements(j,:)',2^2*R,dT);

        states{j} = state;
        stateCovariances{j} = stateCov;

        prevTime = times(j);
    end
    NEES = NEES + calcNEES(states,stateCovariances,trueTraj);
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
title("NEES of $\sigma$ = 2*$\sigma$ Measurement Noise Kalman Filter",'Interpreter', 'latex');
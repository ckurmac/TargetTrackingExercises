clear all;
% Generate the true measurements.
x_0 = [5000;5000;25;25];
sigma_noise = 2;
duration = 100;
T = 1;

trueTraj = GenerateCVData(x_0,duration,sigma_noise,T);
% 
% figure;
% hold on;
% plot(trueTraj(:,1), trueTraj(:,2), 'LineStyle','none','Marker','x', 'MarkerSize', 10, 'DisplayName', 'Path');
% quiver(trueTraj(:,1), trueTraj(:,2), trueTraj(:,3), trueTraj(:,4), 0.08, 'r', 'LineWidth', 1, 'MaxHeadSize', 0.5, 'DisplayName', 'Velocity Vector');
% 
% xlabel('X Position');
% ylabel('Y Position');
% title('Trajectory and Velocity Visualization');
% grid on;
% legend('Location', 'best');
% 
% axis equal; 
% hold off;
%%
% Generate Clutter


beta_fa = 1e-7;
V = 10000*10000; % 10 km^2
clutter_sets = cell(200,1);
for i = 1:200 % not generic for different periods. Will handle later .
    mk = poissrnd(beta_fa*V);
    x_val = 10000*rand(mk,1);
    y_val = 10000*rand(mk,1);
    clutter_sets{i} = [x_val,y_val];
end
%% 
% Generate Measurements
t_s = randi([0,50]);
measurement_sets = cell(200,1);
P_d = 0.9;
meas_noise_std = 20;
for i = 1:200
    curr_set = clutter_sets{i};
    if(i-1>=t_s && i-1<=duration) % start time has passed.
        isSeen = rand;
        if(isSeen<=P_d) %seen
            curr_meas = [trueTraj(i,1),trueTraj(i,2)] + (2*rand(1,2)-1)*meas_noise_std;
            curr_set = [curr_set;curr_meas];          
        end
    end
    measurement_sets{i} = curr_set;
end

%%
% NN tracker

P_g = 0.99;
gate_threshold = chi2inv(P_g,2);
meas_cov = [meas_noise_std^2,0;...
    0,meas_noise_std^2];

% Sigmas for covariance initiation of the filter.
posSig = 50; % V*T
velSig = 50; % V

nnTracker = NNTracker(100,2,3,2,2, gate_threshold, posSig, velSig);

for i = 1:200 % full length of the scenario, not generic for different periods. Will handle later .
    detSet = measurement_sets{i};
    dets = buildDetectionArray(detSet, meas_cov, i);
    [nnInitiators, nnConfirmed] = nnTracker.step(dets);
    fprintf("NN scan %d: %d initiators, %d confirmed.\n", i, numel(nnInitiators), numel(nnConfirmed));
end

% PDA tracker
P_g = 0.99;
gate_threshold = chi2inv(P_g,2);
meas_cov = [meas_noise_std^2,0;...
    0,meas_noise_std^2];

% Sigmas for covariance initiation of the filter.
posSig = 50; % V*T
velSig = 50; % V

pdaTracker = PDATracker(100,2,3,2,2, gate_threshold, posSig, velSig, P_d, P_g, beta_fa);

for i = 1:200 % full length of the scenario, not generic for different periods. Will handle later .
    detSet = measurement_sets{i};
    dets = buildDetectionArray(detSet, meas_cov, i);
    [pdaInitiators, pdaConfirmed] = pdaTracker.step(dets);
    fprintf("PDA scan %d: %d initiators, %d confirmed.\n", i, numel(pdaInitiators), numel(pdaConfirmed));
end
% Generate the true measurements.
x_0 = [5000;5000;25;25];
sigma_noise = 2;
duration = 100;
T = 1;

trueTraj = GenerateCVData(x_0,duration,sigma_noise,T);

figure;
hold on;
plot(trueTraj(:,1), trueTraj(:,2), 'LineStyle','none','Marker','x', 'MarkerSize', 10, 'DisplayName', 'Path');
quiver(trueTraj(:,1), trueTraj(:,2), trueTraj(:,3), trueTraj(:,4), 0.08, 'r', 'LineWidth', 1, 'MaxHeadSize', 0.5, 'DisplayName', 'Velocity Vector');

xlabel('X Position');
ylabel('Y Position');
title('Trajectory and Velocity Visualization');
grid on;
legend('Location', 'best');

axis equal; 
hold off;
%%
% Generate Clutter

beta_fa = 1e-7;
V = 10000*10000; % 10 km^2
clutter_sets = cell(200,1);
for i = 1:200
    mk = poissrnd(beta_fa*V);
    x_val = 10000*rand(mk,1);
    y_val = 10000*rand(mk,1);
    clutter_sets{i} = [x_val,y_val];
end
%% 
% Generate Measurements
t_s = randi(0,50);
measurement_sets = cell(200,1);
P_d = 0.9;
meas_noise_std = 20;
for i = 1:200
    curr_set = clutter_sets{i};
    if(i-1>=t_s) % start time has passed.
        isSeen = rand;
        if(isSeen<=P_d) %seen
            curr_meas = [trueTraj(i,1),trueTraj(i,2)] + (2*rand(1,2)-1)*meas_noise_std;
            curr_set = [curr_set;curr_meas];          
        end
    end
    measurement_sets{i} = curr_set;
end

%%
% initiation algorithm
P_g = 0.99;
gate_threshold = chi2inv(P_g,2);

for i = 1:200 % full length of the scenario


end
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

% monte carlo testing for curiosity
IterN = 1000;
initCount = 0;
for t = 1:IterN

beta_fa = 1e-6;
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
% initiation algorithm
P_g = 0.99;
gate_threshold = chi2inv(P_g,2);
tracker = BaseTracker(100,2,3,2,2); % a base tracker implementation for keeping initiators.
meas_cov = [meas_noise_std^2,0;...
            0,meas_noise_std^2];
isInitiated = false;

for i = 1:200 % full length of the scenario, not generic for different periods. Will handle later .
    detSet = measurement_sets{i};
    isDetAssigned = zeros(1,size(detSet,1));
    trackIDsAtBeginning = tracker.liveTrackIDs; % to keep the order correct.
    for j = 1:tracker.TrackNum
        currTrackID = trackIDsAtBeginning(j); 
        currTrackIdx = tracker.getTrackIndex(currTrackID);
        if(currTrackIdx == 0)
            fprintf("already deleted track\n");
        end
        minDist = inf;
        minDistDetIdx = 0;
        minDistDet = getDetectionStruct;
        for k = 1:size(detSet,1)
            if(isDetAssigned(k))
                continue; % skip the assigned dets.
            end
            detObj = getDetectionStruct();
            detObj.Measurement = detSet(k,:)';
            detObj.MeasurementCovariance = meas_cov;
            detObj.MeasurementTime = i;
            dist = tracker.TracksList{currTrackIdx}.distance(detObj);
            if(dist<=gate_threshold)
                minDist = dist;
                minDistDetIdx = k;
                minDistDet = detObj;
            end
        end
        if(minDist<=gate_threshold)
            isDetAssigned(minDistDetIdx) = 1;
            tracker.TracksList{currTrackIdx}.updateNN(minDistDet);
            tracker.checkTrackStatus(currTrackID,true);
            if(tracker.TracksList{currTrackIdx}.InitiationState(1) == 2) %is Confirmed
                %fprintf("Track %d is confirmed. Track initiation achieved at Time %d s.\n",currTrackID,i);
                isInitiated = true;
                initCount = initCount+1;
                break;
            end
        else % unassigned track/initiator
            tracker.checkTrackStatus(currTrackID,false);
            if(tracker.TracksList{currTrackIdx}.InitiationState(1) == 0) %deletedTrack.
                tracker.removeTrack(currTrackID);
            end
        end
    end
    if(isInitiated)
        break;
    end
    for j = 1:size(detSet,1)
        if(isDetAssigned(j))
            continue;
        else % initiate tracks with unassigned dets.
            detObj.Measurement = detSet(j,:)';
            detObj.MeasurementCovariance = meas_cov;
            detObj.MeasurementTime = i;
            posSig = 50; % V*T
            velSig = 50; % V
            tracker.initiateTrack(detObj,posSig,velSig);
        end
    end
end
end
fprintf("Successful init Count %d.\n",initCount);
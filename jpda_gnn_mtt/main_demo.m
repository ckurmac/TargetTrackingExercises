clear all;

%% Scenario truth
% x1data, x2data, x3data are each [time; x; y; vx; vy].
load("xdata.mat");
targets = {x1data, x2data, x3data};

t = 0:4:300;              % scan times; this range drives every loop below.
numSteps = numel(t);

%% Scenario / clutter / measurement parameters
beta_fa = 1e-7;
V = 10000*10000;         % surveillance region area (10 km x 10 km).
P_d = 0.9;
meas_noise_std = 50;
meas_cov = [meas_noise_std^2, 0; 0, meas_noise_std^2];

%% Tracker parameters
P_g = 0.99;
gate_threshold = chi2inv(P_g, 2);
posSig = 50;             % position sigma for filter initiation.
velSig = 50;             % velocity sigma for filter initiation.
maxTrackNum = 100;
confM = 2; confN = 3; delM = 2; delN = 2;

% Parametrized tracker set. NNTracker takes the common args; the others also
% take P_d, P_g, beta_fa. Each factory builds a fresh tracker per MC run.
common = {maxTrackNum, confM, confN, delM, delN, gate_threshold, posSig, velSig};
extra  = {P_d, P_g, beta_fa};
trackerFactories = { @() NNTracker(common{:}), ...
                     @() PDATracker(common{:}, extra{:}), ...
                     @() GNNTracker(common{:}, extra{:}), ...
                     @() JPDATracker(common{:}, extra{:}) };
trackerNames = {"NN", "PDA", "GNN", "JPDA"};
numTrackers = numel(trackerFactories);

%% Monte-Carlo run
N = 200;                  % number of Monte-Carlo runs (set N = 1 to smoke-test).

% Ground truth is deterministic across runs -> build once.
truthLog = buildTruths(t, targets);

% trackLog{tt}{n,k} = struct array of confirmed tracks (trackOSPAMetric format)
% for tracker tt, run n, scan k.
trackLog = cell(numTrackers, 1);
for tt = 1:numTrackers
    trackLog{tt} = cell(N, numSteps);
end

for n = 1:N
    % Fresh measurement + clutter realization for this run.
    measurement_sets = generateMeasurements(t, targets, P_d, beta_fa, V, meas_noise_std);

    for tt = 1:numTrackers
        tracker = trackerFactories{tt}();
        for k = 1:numSteps
            dets = buildDetectionArray(measurement_sets{k}, meas_cov, t(k));
            [~, confirmed] = tracker.step(dets, t(k));   % real time t(k), not the scan index.
            trackLog{tt}{n, k} = tracksToStructArray(confirmed);
        end
    end
    fprintf("MC run %d/%d complete.\n", n, N);
end

save("mc_results.mat", "truthLog", "trackLog", "t", "trackerNames", "N");
fprintf("Saved track/truth logs to mc_results.mat\n");

%% OSPA evaluation (requires the Sensor Fusion and Tracking Toolbox)
if exist('trackOSPAMetric') > 0 %#ok<EXIST>
    distances = {'posabserr', 'posnees'};
    meanOspa = cell(numel(distances), numTrackers);   % per-scan mean OSPA curve.
    for d = 1:numel(distances)
        for tt = 1:numTrackers
            ospaRuns = zeros(N, numSteps);
            for n = 1:N
                metric = trackOSPAMetric('Distance', distances{d});
                for k = 1:numSteps
                    ospaRuns(n, k) = metric(trackLog{tt}{n, k}, truthLog{k});
                end
            end
            meanOspa{d, tt} = mean(ospaRuns, 1);
        end
    end

    fprintf("\n=== Mean OSPA (averaged over %d runs and %d scans) ===\n", N, numSteps);
    fprintf("%-8s %-14s %-14s\n", "Tracker", distances{1}, distances{2});
    for tt = 1:numTrackers
        fprintf("%-8s %-14.3f %-14.3f\n", trackerNames{tt}, ...
                mean(meanOspa{1, tt}), mean(meanOspa{2, tt}));
    end

    save("mc_results.mat", "meanOspa", "distances", "-append");
else
    fprintf(["\ntrackOSPAMetric not found -- install the Sensor Fusion and " ...
             "Tracking Toolbox to compute OSPA.\nTrack/truth logs are saved in " ...
             "mc_results.mat for later evaluation.\n"]);
end

%% ---- local functions ----

function measurement_sets = generateMeasurements(t, targets, P_d, beta_fa, V, meas_noise_std)
    % One measurement set per scan: Poisson clutter over the surveillance
    % region plus each active target detected with probability P_d.
    numSteps = numel(t);
    measurement_sets = cell(numSteps, 1);
    for i = 1:numSteps
        mk = poissrnd(beta_fa * V);
        curr_set = [10000*rand(mk,1), 10000*rand(mk,1)];   % clutter, [mk x 2].
        for tgt = 1:numel(targets)
            data = targets{tgt};
            if t(i) >= data(1,1) && t(i) <= data(1,end)
                if rand <= P_d
                    idx = find(data(1,:) == t(i));
                    if ~isempty(idx)
                        curr_meas = [data(2,idx), data(3,idx)] + (2*rand(1,2)-1)*meas_noise_std;
                        curr_set = [curr_set; curr_meas];
                    end
                end
            end
        end
        measurement_sets{i} = curr_set;
    end
end

function truthLog = buildTruths(t, targets)
    % truthLog{k} = struct array of truths active at scan k, in the format
    % trackOSPAMetric expects (PlatformID, Position, Velocity).
    numSteps = numel(t);
    truthLog = cell(1, numSteps);
    for k = 1:numSteps
        truths = struct('PlatformID', {}, 'Position', {}, 'Velocity', {});
        for tgt = 1:numel(targets)
            data = targets{tgt};
            idx = find(data(1,:) == t(k));
            if ~isempty(idx)
                s.PlatformID = tgt;
                s.Position = [data(2,idx), data(3,idx), 0];
                s.Velocity = [data(4,idx), data(5,idx), 0];
                truths(end+1) = s; %#ok<AGROW>
            end
        end
        truthLog{k} = truths;
    end
end

function structs = tracksToStructArray(trackCell)
    % Convert the cell of confirmed TrackObj handles into a track struct array.
    structs = struct('TrackID', {}, 'State', {}, 'StateCovariance', {});
    for i = 1:numel(trackCell)
        structs(end+1) = trackToStruct(trackCell{i}); %#ok<AGROW>
    end
end

function s = trackToStruct(track)
    % Convert a TrackObj into a trackOSPAMetric-compatible track struct.
    % trackOSPAMetric's default motion model is 3-D constvel, i.e. a 6-element
    % state [x; vx; y; vy; z; vz], and the truths are 3-D (z = 0). The filter
    % state is [x; y; vx; vy], so reorder into [x; vx; y; vy] and pad the z/vz
    % dimensions with zeros. The z/vz covariance is set to 1 (any positive value
    % works: the z error is always 0, so it contributes 0 to the NEES distance).
    x = track.Filter.x;      % [x; y; vx; vy]
    P4 = track.Filter.xP([1 3 2 4], [1 3 2 4]);   % reorder to [x; vx; y; vy]
    s.TrackID = track.TrackID;
    s.State = [x(1); x(3); x(2); x(4); 0; 0];
    s.StateCovariance = blkdiag(P4, eye(2));
end

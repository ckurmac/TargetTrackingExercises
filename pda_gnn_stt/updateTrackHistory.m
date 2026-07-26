function trackHistory = updateTrackHistory(trackHistory, tracker, currentTime, gateCenters, gateCovs)
    if nargin < 4
        gateCenters = {};
    end
    if nargin < 5
        gateCovs = {};
    end

    activeIDs = tracker.liveTrackIDs(1:tracker.TrackNum);
    activeMask = false(1, numel(trackHistory));

    for i = 1:numel(trackHistory)
        if ~isempty(activeIDs)
            activeMask(i) = any(activeIDs == trackHistory(i).TrackID);
        end
    end

    for i = 1:numel(activeIDs)
        trackID = activeIDs(i);
        trackIdx = tracker.getTrackIndex(trackID);
        trackObj = tracker.TracksList{trackIdx};

        historyIdx = 0;
        for j = 1:numel(trackHistory)
            if trackHistory(j).TrackID == trackID
                historyIdx = j;
                break;
            end
        end

        if historyIdx == 0
            historyIdx = numel(trackHistory) + 1;
            trackHistory(historyIdx).TrackID = trackID;
            trackHistory(historyIdx).x = [];
            trackHistory(historyIdx).y = [];
            trackHistory(historyIdx).status = [];
            trackHistory(historyIdx).cov = {};
            trackHistory(historyIdx).gateCenter = {};
            trackHistory(historyIdx).gateCov = {};
            trackHistory(historyIdx).active = true;
        end

        trackHistory(historyIdx).x(end+1,1) = trackObj.Filter.x(1);
        trackHistory(historyIdx).y(end+1,1) = trackObj.Filter.x(2);
        trackHistory(historyIdx).status(end+1,1) = trackObj.InitiationState(1);
        trackHistory(historyIdx).cov{end+1,1} = trackObj.Filter.xP;
        if ~isempty(gateCenters) && numel(gateCenters) >= i && ~isempty(gateCenters{i})
            trackHistory(historyIdx).gateCenter{end+1,1} = gateCenters{i};
        else
            trackHistory(historyIdx).gateCenter{end+1,1} = [];
        end
        if ~isempty(gateCovs) && numel(gateCovs) >= i && ~isempty(gateCovs{i})
            trackHistory(historyIdx).gateCov{end+1,1} = gateCovs{i};
        else
            trackHistory(historyIdx).gateCov{end+1,1} = [];
        end
        trackHistory(historyIdx).active = true;
    end

    for i = 1:numel(trackHistory)
        if i > numel(activeMask) || ~activeMask(i)
            trackHistory(i).active = false;
        end
    end

    if ~isempty(trackHistory)
        keepMask = [trackHistory.active] | arrayfun(@(s) ~isempty(s.x), trackHistory);
        trackHistory = trackHistory(keepMask);
    end
end
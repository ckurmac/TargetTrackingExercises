function renderTrackerVisualization(figHandle, trackerLabel, trueTraj, measurementSets, currentTime, tracker, trackHistory, gateThreshold, measCov, deletionEvents)
    if nargin < 10
        deletionEvents = struct('TrackID', {}, 'x', {}, 'y', {}, 'time', {});
    end

    figure(figHandle);
    clf(figHandle);
    ax = axes(figHandle);
    set(figHandle, 'Color', [0.12 0.12 0.12]);
    set(ax, 'Color', [0.12 0.12 0.12], 'XColor', [0.95 0.95 0.95], 'YColor', [0.95 0.95 0.95], 'ZColor', [0.95 0.95 0.95]);
    hold(ax, 'on');
    grid(ax, 'on');
    ax.GridColor = [0.45 0.45 0.45];
    ax.GridAlpha = 0.28;
    ax.MinorGridColor = [0.30 0.30 0.30];
    ax.MinorGridAlpha = 0.18;
    axis(ax, 'equal');
    ax.Box = 'on';
    xlim(ax, [0, 10000]);
    ylim(ax, [0, 10000]);
    xlabel(ax, 'X Position', 'Color', [0.95 0.95 0.95]);
    ylabel(ax, 'Y Position', 'Color', [0.95 0.95 0.95]);
    title(ax, sprintf('%s at time step %d', trackerLabel, currentTime), 'Color', [0.98 0.98 0.98]);

    if ~isempty(trueTraj)
        trajEndIdx = min(currentTime, size(trueTraj, 1));
        plot(ax, trueTraj(1:trajEndIdx,1), trueTraj(1:trajEndIdx,2), 'Color', [0.82 0.82 0.82], 'LineWidth', 1.1, 'HandleVisibility', 'off');
    end

    plotDetectionsHistory(ax, measurementSets, currentTime);

    detectionLegendHandle = plot(ax, nan, nan, 'x', 'Color', [0.95 0.70 0.20], 'LineWidth', 1.4, 'MarkerSize', 8, 'DisplayName', 'Detections');
    confirmedLegendHandle = plot(ax, nan, nan, '-', 'Color', [0.00 0.75 1.00], 'LineWidth', 1.9, 'Marker', '>', 'MarkerSize', 8, 'MarkerFaceColor', [0.00 0.75 1.00], 'DisplayName', 'Confirmed track');
    initiatorLegendHandle = plot(ax, nan, nan, 'o', 'Color', [0.80 0.80 0.80], 'MarkerFaceColor', [0.80 0.80 0.80], 'LineStyle', 'none', 'MarkerSize', 6, 'DisplayName', 'Track initiator');
    gateLegendHandle = plot(ax, nan, nan, '--', 'Color', [0.95 0.95 0.95], 'LineWidth', 1.1, 'DisplayName', 'Gate');

    activeIDs = tracker.liveTrackIDs(1:tracker.TrackNum);
    for i = 1:numel(activeIDs)
        trackID = activeIDs(i);
        historyIdx = find([trackHistory.TrackID] == trackID, 1, 'first');
        if isempty(historyIdx)
            continue;
        end

        history = trackHistory(historyIdx);
        trackColor = getTrackColor(trackID);
        initiatorColor = blendColor(trackColor, [1 1 1], 0.45);

        confirmedIdx = find(history.status == 2, 1, 'first');
        if isempty(confirmedIdx)
            plot(ax, history.x, history.y, '--', 'Color', initiatorColor, 'LineWidth', 1.3, 'HandleVisibility', 'off');
            plot(ax, history.x(end), history.y(end), 'o', 'Color', initiatorColor, 'MarkerFaceColor', initiatorColor, 'MarkerSize', 6, 'HandleVisibility', 'off');
        else
            if confirmedIdx > 1
                plot(ax, history.x(1:confirmedIdx-1), history.y(1:confirmedIdx-1), '--', 'Color', initiatorColor, 'LineWidth', 1.2, 'HandleVisibility', 'off');
            end
            plot(ax, history.x(confirmedIdx:end), history.y(confirmedIdx:end), '-', 'Color', trackColor, 'LineWidth', 1.9, 'DisplayName', sprintf('Track %d', trackID), 'HandleVisibility', 'off');
            plot(ax, history.x(end), history.y(end), '>', 'Color', trackColor, 'MarkerFaceColor', trackColor, 'MarkerSize', 8, 'LineStyle', 'none', 'HandleVisibility', 'off');
        end

        if history.status(end) == 1
            gateColor = initiatorColor;
        else
            gateColor = trackColor;
        end

        gateCenter = history.gateCenter{end};
        gateCov = history.gateCov{end};
        if ~isempty(gateCenter) && ~isempty(gateCov)
            drawGateEllipse(ax, gateCenter, gateCov, gateThreshold, gateColor);
        elseif tracker.getTrackIndex(trackID) ~= 0
            currTrack = tracker.TracksList{tracker.getTrackIndex(trackID)};
            gateCenter = currTrack.Filter.C * currTrack.Filter.x;
            gateCov = currTrack.Filter.C * currTrack.Filter.xP * currTrack.Filter.C' + measCov;
            drawGateEllipse(ax, gateCenter, gateCov, gateThreshold, gateColor);
        end

    end

    for i = 1:numel(deletionEvents)
        if deletionEvents(i).time <= currentTime
            plot(ax, deletionEvents(i).x, deletionEvents(i).y, 'x', 'Color', [0.90 0.15 0.15], ...
                'MarkerSize', 8, 'LineWidth', 1.6, 'HandleVisibility', 'off');
        end
    end

    lgd = legend(ax, [detectionLegendHandle, confirmedLegendHandle, initiatorLegendHandle, gateLegendHandle], ...
        {'Detections', 'Confirmed track', 'Track initiator', 'Gate'}, 'Location', 'best');
    set(lgd, 'TextColor', [0.96 0.96 0.96], 'Color', [0.16 0.16 0.16], 'EdgeColor', [0.40 0.40 0.40]);
    drawnow limitrate;
end

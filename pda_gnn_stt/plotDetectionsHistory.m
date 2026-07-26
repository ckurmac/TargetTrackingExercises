function plotDetectionsHistory(ax, measurementSets, currentTime)
    for stepIdx = 1:currentTime
        detSet = measurementSets{stepIdx};
        if isempty(detSet)
            continue;
        end

        recency = (stepIdx - 1) / max(currentTime - 1, 1);
        color = [0.85 + 0.15 * recency, 0.45 + 0.45 * recency, 0.08 + 0.20 * recency];
        alphaValue = 0.35 + 0.55 * recency;
        sizeValue = 8 + 4 * recency;

        plot(ax, detSet(:,1), detSet(:,2), 'x', 'Color', color, 'MarkerSize', sizeValue / 2, ...
            'LineWidth', 1.0 + 0.3 * recency, 'HandleVisibility', 'off');
    end
end
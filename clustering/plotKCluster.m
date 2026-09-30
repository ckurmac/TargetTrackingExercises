function plotKCluster(rnk, x, means)
%PLOTCLUSTERS Visualize k-means assignments and cluster means.
%   rnk   : N-by-K responsibility matrix (rnk(i,j) = 1 if point i is in cluster j)
%   x     : 2-by-N data points
%   means : 2-by-K cluster means
%
%   Points are drawn as dots and means as crosses, both colored from colorArray.

    colorArray = [ ...
        0.0000, 0.4470, 0.7410; ... % Blue
        0.8500, 0.3250, 0.0980; ... % Orange
        0.9290, 0.6940, 0.1250; ... % Yellow
        0.4940, 0.1840, 0.5560; ... % Purple
        0.4660, 0.6740, 0.1880; ... % Green
        0.9000, 0.4500, 0.7000; ... % Pink
        0.6350, 0.0780, 0.1840; ... % Dark red
        0.4600, 0.4600, 0.4600; ... % Gray
        0.0000, 0.6000, 0.5500; ... % Teal
        0.4000, 0.2600, 0.1300  ... % Brown
    ];

    K = size(means, 2);
    nColors = size(colorArray, 1);

    % Cluster index per point; points with no assignment yet get index 0
    [~, idx] = max(rnk, [], 2);
    idx(sum(rnk, 2) == 0) = 0;

    cla;
    hold on;

    % Unassigned points (e.g. before the first iteration)
    unassigned = (idx == 0);
    if any(unassigned)
        plot(x(1, unassigned), x(2, unassigned), '.', ...
            'Color', [0.8 0.8 0.8], 'MarkerSize', 16);
    end

    for k = 1:K
        c = colorArray(mod(k - 1, nColors) + 1, :);

        % Points of cluster k as dots
        pts = (idx == k);
        plot(x(1, pts), x(2, pts), '.', 'Color', c, 'MarkerSize', 16);

        % Mean as a cross, with a dark outline so it stays visible on top of its own points
        plot(means(1, k), means(2, k), 'x', 'Color', 'k', ...
            'MarkerSize', 14, 'LineWidth', 5);
        plot(means(1, k), means(2, k), 'x', 'Color', c, ...
            'MarkerSize', 14, 'LineWidth', 2.5);
    end

    hold off;
    axis equal;
    grid on;
    box on;
    drawnow;
end
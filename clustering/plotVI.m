function plotVI(responsibilities, x, means, covariances, pi_k,isActive)
%PLOTVI Visualize a Bayesian Gaussian mixture fitted with variational inference.
%   responsibilities : N-by-K soft assignments E[z_nk] (rows sum to 1)
%   x                : 2-by-N data points
%   means            : 2-by-K posterior mean parameters (gw_means)
%   covariances      : 1-by-K cell array of 2-by-2 matrices, or 2-by-2-by-K array
%                      (here inv(nu_k * W_k), the inverse of the expected precision)
%   pi_k             : (optional) 1-by-K expected mixing weights alpha_k / sum(alpha).
%                      Components with weight below 1% are treated as pruned and
%                      drawn faded with a dashed ellipse, since VI tends to switch
%                      off unneeded components.
%
%   Points are colored by the responsibility-weighted mean of the component colors.
%   Each component is drawn as a 95% covariance ellipse and a cross at its mean,
%   both in the component's own color.

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
    compColors = colorArray(mod((1:K) - 1, nColors) + 1, :);   % K-by-3


    % Normalize rows in case they don't sum exactly to 1
    R = responsibilities ./ max(sum(responsibilities, 2), eps);

    % Weighted mean of component colors for every point (N-by-3)
    pointColors = min(max(R * compColors, 0), 1);

    chi2_95 = 5.9915;            % 95% quantile of chi-square with 2 dof
    t = linspace(0, 2*pi, 100);
    circle = [cos(t); sin(t)];

    cla;
    hold on;

    % Points as dots
    scatter(x(1, :), x(2, :), 28, pointColors, 'filled');

    for k = 1:K
        c = compColors(k, :);

        % Get the k-th covariance from either a cell array or a 3-D array
        if iscell(covariances)
            S = covariances{k};
        else
            S = covariances(:, :, k);
        end
        S = (S + S') / 2;                        % enforce symmetry
        [V, D] = eig(S);
        D = max(D, 0);                           % guard against tiny negative eigenvalues

        ellipse = V * sqrt(D) * sqrt(chi2_95) * circle + means(:, k);

        if isActive(k)
            % Active component: filled ellipse, solid outline, bold cross
            fill(ellipse(1, :), ellipse(2, :), c, 'FaceAlpha', 0.12, 'EdgeColor', 'none');
            plot(ellipse(1, :), ellipse(2, :), '-', 'Color', c, 'LineWidth', 2);
            plot(means(1, k), means(2, k), 'x', 'Color', 'k', 'MarkerSize', 14, 'LineWidth', 5);
            plot(means(1, k), means(2, k), 'x', 'Color', c,   'MarkerSize', 14, 'LineWidth', 2.5);
        else
            % % Pruned component: dashed ellipse, thin cross, no fill
            % plot(ellipse(1, :), ellipse(2, :), '--', 'Color', c, 'LineWidth', 1);
            % plot(means(1, k), means(2, k), 'x', 'Color', c, 'MarkerSize', 10, 'LineWidth', 1);
        end
    end

    hold off;
    axis equal;
    grid on;
    box on;
    drawnow;
end
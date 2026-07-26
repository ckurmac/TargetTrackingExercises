function gateHandle = drawGateEllipse(ax, center, covMatrix, gamma, colorValue)
    covMatrix = (covMatrix + covMatrix') / 2;
    theta = linspace(0, 2*pi, 100);
    unitCircle = [cos(theta); sin(theta)];

    try
        upperFactor = chol(covMatrix, 'upper');
        ellipsePoints = sqrt(gamma) * upperFactor' * unitCircle;
    catch
        [V, D] = eig(covMatrix);
        ellipsePoints = V * sqrt(gamma * D) * unitCircle;
    end

    xPoints = center(1) + ellipsePoints(1, :);
    yPoints = center(2) + ellipsePoints(2, :);

    gateHandle = plot(ax, xPoints, yPoints, '--', 'Color', colorValue, 'LineWidth', 1.1, ...
        'HandleVisibility', 'off', 'Tag', 'TrackerGate');
end
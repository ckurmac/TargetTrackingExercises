function draw_ellipse(center, Cov, gamma,toLegend)
    [V, D] = eig(Cov);
    
    t = linspace(0, 2*pi, 100);
    circle = [cos(t); sin(t)];
    
    ellipse_points = V * sqrt(gamma * D) * circle;
    
    x_points = center(1) + ellipse_points(1, :);
    y_points = center(2) + ellipse_points(2, :);
    if(toLegend)
        plot(x_points, y_points, 'Color', [1, 0, 0, 0.4],'LineStyle','--', 'LineWidth',0.5,'DisplayName',"Covariance Ellipse");
    else
        plot(x_points, y_points, 'Color', [1, 0, 0, 0.4],'LineStyle','--', 'LineWidth',0.5,'HandleVisibility','off');
    end
end
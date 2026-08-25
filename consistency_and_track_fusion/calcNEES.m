function [NEES] = calcNEES(est_states,est_stateCovs,true_State)
    NEES = zeros(1,length(true_State));
    for i = 1:length(true_State)
        x_est = est_states{i};
        P_est = est_stateCovs{i};
        x_true = true_State(i,:)';
        NEES(i) = (x_true-x_est)'*(P_est\(x_true-x_est));
    end
end
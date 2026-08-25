function [SqrErr] = calcSqrErr(est_states,true_State)
    SqrErr = zeros(1,length(true_State));
    for i = 1:length(true_State)
        x_est = est_states{i};
        x_true = true_State(i,:)';
        SqrErr(i) = (x_est-x_true)'*(x_est-x_true);
    end
end
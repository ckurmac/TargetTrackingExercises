function [trajectory] = GenerateCVData(init_state,duration,noise_std,T)
    % create synthetic data for a 2-D CV WNA model.
    % init_state: 4x1 state vector, [x y vx vy]^T
    % duration: motion duration
    % noise_std: standard deviation of white noise accel.
    % T: sampling period of the motion, in terms of seconds.
    % trajectory: Nx4 array of measured points [x,y,vx,vy]
    
    trajectory(1,:) = init_state';
    idx = 2;
    prev_state = init_state;
    pointNum = duration/T;
    for i = 1:pointNum
        A = getCVStateTransitionMtx(T);
        G = getCVNoiseGainMtx(T);
        noise = randn(2,1)*noise_std;
        next_state = A*prev_state+G*noise;
        prev_state = next_state;
        trajectory(idx,:) = next_state';
        idx = idx+1;
    end
end
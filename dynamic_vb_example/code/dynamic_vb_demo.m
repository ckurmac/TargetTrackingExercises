lambda_actual = 1; 
sigma_x = 1;
mu_x = 0.5;
sigma_w = 1;


N_iter = 100;
N_sample = 10000; 

% generate data

y = zeros(1,N_sample);
x_init = 0;
x = x_init;
for i = 1:N_sample
    x_next = x + normrnd(0,sigma_w);
    x = x_next;
    y(i) = x_next + normrnd(0,1/sqrt(lambda_actual));
end


lambda_init = 0.01;


%% EM Algorithm

lambda_arr_em = zeros(1,N_iter);
lambda = lambda_init; % initial value for lambda.
x_all = zeros(1,N_sample);
sigma_all = zeros(1,N_sample);
for i = 1:N_iter
    % E step, calculate the kalman smoother for getting posterior of the latent
    % variables.
    for j = 1:N_sample
        %first, use kalman update.
        if j == 1
            mu_prior = x_init;
            sigma_prior = sigma_w^2;
        else
            mu_prior = x_all(j-1);
            sigma_prior = sigma_all(j-1) + sigma_w^2;
        end
        kalmanGain = sigma_prior/(sigma_prior + 1/lambda);
        mu_post = mu_prior + kalmanGain*(y(j) - mu_prior);
        sigma_post = (1 - kalmanGain)*sigma_prior;
        x_all(j) = mu_post;
        sigma_all(j) = sigma_post;
    end

    for j=1:N_sample
        %then, RTS smoother
        idx = N_sample - j;
        if idx > 0
            smootherGain = sigma_all(idx)/(sigma_all(idx) + sigma_w^2);
            x_all(idx) = x_all(idx) + smootherGain*(x_all(idx+1) - x_all(idx));
            sigma_all(idx) = sigma_all(idx) + smootherGain^2*(sigma_all(idx+1) - sigma_all(idx) - sigma_w^2);
        end
    end
    % We also have derived the expectation on paper. Calculate the optimum
    % lambda with these parameters.
    lambda_denum = 0;
    for j = 1:N_sample
        lambda_denum = lambda_denum + (y(j)-x_all(j))^2 + sigma_all(j);
    end
    lambda = N_sample/lambda_denum;
    lambda_arr_em(i) = lambda;
end

%% VB Algorithm
lambda_arr_vb = zeros(1,N_iter);
lambda = lambda_init; % initial value for lambda.
% initial values for distibutions. 
a_0 = 10;  
b_0 = 10;
a_vb = lambda_init*b_0;  
b_vb = b_0;   % so that E[lambda] starts at lambda_init
mu_x_all_vb = zeros(1,N_sample);
sigma_x_all_vb = zeros(1,N_sample);
for i = 1:N_iter
    % Use the derived relations for optimum values. 
    % q(x) is result of RTS smoother.
    R = b_vb/a_vb; % 1/lambda
    for j = 1:N_sample
        %first, use kalman update.
        if j == 1
            mu_prior = x_init;
            sigma_prior = sigma_w^2;
        else
            mu_prior = mu_x_all_vb(j-1);
            sigma_prior = sigma_x_all_vb(j-1) + sigma_w^2;
        end
        kalmanGain = sigma_prior/(sigma_prior + R);
        mu_post = mu_prior + kalmanGain*(y(j) - mu_prior);
        sigma_post = (1 - kalmanGain)*sigma_prior;
        mu_x_all_vb(j) = mu_post;
        sigma_x_all_vb(j) = sigma_post;
    end

    for j=1:N_sample
        %then, RTS smoother
        idx = N_sample - j;
        if idx > 0
            smootherGain = sigma_x_all_vb(idx)/(sigma_x_all_vb(idx) + sigma_w^2);
            mu_x_all_vb(idx) = mu_x_all_vb(idx) + smootherGain*(mu_x_all_vb(idx+1) - mu_x_all_vb(idx));
            sigma_x_all_vb(idx) = sigma_x_all_vb(idx) + smootherGain^2*(sigma_x_all_vb(idx+1) - sigma_x_all_vb(idx) - sigma_w^2);
        end
    end
    
    a_vb = a_0 + N_sample/2;
    b_vb = b_0;
    for j=1:N_sample
        b_vb = b_vb + 0.5*((y(j)-mu_x_all_vb(j))^2+sigma_x_all_vb(j));
    end
    lambda = a_vb/b_vb;
    lambda_arr_vb(i) = lambda;
end
%% Visualization
figure 
for i = 1:N_iter
    t = 1:1:i;
    plot(t,lambda_arr_em(1:i));
    hold on;
    plot(t,lambda_arr_vb(1:i));
    legend("EM result","VB result");
    ylabel("Lambda");
    xlabel("Iteration");
    hold off;
    pause(0.05);

end 
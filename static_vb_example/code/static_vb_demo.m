lambda_actual = 1; % no way of converging to actual from a single data point.
sigma_x = 1;
mu_x = 0.5;
rng(11);

N_iter = 100;

y = normrnd(mu_x,sigma_x) + normrnd(0,1/sqrt(lambda_actual));

lambda_init = 0.01;


%% EM Algorithm

lambda_arr_em = zeros(1,N_iter);
lambda = lambda_init; % initial value for lambda.
for i = 1:N_iter
    % E step, calculate the kalman update for getting posterior of the latent
    % variable.
    K = sigma_x^2/(sigma_x^2+1/lambda);
    mu_post = mu_x + K*(y-mu_x);
    sigma_post = sqrt(sigma_x^2-K*sigma_x^2);
    % We also have derived the expectation on paper. Calculate the optimum
    % lambda with these parameters.
    lambda = 1/((y-mu_post)^2 + sigma_post^2);
    lambda_arr_em(i) = lambda;
end

%% VB Algorithm
lambda_arr_vb = zeros(1,N_iter);
lambda = lambda_init; % initial value for lambda.
% initial values for distibutions. 
a_0 = 1;  
b_0 = 1;
a_vb = lambda_init*b_0;  
b_vb = b_0;   % so that E[lambda] starts at lambda_init
for i = 1:N_iter
    % Use the derived relations for optimum values. 
    mu_x_vb = (a_vb*y/b_vb + mu_x/(sigma_x^2))/(a_vb/b_vb+1/(sigma_x^2));
    sigma_x_vb = sqrt(b_vb*sigma_x^2/(a_vb*sigma_x^2+b_vb));
    a_vb = a_0 + 0.5;
    b_vb = b_0 + 0.5*((y-mu_x_vb)^2+sigma_x_vb^2);
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
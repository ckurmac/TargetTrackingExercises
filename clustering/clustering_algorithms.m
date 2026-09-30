load("x.mat");

rng(7); % fixed random seed.

N = size(x,2);
D = size(x,1);
minX = min(x(1,:));
maxX = max(x(1,:));
minY = min(x(2,:));
maxY = max(x(2,:));

K = 10; % Component number
maxIter = 400; % All three algorithms run for the same number of iterations, no convergence check.

%% Initialization

% K-Means Clustering
km_means = zeros(2,K);
% Initiate means as random
for i = 1:K
    km_means(1,i) = minX + (maxX-minX)*rand;
    km_means(2,i) = minY + (maxY-minY)*rand;
end
km_rnk = zeros(N,K);

% Expectation Maximization
em_means = zeros(2,K);
em_covariances = cell(1,K);
em_pi_k = ones(1,K)*1/K;
% Initiate means as random, covariance as eye matrix.
for i = 1:K
    em_means(1,i) = minX + (maxX-minX)*rand;
    em_means(2,i) = minY + (maxY-minY)*rand;
    em_covariances{1,i} = eye(2,2);
end
em_responsibilities = zeros(N,K);

% Variational Bayesian
vi_means = zeros(2,K);
vi_covariances = cell(1,K);
% Initiate means as random, covariance as eye matrix.
vi_gw_means = zeros(2,K);
for i = 1:K
    vi_gw_means(1,i) = minX + (maxX-minX)*rand;
    vi_gw_means(2,i) = minY + (maxY-minY)*rand;
    vi_covariances{1,i} = eye(2,2);
end
vi_responsibilities = zeros(N,K); % \rho_nk in this context.

% Conjugate prior parameters
% Dirichlet
vi_alpha_0 = 0.001;
vi_alpha = ones(1,K)*vi_alpha_0;
vi_pi_k = ones(1,K)*1/K;

% Gauss-Wishart
vi_gw_m_0 = zeros(2,1);
vi_gw_weights = cell(1,K);
vi_nu_0 = D;
vi_gw_weight_0 = eye(D,D)/vi_nu_0;
vi_beta_0 = 0.1;
vi_beta_k = ones(1,K)*vi_beta_0;
vi_nu_k = ones(1,K)*vi_nu_0;
for i=1:K
    vi_gw_weights{1,i} = eye(D,D)/vi_nu_0;
end

vi_pruneTol = 0.1*N;
vi_isCompActive = true(1,K);

% Side by side figure
fig = figure;
ax = gobjects(1,3);
for p = 1:3
    ax(p) = subplot(1,3,p);
end

%% Main loop

for iter = 0:maxIter

    % ---------------- K-Means ----------------
    km_cost = 0;
    km_rnk = zeros(N,K); % hard assignments are recomputed from scratch every iteration
    for i = 1:N
        minDist = inf;
        closestMean = 0;
        for j = 1:K
            sqrtDist = (x(:,i) - km_means(:,j))'*(x(:,i) - km_means(:,j));
            if(sqrtDist<minDist)
                closestMean = j;
                minDist = sqrtDist;
            end
        end
        km_rnk(i,closestMean) = 1;
        km_cost = km_cost + minDist;
    end
    % Re-calculate new means
    for i = 1:K
        km_means(:,i) = [sum(x(1,:)*km_rnk(:,i)),sum(x(2,:)*km_rnk(:,i))]/sum(km_rnk(:,i));
    end

    % ---------------- Expectation Maximization ----------------
    em_cost = 0;
    % E step
    for i = 1:N
        for j = 1:K
            em_responsibilities(i,j) = em_pi_k(j) * gaussian_likelihood(x(:,i),em_means(:,j),em_covariances{1,j});
        end
        em_responsibilities(i,:) = em_responsibilities(i,:)./sum(em_responsibilities(i,:));
    end
    % M step
    for i = 1:K
        N_k = sum(em_responsibilities(:,i));
        em_pi_k(i) = N_k/N;
        u = zeros(2,1);
        for j = 1:N
            u = u + em_responsibilities(j,i)*x(:,j);
        end
        em_means(:,i) = u / N_k;

        sig = zeros(2,2);
        for j = 1:N
            d = x(:,j) - em_means(:,i);
            sig = sig + em_responsibilities(j,i)*(d*d');
        end
        em_covariances{1,i} = sig / N_k;
    end
    % Evaluate log likelihood.
    for i = 1:N
        lh = 0;
        for j = 1:K
            lh = lh + em_pi_k(j)*gaussian_likelihood(x(:,i),em_means(:,j),em_covariances{1,j});
        end
        em_cost = em_cost + log(lh);
    end

    % ---------------- Variational Bayesian ----------------
    % E step, calculate E[znk].
    for i = 1:N
        ln_rho_row = -inf(1,K); % inactive components keep -inf -> zero responsibility
        for j = 1:K
            if(~vi_isCompActive(j))
                continue;
            end
            E_u_lambda = D/vi_beta_k(j)+vi_nu_k(j)*(x(:,i)-vi_gw_means(:,j))'*vi_gw_weights{1,j}*(x(:,i)-vi_gw_means(:,j));
            E_ln_lambda = D*log(2)+log(det(vi_gw_weights{1,j}));
            for k = 1:D
                E_ln_lambda = E_ln_lambda + psi((vi_nu_k(j)+1-k)/2);
            end
            E_ln_pi_k = psi(vi_alpha(j))-psi(sum(vi_alpha));
            ln_rho_row(j) = E_ln_pi_k + 0.5*E_ln_lambda - 0.5*D*log(2*pi) - 0.5*E_u_lambda;
        end
        % Log-sum-exp normalization: small alpha_0 makes psi(alpha_0) very negative,
        % so exp(ln_rho) underflows to 0 for every component -> 0/0 = NaN.
        ln_rho_row = ln_rho_row - max(ln_rho_row);
        vi_responsibilities(i,:) = exp(ln_rho_row)./sum(exp(ln_rho_row));
    end
    % M step
    vi_varLowBound = 0;
    for i = 1:K
        if(~vi_isCompActive(i))
            continue;
        end
        N_k = sum(vi_responsibilities(:,i));
        if(N_k<vi_pruneTol && iter>20)
            vi_isCompActive(i) = false;
            vi_responsibilities(:,i) = 0;
            vi_beta_k(i) = vi_beta_0;
            vi_gw_means(:,i) = vi_gw_m_0;
            vi_gw_weights{1,i} = vi_gw_weight_0;
            vi_nu_k(i) = vi_nu_0;
            vi_alpha(i) = vi_alpha_0;
            continue;
        end

        x_bar_k = zeros(2,1);
        for j = 1:N
            x_bar_k = x_bar_k + vi_responsibilities(j,i)*x(:,j);
        end
        % A starved component can reach N_k = 0 before pruning kicks in (iter>20).
        % Guard the 0/0 so x_bar_k = S_k = 0 and the update falls back to the prior.
        x_bar_k = x_bar_k/max(N_k, realmin);

        S_k = zeros(2,2);
        for j = 1:N
            S_k = S_k + vi_responsibilities(j,i)*(x(:,j)-x_bar_k)*(x(:,j)-x_bar_k)';
        end
        S_k = S_k/max(N_k, realmin);

        vi_beta_k(i) = vi_beta_0 + N_k;
        vi_gw_means(:,i) = (vi_beta_0*vi_gw_m_0+N_k*x_bar_k)/vi_beta_k(i);
        vi_gw_weights{1,i} = inv(inv(vi_gw_weight_0) + N_k*S_k + vi_beta_0*N_k*(x_bar_k-vi_gw_m_0)*(x_bar_k-vi_gw_m_0)'/(vi_beta_0+N_k));
        vi_nu_k(i) = vi_nu_0 + N_k;
        vi_alpha(i) = vi_alpha_0 + N_k;

        vi_pi_k(i) = vi_alpha(i)/sum(vi_alpha);
        vi_means(:,i) = vi_gw_means(:,i);
        vi_covariances{1,i} = inv(vi_nu_k(i)*vi_gw_weights{1,i});

        % Calculate variational lower bound increment.
        E_ln_lambda = D*log(2)+log(det(vi_gw_weights{1,i}));
        ln_B = -0.5*vi_nu_k(i)*log(det(vi_gw_weights{1,i})) - 0.5*vi_nu_k(i)*D*log(2) - D*(D-1)/4*log(pi);
        ln_B_0 = -0.5*vi_nu_0*log(det(vi_gw_weight_0)) - 0.5*vi_nu_0*D*log(2) - D*(D-1)/4*log(pi);
        for k = 1:D
            E_ln_lambda = E_ln_lambda + psi((vi_nu_k(i)+1-k)/2);
            ln_B = ln_B - gammaln((vi_nu_k(i)+1-k)/2);
            ln_B_0 = ln_B_0 - gammaln((vi_nu_0+1-k)/2);
        end
        H_lambda = -ln_B - 0.5*(vi_nu_k(i)-D-1)*E_ln_lambda + 0.5*vi_nu_k(i)*D;
        E_ln_p_x = 0.5*N_k*(E_ln_lambda - D/vi_beta_k(i) - vi_nu_k(i)*trace(S_k*vi_gw_weights{1,i}) - vi_nu_k(i)*(x_bar_k-vi_gw_means(:,i))'*vi_gw_weights{1,i}*(x_bar_k-vi_gw_means(:,i)) - D*log(2*pi));
        E_ln_p_mu_lambda = 0.5*(D*log(vi_beta_0/(2*pi)) + E_ln_lambda - D*vi_beta_0/vi_beta_k(i) - vi_beta_0*vi_nu_k(i)*(vi_gw_means(:,i)-vi_gw_m_0)'*vi_gw_weights{1,i}*(vi_gw_means(:,i)-vi_gw_m_0)) + ln_B_0 + 0.5*(vi_nu_0-D-1)*E_ln_lambda - 0.5*vi_nu_k(i)*trace(inv(vi_gw_weight_0)*vi_gw_weights{1,i});
        E_ln_q_mu_lambda = 0.5*E_ln_lambda + 0.5*D*log(vi_beta_k(i)/(2*pi)) - 0.5*D - H_lambda;
        E_ln_q_z = 0;
        for j = 1:N
            if vi_responsibilities(j,i) > 0
                E_ln_q_z = E_ln_q_z + vi_responsibilities(j,i)*log(vi_responsibilities(j,i));
            end
        end
        vi_varLowBound = vi_varLowBound + E_ln_p_x + E_ln_p_mu_lambda - E_ln_q_mu_lambda - E_ln_q_z;
    end
    E_ln_p_z = 0;
    E_ln_p_pi = gammaln(K*vi_alpha_0) - K*gammaln(vi_alpha_0);
    E_ln_q_pi = gammaln(sum(vi_alpha));
    for i = 1:K
        E_ln_pi_k = psi(vi_alpha(i))-psi(sum(vi_alpha));
        E_ln_p_z = E_ln_p_z + sum(vi_responsibilities(:,i))*E_ln_pi_k;
        E_ln_p_pi = E_ln_p_pi + (vi_alpha_0-1)*E_ln_pi_k;
        E_ln_q_pi = E_ln_q_pi + (vi_alpha(i)-1)*E_ln_pi_k - gammaln(vi_alpha(i));
    end
    vi_varLowBound = vi_varLowBound + E_ln_p_z + E_ln_p_pi - E_ln_q_pi;

    % ---------------- Plot side by side ----------------
    set(fig, 'CurrentAxes', ax(1));
    plotKCluster(km_rnk, x, km_means);
    title(sprintf('K-Means Iteration %d', iter));

    set(fig, 'CurrentAxes', ax(2));
    plotEM(em_responsibilities, x, em_means, em_covariances);
    title(sprintf('EM Iteration %d', iter));

    set(fig, 'CurrentAxes', ax(3));
    plotVI(vi_responsibilities, x, vi_means, vi_covariances, vi_pi_k, vi_isCompActive);
    title(sprintf('VI Iteration %d', iter));

end

load("x.mat");

rng(13); % fixed random seed.

N = size(x,2);
D = size(x,1);
minX = min(x(1,:));
maxX = max(x(1,:));
minY = min(x(2,:));
maxY = max(x(2,:));

K = 5; % Component number

%K-Means Clustering

means = zeros(2,K);
% Initiate means as random

for i = 1:K
    means(1,i) = minX + (maxX-minX)*rand;
    means(2,i) = minY + (maxY-minY)*rand;
end

rnk = zeros(N,K);

iter = 0;
maxIter = 100;
cost = 0;
prevCost = 0;
while iter<maxIter
    
    cost = 0;
    for i = 1:N
        minDist = inf;
        closestMean = 0;
        for j = 1:K
            sqrtDist = (x(:,i) - means(:,j))'*(x(:,i) - means(:,j));
            if(sqrtDist<minDist)
                closestMean = j;
                minDist = sqrtDist;
            end
        end
        rnk(i,closestMean) = 1;
        cost = cost + minDist;
    end
    % Re-calculate new means
    for i = 1:K
        means(:,i) = [sum(x(1,:)*rnk(:,i)),sum(x(2,:)*rnk(:,i))]/sum(rnk(:,i));
    end
    plotKCluster(rnk, x, means);
    title(sprintf('K-Means Iteration %d, cost = %.3f', iter, cost));
    pause(0.5);

    if(abs(cost-prevCost)<1e-5)
        break;
    end
    prevCost=cost;

    iter = iter+1;
end

%%

% Expectation Maximization

means = zeros(2,K);
covariances = cell(1,K);
pi_k = ones(1,K)*1/K;

% Initiate means as random, covariance as eye matrix.

for i = 1:K
    means(1,i) = minX + (maxX-minX)*rand;
    means(2,i) = minY + (maxY-minY)*rand;
    covariances{1,i} = eye(2,2);
end



responsibilities = zeros(N,K);

iter = 0;
maxIter = 400;
cost = 0;
prevCost = 0;
while iter<=maxIter
    
    cost = 0;
    % E step
    for i = 1:N
        for j = 1:K
            responsibilities(i,j) = pi_k(j) * gaussian_likelihood(x(:,i),means(:,j),covariances{1,j}); 
        end
        responsibilities(i,:) = responsibilities(i,:)./sum(responsibilities(i,:));
    end
    % M step
    for i = 1:K
        N_k = sum(responsibilities(:,i));
        pi_k(i) = N_k/N;
        u = zeros(2,1);
        for j = 1:N
            u = u + responsibilities(j,i)*x(:,j);
        end
        means(:,i) = u / N_k;

        sig = zeros(2,2);
        for j = 1:N
            d = x(:,j) - means(:,i);
            sig = sig + responsibilities(j,i)*(d*d');
        end
        covariances{1,i} = sig / N_k;
        means(:,i) = u./N_k;
        covariances{1,i} = sig./N_k;
    end
    % Evaluate log likelihood.
    for i = 1:N
        lh = 0;
        for j = 1:K
            lh = lh + pi_k(j)*gaussian_likelihood(x(:,i),means(:,j),covariances{1,j});
        end
        cost = cost + log(lh);
    end
    plotEM(responsibilities, x, means, covariances);
    title(sprintf('EM Iteration %d, cost = %.3f', iter, cost));
    pause(0.1);

    if(abs(cost-prevCost)<1e-5)
        break;
    end
    prevCost=cost;
    iter = iter+1;
end

%%

% Variational Bayesian

means = zeros(2,K);
precisions = cell(1,K);

% Initiate means as random, covariance as eye matrix.

gw_means = zeros(2,K);
for i = 1:K
    gw_means(1,i) = minX + (maxX-minX)*rand;
    gw_means(2,i) = minY + (maxY-minY)*rand;
    precisions{1,i} = eye(2,2);
end

responsibilities = zeros(N,K); % \rho_nk in this context.

iter = 0;
maxIter = 400;
varLowBound= 0;
prevVarLowBound= 0;
% Conjugate prior parameters

% Dirichlet
alpha_0 = 0.001;
alpha = ones(1,K)*alpha_0;
pi_k = ones(1,K)*1/K;

% Gauss-Wishart
gw_m_0 = zeros(2,1);
gw_weights = cell(1,K);
nu_0 = D;
gw_weight_0 = eye(D,D)/nu_0;
beta_0 = 0.1;
beta_k = ones(1,K)*beta_0;

nu_k = ones(1,K)*nu_0;

for i=1:K
    gw_weights{1,i} = eye(D,D)/nu_0;
end

pruneTol = 0.01*N;
isCompActive = true(1,K);
while iter<=maxIter
    % E step, calculate E[znk].
    for i = 1:N
        ln_rho_row = -inf(1,K); % inactive components keep -inf -> zero responsibility
        for j = 1:K
            if(~isCompActive(j))
                continue;
            end
            E_u_lambda = D/beta_k(j)+nu_k(j)*(x(:,i)-gw_means(:,j))'*gw_weights{1,j}*(x(:,i)-gw_means(:,j));
            E_ln_lambda = D*log(2)+log(det(gw_weights{1,j}));
            for k = 1:D 
                E_ln_lambda = E_ln_lambda + psi((nu_k(j)+1-k)/2);
            end
            E_ln_pi_k = psi(alpha(j))-psi(sum(alpha));
            ln_rho = E_ln_pi_k + 0.5*E_ln_lambda - 0.5*D*log(2*pi) - 0.5*E_u_lambda;
            ln_rho_row(j) = ln_rho;
        end
        % Log-sum-exp normalization: small alpha_0 makes psi(alpha_0) very negative,
        % so exp(ln_rho) underflows to 0 for every component -> 0/0 = NaN.
        ln_rho_row = ln_rho_row - max(ln_rho_row);
        responsibilities(i,:) = exp(ln_rho_row)./sum(exp(ln_rho_row));
    end
    % M step
    varLowBound= 0;
    for i = 1:K
        if(~isCompActive(i))
            continue;
        end
        N_k = sum(responsibilities(:,i));
        if(N_k<pruneTol && iter>20)
            isCompActive(i) = false; 
            responsibilities(:,i) = 0;
            beta_k(i) = beta_0;
            gw_means(:,i) = (beta_0*gw_m_0)/beta_0;
            gw_weights{1,i} = gw_weight_0;
            nu_k(i) = nu_0;
            alpha(i) = alpha_0;
            continue;
        end
        
        x_bar_k = zeros(2,1);
        for j = 1:N
            x_bar_k = x_bar_k + responsibilities(j,i)*x(:,j);
        end
        x_bar_k = x_bar_k/N_k;

        S_k = zeros(2,2);
        for j = 1:N
            S_k = S_k + responsibilities(j,i)*(x(:,j)-x_bar_k)*(x(:,j)-x_bar_k)';
        end
        S_k = S_k/N_k;

        beta_k(i) = beta_0 + N_k;
        gw_means(:,i) = (beta_0*gw_m_0+N_k*x_bar_k)/beta_k(i);
        gw_weights{1,i} = inv(inv(gw_weight_0) + N_k*S_k + beta_0*N_k*(x_bar_k-gw_m_0)*(x_bar_k-gw_m_0)'/(beta_0+N_k));
        nu_k(i) = nu_0 + N_k;
        alpha(i) = alpha_0 + N_k;

        pi_k(i) = alpha(i)/sum(alpha);
        means(:,i) = gw_means(:,i);
        covariances{1,i} = inv(nu_k(i)*gw_weights{1,i});
        
        % Calculate variational lower bound increment.
        E_ln_lambda = D*log(2)+log(det(gw_weights{1,i}));
        ln_B = -0.5*nu_k(i)*log(det(gw_weights{1,i})) - 0.5*nu_k(i)*D*log(2) - D*(D-1)/4*log(pi);
        ln_B_0 = -0.5*nu_0*log(det(gw_weight_0)) - 0.5*nu_0*D*log(2) - D*(D-1)/4*log(pi);
        for k = 1:D
            E_ln_lambda = E_ln_lambda + psi((nu_k(i)+1-k)/2);
            ln_B = ln_B - gammaln((nu_k(i)+1-k)/2);
            ln_B_0 = ln_B_0 - gammaln((nu_0+1-k)/2);
        end
        H_lambda = -ln_B - 0.5*(nu_k(i)-D-1)*E_ln_lambda + 0.5*nu_k(i)*D;
        E_ln_p_x = 0.5*N_k*(E_ln_lambda - D/beta_k(i) - nu_k(i)*trace(S_k*gw_weights{1,i}) - nu_k(i)*(x_bar_k-gw_means(:,i))'*gw_weights{1,i}*(x_bar_k-gw_means(:,i)) - D*log(2*pi));
        E_ln_p_mu_lambda = 0.5*(D*log(beta_0/(2*pi)) + E_ln_lambda - D*beta_0/beta_k(i) - beta_0*nu_k(i)*(gw_means(:,i)-gw_m_0)'*gw_weights{1,i}*(gw_means(:,i)-gw_m_0)) + ln_B_0 + 0.5*(nu_0-D-1)*E_ln_lambda - 0.5*nu_k(i)*trace(inv(gw_weight_0)*gw_weights{1,i});
        E_ln_q_mu_lambda = 0.5*E_ln_lambda + 0.5*D*log(beta_k(i)/(2*pi)) - 0.5*D - H_lambda;
        E_ln_q_z = 0;
        for j = 1:N
            if responsibilities(j,i) > 0
                E_ln_q_z = E_ln_q_z + responsibilities(j,i)*log(responsibilities(j,i));
            end
        end
        varLowBound = varLowBound + E_ln_p_x + E_ln_p_mu_lambda - E_ln_q_mu_lambda - E_ln_q_z;
    end
    E_ln_p_z = 0;
    E_ln_p_pi = gammaln(K*alpha_0) - K*gammaln(alpha_0);
    E_ln_q_pi = gammaln(sum(alpha));
    for i = 1:K
        E_ln_pi_k = psi(alpha(i))-psi(sum(alpha));
        E_ln_p_z = E_ln_p_z + sum(responsibilities(:,i))*E_ln_pi_k;
        E_ln_p_pi = E_ln_p_pi + (alpha_0-1)*E_ln_pi_k;
        E_ln_q_pi = E_ln_q_pi + (alpha(i)-1)*E_ln_pi_k - gammaln(alpha(i));
    end
    varLowBound = varLowBound + E_ln_p_z + E_ln_p_pi - E_ln_q_pi;
    
    plotVI(responsibilities, x, means, covariances, pi_k,isCompActive);
    title(sprintf('VI Iteration %d', iter));
    pause(0.1);

    if(abs(varLowBound-prevVarLowBound)<1e-5)
        break;
    end
    prevVarLowBound=varLowBound;
    iter = iter+1;
end

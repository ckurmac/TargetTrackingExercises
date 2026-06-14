function [likelihood] = calcNormalLikelihood(z,mu,sigma)
    k = length(z);
    likelihood = exp(-0.5 * (z-mu)' * (sigma \ (z-mu))) / sqrt(((2*pi)^k) * det(sigma));
end
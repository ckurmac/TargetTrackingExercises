function [lh] = gaussian_likelihood(x,u,Cov)
    lh = exp(-0.5*(x-u)'*(Cov\(x-u))) / sqrt((2*pi)^numel(x)*det(Cov));
end
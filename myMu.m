% baseline function mu0
function [mu0] = myMu(X, mu, sigma, p)
    [d,n] = size(X);
    mu0 = zeros(1,n);
    for i = 1 : length(mu)
        mu0 = mu0 + normpdf(sqrt(d)*(mean(X,1)-0.5)+0.5,mu(i),sigma(i)) * p(i);
    end
end

    
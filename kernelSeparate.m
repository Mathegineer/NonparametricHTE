% estimate mu1, mu2 separately using kernel method and take the difference
function [tauHat] = kernelSeparate (X0, X1, Y0, Y1, X, h1, h2)
    [~, n] = size(X); 
    [~, n0] = size(X0); 
    [~, n1] = size(X1); 
    tauHat = zeros(1,n);
    for i = 1:n
        dist0 = sum((X0 - repmat(X(:,i),1,n0)).^2,1);
        dist1 = sum((X1 - repmat(X(:,i),1,n1)).^2,1); 
        tauHat(i) = sum(Y1 .* normpdf(dist1, 0, h2))/sum(normpdf(dist1, 0, h2)) - ...
            sum(Y0 .* normpdf(dist0, 0, h1))/sum(normpdf(dist0, 0, h1));    
    end 
end
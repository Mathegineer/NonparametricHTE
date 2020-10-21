function tauHat = minimaxEstimator(X0, X1, Y0, Y1, X, m1, m2, kappa)
% preprocess
[~,n] = size(X);
if m2 > 0
    m2 = ceil(m2); 
    m1 = max(ceil(m1),ceil(kappa * m2));
else
    m1 = ceil(m1);
    m2 = m1;
end

% take m1 nearest neighbors
neighborIndex0 = knnsearch(X0',X','K',m1)'; % m1*n matrix
index_vector = reshape(neighborIndex0,n*m1,1); % (n*m1)*1 column vector
[neighborIndex1, Dist] = knnsearch(X1',(X0(:,index_vector))'); %(n*m1)*1 column vector
Dist = reshape(Dist,m1,n); % m1*n matrix
neighborIndex1 = reshape(neighborIndex1,m1,n); % m1*n matrix

% select m2 nearest pairs out of m1 
[~, I] = sort(Dist,'ascend');
I = I + m1 * ones(m1,1) * linspace(0,n-1,n);

tauHat = mean(Y1(neighborIndex1(I(1:m2,:))) ...
    - Y0(neighborIndex0(I(1:m2,:))),1);

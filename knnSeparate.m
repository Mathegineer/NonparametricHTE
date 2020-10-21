% estimate mu1, mu2 separately using knn and take the difference
function tauHat = knnSeparate(X0, X1, Y0, Y1, X, m1, m2)
    m1 = ceil(m1); 
    m2 = ceil(m2);

    neighborIndex0 = knnsearch(X0',X','K',m1)'; % m1*n matrix
    neighborIndex1 = knnsearch(X1',X','K',m2)'; % m2*n matrix
    tauHat =  mean(Y1(neighborIndex1)) - mean(Y0(neighborIndex0));       
end
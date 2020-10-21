% treatment effect tau
function [tau] = myTau(X)
    [d,~] = size(X);
    tau = normpdf(2*sqrt(d)*(mean(X,1)-0.5),0,1); 
end

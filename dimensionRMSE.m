% dimension
% inspect the influence of dimension, i.e. balance between the treatment 
% group size and the control group size on RMSE
function[] = dimensionRMSE(m)
    if(nargin == 0)
        m = 100;
    end
    %% plot of RMSE w.r.t. n, d = 8
    rng(318);
    d = 8;
    betaMu = 0.65; betaTau = 1;
    c1 = 0.5; c2  = 0.5; 
    meanPara =[0.1,0.4,0.8]; sdPara = [0.15,0.05,0.1]; pPara = [0.4,0.5,0.8]*5;
    kappa = 1; 
    nSeq = (1:20)*50; 
    nEval = 100;
    XEval = rand(d, nEval);
    errorSelected = zeros(m, length(nSeq)); errorFull = errorSelected;
    errorSeparateKNN = errorSelected; errorSeparateKernel = errorSelected;

    tic
    for j = 1:length(nSeq)
        n = nSeq(j); sigma = n^(-c1 + betaMu*(1-1/d))/c2; 
        m1 = n * (kappa * sigma^2/n^2)^(d*betaMu/(2*betaMu*betaTau + d*betaMu + d*betaTau));
        m2 = (n^2/kappa)^(2*betaMu*betaTau/(2*betaMu*betaTau + d*betaMu + d*betaTau)) * ...
             sigma^(2*d*(betaMu+betaTau)/(2*betaMu*betaTau + d*betaMu + d*betaTau));
        m1Keep = m1; m2Keep = 0;
        m1Knn = min(n, n * (sigma^2/n)^(d/(2*betaMu+d))); m2Knn = m1Knn;
        h1Kernel = (m1Knn/n)^(1/d)/3; h2Kernel = h1Kernel;
        for i = 1:m
            X0 = rand(d,n); 
            X1 = rand(d,n); 
            Y0 = myMu(X0, meanPara, sdPara, pPara) + randn(1,n) * sigma; 
            Y1 = myMu(X1, meanPara, sdPara, pPara) + ...
                sqrt(d) * myTau(X1) + randn(1,n) * sigma; 
            % minimax
            tau = myTau(XEval);
            tauHat = minimaxEstimator(X0, X1, Y0, Y1, XEval, m1, m2, kappa);
            errorSelected(i,j) = sqrt(mean((tauHat - tau).^2));
            % minimax keep all the data
            tauHatMinimaxKeep = minimaxEstimator(X0, X1, Y0, Y1, XEval, m1Keep, m2Keep, kappa);
            errorFull(i,j) = sqrt(mean((tauHatMinimaxKeep - tau).^2));
            % separate KNN
            tauHatSeparateKNN = knnSeparate(X0, X1, Y0, Y1, XEval, m1Knn, m2Knn);
            errorSeparateKNN(i,j)  = sqrt(mean((tauHatSeparateKNN - tau).^2));
            % separate kernel
            tauHatSeparateKernel = kernelSeparate(X0, X1, Y0, Y1, XEval, h1Kernel, h2Kernel);
            errorSeparateKernel(i,j) = sqrt(mean((tauHatSeparateKernel - tau).^2));
        end
    end
    errorMinimaxAve = nanmean(errorSelected, 1);
    errorMinimaxKeepAve = nanmean(errorFull, 1);
    errorSeparateKNNAve = nanmean(errorSeparateKNN, 1);
    errorSeparateKernelAve = nanmean(errorSeparateKernel, 1);

    figure;
    plot(nSeq, errorMinimaxAve, 'rs-', 'LineWidth',2, 'MarkerFaceColor', 'r'); hold on; 
    plot(nSeq, errorMinimaxKeepAve, 'bo--', 'LineWidth',2, 'MarkerFaceColor', 'b')
    plot(nSeq, errorSeparateKNNAve, 'v-.', 'Color', [0,0.5,0], 'LineWidth',2,'MarkerFaceColor', [0,0.5,0]);
    plot(nSeq, errorSeparateKernelAve, '^:', 'Color', [0.8,0,0.4], 'LineWidth',2,'MarkerFaceColor',[0.8,0,0.4]);
    xlabel('n'); ylabel('RMSE');
    legend('selected matching','full matching', 'kNN differencing', 'kernel differencing'); 
    title(strcat(string('d='), string(d)));

    %% RMSE w.r.t. d
    rng(318);
    dSeq = 1:10;
    betaMu = 0.8; betaTau = 1;
    c1 = 0.5; c2 = 0.5; 
    meanPara =[0.1,0.4,0.8]; sdPara = [0.15,0.05,0.1]; pPara = [0.4,0.5,0.8]*5;
    kappa = 1; 
    n = 1000;
    nEval = 100;
    errorSelected = zeros(m, length(dSeq)); errorFull = errorSelected;
    errorSeparateKNN = errorSelected; errorSeparateKernel = errorSelected;

    for j = 1:length(dSeq)
        d = dSeq(j); sigma = n^(-c1+betaMu*(1-1/d))/c2; 
        XEval = reshape(rand(1, d*nEval), [d, nEval]);
        m1 = n * (kappa * sigma^2/n^2)^(d*betaMu/(2*betaMu*betaTau + d*betaMu + d*betaTau));
        m2 = (n^2/kappa)^(2*betaMu*betaTau/(2*betaMu*betaTau + d*betaMu + d*betaTau)) * ...
            sigma^(2*d*(betaMu+betaTau)/(2*betaMu*betaTau + d*betaMu + d*betaTau));
        m1Keep = m1; m2Keep = 0;
        m1Knn = min(n, n * (sigma^2/n)^(d/(2*betaMu+d))); m2Knn = m1Knn;
        h1Kernel = (m1Knn/n)^(1/d)/3; h2Kernel = h1Kernel;
        for i = 1:m
            X0 = reshape(rand(1,d*n), [d,n]); 
            X1 = reshape(rand(1,d*n), [d,n]); 
            Y0 = myMu(X0, meanPara, sdPara, pPara) + randn(1,n) * sigma; 
            Y1 = myMu(X1, meanPara, sdPara, pPara) + ...
                sqrt(d) * myTau(X1) + randn(1,n) * sigma; 
            % minimax
            tau = myTau(XEval);
            tauHat = minimaxEstimator(X0, X1, Y0, Y1, XEval, m1, m2, kappa);
            errorSelected(i,j) = sqrt(mean((tauHat - tau).^2));
            % minimax keep all the data
            tauHatMinimaxKeep = minimaxEstimator(X0, X1, Y0, Y1, XEval, m1Keep, m2Keep, kappa);
            errorFull(i,j) = sqrt(mean((tauHatMinimaxKeep - tau).^2));
            % separate KNN
            tauHatSeparateKNN = knnSeparate(X0, X1, Y0, Y1, XEval, m1Knn, m2Knn);
            errorSeparateKNN(i,j)  = sqrt(mean((tauHatSeparateKNN - tau).^2));
            % separate kernel
            tauHatSeparateKernel = kernelSeparate(X0, X1, Y0, Y1, XEval, h1Kernel, h2Kernel);
            errorSeparateKernel(i,j) = sqrt(mean((tauHatSeparateKernel - tau).^2));
        end
    end
    toc
    errorMinimaxAve = nanmean(errorSelected, 1);
    errorMinimaxKeepAve = nanmean(errorFull, 1);
    errorSeparateKNNAve = nanmean(errorSeparateKNN, 1);
    errorSeparateKernelAve = nanmean(errorSeparateKernel, 1);

    figure;
    plot(dSeq, errorMinimaxAve, 'rs-', 'LineWidth',2, 'MarkerFaceColor', 'r'); hold on; 
    plot(dSeq, errorMinimaxKeepAve, 'bo--', 'LineWidth',2, 'MarkerFaceColor', 'b')
    plot(dSeq, errorSeparateKNNAve, 'v-.', 'Color', [0,0.5,0], 'LineWidth',2,'MarkerFaceColor', [0,0.5,0]);
    plot(dSeq, errorSeparateKernelAve, '^:', 'Color', [0.8,0,0.4], 'LineWidth',2,'MarkerFaceColor',[0.8,0,0.4]);
    xlabel('dimension'); ylabel('RMSE');
    legend('selected matching','full matching', 'kNN differencing', 'kernel differencing'); 


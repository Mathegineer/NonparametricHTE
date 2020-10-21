% default setting
function [] = defaultRMSE(m)
    if(nargin == 0)
        m = 100;
    end
    %% plot of RMSE w.r.t. n, default
    rng(318); 
    betaMu = 0.65; betaTau = 1;
    c1 = 0.5; c2 = 0.5; 
    meanPara =[0.1,0.4,0.8]; sdPara = [0.15,0.05,0.1]; pPara = [0.4,0.5,0.8]*5;
    kappa = 1;
    nSeq = (1:20)*50;
    nEval = 100;
    XEval = (0:nEval)/nEval;
    errorSelected = zeros(m, length(nSeq)); errorFull = errorSelected;
    errorSeparateKNN = errorSelected; errorSeparateKernel = errorSelected;

    tic
    for j = 1:length(nSeq)
        n = nSeq(j); sigma = n^(-c1)/c2; 
        m1 = n * (kappa * sigma^2/n^2)^(betaMu/(2*betaMu*betaTau + betaMu + betaTau));
        m2 = (n^2/kappa)^(2*betaMu*betaTau/(2*betaMu*betaTau + betaMu + betaTau)) * ...
             sigma^(2*(betaMu+betaTau)/(2*betaMu*betaTau + betaMu + betaTau));
        m1Keep = m1; m2Keep = 0;
        m1Knn = min(n, n * (sigma^2/n)^(1/(2*betaMu+1))); m2Knn = m1Knn;
        h1Kernel = m1Knn/n; h2Kernel = h1Kernel;

        for i = 1:m
            X0 = sort(rand(1,n)); % control
            X1 = sort(rand(1,n)); % treatment
            Y0 = myMu(X0, meanPara, sdPara, pPara) + randn(1,n) * sigma; 
            Y1 = myMu(X1, meanPara, sdPara, pPara) + myTau(X1) + randn(1,n) * sigma; 
            % selected matching
            tau = myTau(XEval);
            tauHat = minimaxEstimator(X0, X1, Y0, Y1, XEval, m1, m2, kappa);
            errorSelected(i,j) = sqrt(mean((tauHat - tau).^2));
            % full matching
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
    toc

    figure;
    plot(nSeq, errorMinimaxAve, 'rs-', 'LineWidth',2, 'MarkerFaceColor', 'r'); hold on; 
    plot(nSeq, errorMinimaxKeepAve, 'bo--', 'LineWidth',2, 'MarkerFaceColor', 'b')
    plot(nSeq, errorSeparateKNNAve, 'v-.', 'Color', [0,0.5,0], 'LineWidth',2,'MarkerFaceColor', [0,0.5,0]);
    plot(nSeq, errorSeparateKernelAve, '^:', 'Color', [0.8,0,0.4], 'LineWidth',2,'MarkerFaceColor',[0.8,0,0.4]);
    xlabel('n'); ylabel('RMSE');
    legend('selected matching','full matching', 'kNN differencing', 'kernel differencing'); 

    %% estimate plot, default
    rng(320);
    c1 = 0.5; c2 = 0.5; betaMu = 0.65; betaTau = 1;
    n = 1000; sigma = n^(-c1)/c2;
    meanPara =[0.1,0.4,0.8]; sdPara = [0.15,0.05,0.1]; pPara = [0.4,0.5,0.8]*5;
    % minimax; moderate noise
    m1 = n * (kappa * sigma^2/n^2)^(betaMu/(2*betaMu*betaTau + betaMu + betaTau));
    m2 = (n^2/kappa)^(2*betaMu*betaTau/(2*betaMu*betaTau + betaMu + betaTau)) * ...
    sigma^(2*(betaMu+betaTau)/(2*betaMu*betaTau + betaMu + betaTau));
    m1Keep = m1; m2Keep = 0;
    m1Knn = min(n, n * (sigma^2/n)^(1/(2*betaMu+1))); m2Knn = m1Knn;
    h1Kernel = m1Knn/n; h2Kernel = h1Kernel;

    XEval = (0:100)/100;
    X0 = sort(rand(1,n)); % control
    X1 = sort(rand(1,n)); % treatment
    Y0 = myMu(X0, meanPara, sdPara, pPara) + randn(1,n) * sigma; 
    Y1 = myMu(X1, meanPara, sdPara, pPara) + myTau(X1) + randn(1,n) * sigma; 
    % minimax
    tau = myTau(XEval);
    tauHatMinimax = minimaxEstimator(X0, X1, Y0, Y1, XEval, m1, m2, kappa);
    % minimax keep all the data
    tauHatMinimaxKeep = minimaxEstimator(X0, X1, Y0, Y1, XEval, m1Keep, m2Keep, kappa);
    % separate KNN
    tauHatSeparateKNN = knnSeparate(X0, X1, Y0, Y1, XEval, m1Knn, m2Knn);
    % separate kernel
    tauHatSeparateKernel = kernelSeparate(X0, X1, Y0, Y1, XEval, h1Kernel, h2Kernel);

    figure; 
    plot(XEval, tau, 'k', 'LineWidth',2); hold on;
    plot(XEval, tauHatMinimax, 'r', 'LineWidth',2);
    plot(XEval, tauHatMinimaxKeep, 'b--', 'LineWidth',1);
    plot(XEval, tauHatSeparateKNN, '-.', 'Color', [0,0.5,0], 'LineWidth',1);
    plot(XEval, tauHatSeparateKernel, ':','Color', [0.8,0,0.4], 'LineWidth',2);
    legend('true HTE','selected matching','full matching', 'kNN differencing', 'kernel differencing'); 

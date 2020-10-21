clear; clc; close all;
%% default setting:
%     plot of RMSE w.r.t. sample size, default setting; 
%     plot of estimate, default setting. 
defaultRMSE(); 

%% dependence on kappa: 
%     plot of RMSE w.r.t. sample size, kappa = 4; 
%     plot of estimate, kappa = 4. 
%     plot or RMSE w.r.t. kappa; 
kappaRMSE(); 

%% dependence on dimension:
%     plot of RMSE w.r.t. sample size, dimension = 8; 
%     plot of RMSE w.r.t. dimension. 
dimensionRMSE(); 
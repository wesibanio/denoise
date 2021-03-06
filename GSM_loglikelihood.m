function [ll] = GSM_loglikelihood(X, model)
% Calculate the log likelihood of X, given a GSM model.
% 
% The model assumes that y = x + noise where x is generated by a mixture of
% 0-mean gaussian components sharing the same covariance up to a scaling
% factor.
%
% Argument
%  X - A DxM matrix, whose every column corresponds to a patch in D
%      dimensions (typically D=64).
%  model - a struct with 3 fields:
%           mix - Mixture proportions.
%           covs - A DxDxK array, whose every page is a scaled covariance
%                  matrix according to scaling parameters.
%           means - K 0-means.
%
ll = GMM_loglikelihood(X, model);
end
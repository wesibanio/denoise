function [model] = learn_MVN(X, options)
% Learn parameters for a 0-mean multivariate normal model for X.
%
% Arguments:
%   X - Data, a DxM data matrix, where D is the dimension, and M is the
%       number of samples.
%   options - options for learn_GMM (optional).
% Returns:
%   model - a struct with 3 fields:
%            cov - DxD covariance matrix.
%            mean - 0
[D, M] = size(X);
model = struct();
model.means = 0;
temp = zeros(D, D);
for i=1:M
    temp = temp + (X(:,i) * transpose(X(:,i)));
model.cov = 1/M * temp; % calculating covariance
end

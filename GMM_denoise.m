function [xhat] = GMM_denoise(y, gmm, noise)
% Denoises every column in y, assuming a gaussian mixture model and white
% noise.
% 
% The model assumes that y = x + noise where x is generated from a GMM.
%
% Arguments
%  y - A DxM matrix, whose every column corresponds to a patch in D
%      dimensions (typically D=64).
%  gmm - The mixture model, with 4 fields:
%          means - A KxD matrix where K is the number of components in
%                  mixture and D is the dimension of the data.
%          covs - A DxDxK array whose every page is a covariance matrix of
%                 the corresponding component.
%          mix - A Kx1 vector with mixing proportions.
%  noise - the std of the noise in y.
%

% =========================================================================
% This is an optional file - use if if you want to implement all denoising
% code in one place...
% =========================================================================
[D, M] = size(y);
xhat = zeros(D, M);
for sample=1:M
    vec_prob_sample = postrioryProb(noise, gmm, y(:,sample));
    xhat_temp = zeros(D, 1);
    for guassian=1:K
        xhat_temp = xhat_temp + vec_prob_sample(guassian) * inv(inv(mvn.cov(:, :, guassian) + (1/(noise*noise))*(eye(D))) * (1/(noise*noise)* y(:,sample));
    end
    xhat(:,sample) = xhat_temp;
end
end

function [prob] = postrioryProb(noise, gmm, y)
% the function calculate p(k|y) for every k
[K, ~] = size(gmm.mix);
[D, ~] = size(gmm.covs);
yGivenK = zeros(K);
for i = 1:K
    yGivenK(i) = mvnpdf(y, gmm.means(i,:), gmm.covs(:,:,i) + eye(D)*noise);
end
yGivenKSum = sum(yGivenK);
prob = (yGivenK.*gmm.mix) / yGivenKSum ;
end
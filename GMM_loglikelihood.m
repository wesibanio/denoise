function [ll] = GMM_loglikelihood(X, theta)
% Calculate the log likelihood of X, given a mixture model.
% 
% The model assumes each column of x is independently generated by a
% mixture model with parameters theta.
%
% Arguments
%  X - A DxM matrix, whose every column corresponds to a patch in D
%      dimensions (typically D=64).
%  theta - A struct with fields:
%          means - A KxD matrix where K is the number of components in
%                  mixture and D is the dimension of the data.
%          covs - A DxDxK array whose every page is a covariance matrix of
%                 the corresponding component.
%          mix - A Kx1 vector with mixing proportions.
%

% =========================================================================
% This is an optional file
% =========================================================================

[~, M] = size(X);
[K, ~] = size(theta.mix);
ll = 0;

pdf_mat = nan(M,K);
for k=1:K
    pdf_mat(:, k) = log_mvnpdf(X', theta.means(k,:), theta.covs(:,:,k));
end
pdf_mat = bsxfun(@plus, pdf_mat, log(theta.mix)');
ll = sum(logsum(pdf_mat, 2));

%{
for i=1:M
    sum_xi = 0;
    for j=1:K
        sum_xi = sum_xi + mvnpdf(transpose(X(:,i)), theta.means(j,:), theta.covs(:,:,j)) * theta.mix(j);
    end

    sum_xi = log(sum_xi);
    ll = ll + sum_xi;
end
%}
end
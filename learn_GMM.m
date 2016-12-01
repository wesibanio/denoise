function [theta, LL] = learn_GMM(X, K, params0, options)
% Learn parameters for a gaussian mixture model via EM.
%
% Arguments:
%   X - Data, a DxM data matrix, where D is the dimension, and M is the
%       number of samples.
%   K - Number of components in mixture.
%   params0 - An optional struct with intialization parameters. Has 3
%             optional fields:
%               means - a KxD matrix whose every row corresponds to a
%                       component mean.
%               covs - A DxDxK array, whose every page is a component
%                      covariance matrix.
%               mix - A Kx1 mixture vector (should sum to 1).
%             If not given, a random starting point is generated.
%   options - Algorithm options struct, with fields:
%              learn - A struct of booleans, denoting which parameters
%                      should be learned: learn.means, learn.covs and
%                      learn.mix. The default is that given parameters
%                      (in params0) are not learned.
%              max_iter - maximum #iterations. Default = 100.
%              thresh - if previous_LL * thresh > current_LL,
%                       algorithm halts. default = 1.01.
%              verbosity - either 'none', 'iter' or 'plot'. default 'none'.
% Returns:
%   params - A struct with learned parameters (fields):
%               means - a KxD matrix whose every row corresponds to a
%                       component mean.
%               covs - A DxDxK array, whose every page is a component
%                      covariance matrix.
%               mix - A Kx1 mixture vector.
%   LL - log likelihood history
%
% =========================================================================
% This is an optional file - use it if you want to implement a single EM
% algorithm
% =========================================================================
%
% EPS = 1e-10;
if ~exist('params0', 'var') 
    params0 = struct(); 
end
[theta, default_learn] = get_params0(X, K, params0);

if ~exist('options', 'var') 
    options = struct(); 
end
options = organize_options(options, default_learn);

%likelihood array
LL = nan(1, options.max_iter);
[D, M] = size(X);
counter = 1; % how many iter

%calculate ll and condition. notice : max(A) (with trans) will give indexs
%and values.
ll_curr = -inf;

while (counter <= options.max_iter)
    %expectation
    weights = calculat_weights(X, theta.mix, theta.covs, theta.means);
    
    %maximization
    if options.learn.mix
        theta.mix = (sum(weights) / M)';
    end
    
    if options.learn.means
        theta.means = calculate_mean(X, weights);
    end
    
    if options.learn.covs
		theta.covs = calculate_cov(X, weights, theta.means);
    end
    
    if options.learn.base_GSM_cov
		theta.cov = calculate_GSM_cov(X, weights, theta.base_GSM_cov);
    end
    
    
    %calculating ll
    
    ll_prev = ll_curr;
    ll_curr = GMM_loglikelihood(X, theta);
    LL(counter) = ll_curr;
    %checking if previous_LL * thresh > current_LL
    if(ll_prev * options.threshold > ll_curr)
        LL = LL(1, 1:counter);
        break
    end
    
    counter = counter + 1;    
end
end

function [params0, default_learn] = get_params0(X, K, params0)
% organizes the params0 struct and output the starting point of the
% algorithm - "params0".
default_learn.mix = false;
default_learn.means = false;
default_learn.covs = false;
default_learn.base_GSM_cov = false;
[D,M] = size(X);

if ~isfield(params0, 'means')
    default_learn.means = false;
    % params0.means = reshape((sum(reshape(X(:,randi(M, [1,K*K])),[D*K,K]), 2) / K), [D,K])';
    % params0.means = params0.means + nanstd(X(:))*randn(size(params0.means));
    params0.means = zeros(K,D);
end

if ~isfield(params0, 'covs')
    default_learn.covs = true;
    params0.covs = nan(D,D,K);
    for k = 1:K
        params0.covs(:,:,k) = nancov(X(:,randi(M, [1,100]))');
    end
end

if ~isfield(params0, 'base_GSM_cov')
    params0.base_GSM_cov = nancov(X');
end

if ~isfield(params0, 'mix')
    default_learn.mix = true;
    params0.mix = rand(K,1);
    params0.mix = params0.mix / sum(params0.mix);
end

end


function [weights] = calculat_weights(X, prob_k, cov, mu)
[~, N] = size(X);
[K, ~] = size(prob_k);
weights = zeros(N, K); %M=number of samples, K=number of guassian
 
for k = 1:K
	weights(:,k) = log_mvnpdf(X', mu(k,:), cov(:,:,k));
end
weights = bsxfun(@plus,weights,log(prob_k)');
temp_sum = logsum(weights, 2);
weights = exp(bsxfun(@minus,weights,temp_sum));
end

function [cov] = calculate_cov(X, weights, means)
[N, K] = size(weights);
[D, ~] = size(X);

cov = zeros(D, D, K);

for k=1:K
	% curr_cov = zeros(D, D);
    cov(:,:,k) = X * (X' .* repmat(weights(:,k), [1,D]));
	
	% for i=1:N
    %    curr_cov = curr_cov + X(:,i) * X(:,i)' * weights(i,k);
    % end
	cov(:,:,k) = curr_cov / sum(weights(:,k));
end
end

function [cov] = calculate_GSM_cov(X, weights, base_cov)
[~, K] = size(weights);
[D, ~] = size(X);

mix_cov_scalar = nan(K);
for k = 1:K
    mix_cov_scalar(k) = exp(logsum(weights(:, k) + log(diag(X' * base_cov * X))) + log(D) - logsum(weights(:, k)));
end;
cov = reshape(bsxfun(@times, base_cov(:), mix_cov_scalar), [size(base_cov) K]);
end

function [means] = calculate_mean(X, weights)
% The function calculate the mean of k guassians
[~, K] = size(weights);
[D, ~] = size(X);


% means = X * exp(weights);
means = nan([K, D]);
X = log(X);
for k = 1:K
	means(k, :) = exp(logsum(X + repmat(weights(:, k)',[D, 1]), 2) - logsum(weights(:, k)));
end
end

function [options] = organize_options(options, default_learn)
%organize the options.
if ~isfield(options, 'threshold') options.threshold = 1.01; end
if ~isfield(options, 'max_iter') options.max_iter = 100; end
if ~isfield(options, 'verbosity') options.verbosity = 'none'; end
if ~isfield(options, 'learn') options.learn = default_learn;
else
    if ~isfield(options.learn, 'means') options.learn.means = default_learn.means; end;
    if ~isfield(options.learn, 'mix') options.learn.mix = default_learn.mix; end;
    if ~isfield(options.learn, 'covs')
        if ~isfield(options.learn, 'base_GSM_cov')
            options.learn.covs = default_learn.covs; 
        else
            options.learn.base_GSM_cov = ~default_learn.base_GSM_cov;
        end;
    end;
end
end
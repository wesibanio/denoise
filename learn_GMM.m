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
EPS = 1e-10;
if ~exist('params0', 'var') 
    params0 = struct(); 
end
[theta, default_learn] = get_params0(X, K, params0);

if ~exist('options', 'var') 
    options = struct(); 
end
options = organize_options(options, default_learn);

%likelihood array
LL = zeros(0);

counter = options.max_iter; % hoe many iter

%calculate ll and condition. notice : max(A) (with trans) will give indexs
%and values.
ll_curr = 0;
ll_prev = 0;
counter_ll = 1;
while (~counter == 0)
    %maximition
    if d
    weights = calculat_weights(X, theta.mix, theta.covs, theta.means);
    
    %exception
    if learn.means
        theta.mean = calculate_mean(X, weights);
    end
    theta.cov = calculate_cov(X, weights, theta.mean);
    
    %calculationg ll 
    ll_prev = ll_curr;
    ll_curr = GMM_loglikelihood(X, theta);
    LL = [LL ll_curr];
    %checking if previous_LL * thresh > current_LL
    if(ll_prev * options.threshold > ll_curr)
        break
    end
    counter = counter - 1;
    counter_ll = counter_ll + 1;
    
end
end

function [params0, default_learn] = get_params0(X, K, params0)
% organizes the params0 struct and output the starting point of the
% algorithm - "params0".
default_learn.mix = false;
default_learn.means = false;
default_learn.covs = false;

[D,M] = size(X);

if ~isfield(params0, 'means')
    default_learn.means = true;
    params0.means = X(:,randi(M, [1,K]))';
    params0.means = params0.means + nanstd(X(:))*randn(size(params0.means));
end

if ~isfield(params0, 'covs')
    default_learn.covs = true;
    params0.covs = nan(D,D,K);
    for k = 1:K
        params0.covs(:,:,k) = nancov(X(:,randi(M, [1,10]))');
    end
end

if ~isfield(params0, 'mix')
    default_learn.mix = true;
    params0.mix = rand(K,1);
    params0.mix = params0.mix / sum(params0.mix);
end

end

function [weights] = calculat_weights(X, prob_k, cov, mu)
[D,M] = size(X);
[K,L] = size(prob_k);
weights = nan(M, K); %M=number of samples, K=number of guassian
for i=1:M % which sample
    sum = 0;
    for k=1:K % sigma (k=1...K) (pk(xi |zk,tethak) * prob_is_z)
        sum = sum + (mvnpdf(transpose(X(:,i)), mu(k,:), cov(:,:,k)) * prob_k(k));
    end
    for j=1:K % which guassian
        temp = mvnpdf(transpose(X(:,i)), mu(j,:), cov(:,:,j)) * prob_k(j);
        weights(i, j) = temp / sum; 
    end
end
end
 
function [prob_k] = calculate_prob_k(N, W)
[N, K] = size(W);
row_sum_W = sum(W, 1); %sigma(i=1..N) wik
prob_k = nan(K);
for i=1:K
    prob_k(i) = row_sum_W(i) / N;
end
end

function [cov] = calculate_cov(X, weights, means)
[N, K] = size(weights);
[D,M] = size(X);
cov = zeros(D, D, K);
row_sum_W = sum(weights, 1); %sigma(i=1..N) wik = Nk
for j=1:K
    temp_sum = 0;
    for i=1:M
        temp_sum = temp_sum + (X(:,i) - means(j)) * transpose((X(:,i) - means(j))) * weights(i,j);
    end
    cov(:,:,j) = 1/row_sum_W(j) * temp_sum;
end
end

function [means] = calculate_mean(X, weights)
% The function calculate the covariance of k guassians
[N, K] = size(weights);
[D, M] = size(X);
means = nan(K, D);
row_sum_W = sum(weights, 1); %sigma(i=1..N) wik = Nk
for j=1:K
    temp_sum = zeros(1,D);
    for i=1:M
        temp_sum = temp_sum + (transpose(X(:,i)) * weights(i,j));
    end
    means(j,:) = (1/row_sum_W(j)) * temp_sum;
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
    if ~isfield(options.learn, 'covs') options.learn.covs = default_learn.covs; end;
    if ~isfield(options.learn, 'mix') options.learn.mix = default_learn.mix; end;
end
end



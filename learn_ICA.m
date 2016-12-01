function model = learn_ICA(X, K, options)
% Learn parameters for a complete invertible ICA model.
%
% We learn a matrix P such that X = P*S, where S are D independent sources
% And for each of the D coordinates we learn a mixture of K (univariate)
% 0-mean gaussians via EM.
%
% Arguments:
%   X - Data, a DxM data matrix, where D is the dimension, and M is the
%       number of samples.
%   K - Number of components in a mixture.
%   options - options for learn_GMM (optional).
% Returns:
%   model - A struct with 3 fields:
%           P - mixing matrix of sources (P: D ind. sources -> D signals)
%           vars - a DxK matrix whose (d,k) element correponsds to the
%                  variance of the k'th component in dimension d.
%           mix - a DxK matrix whose (d,k) element correponsds to the
%                 mixing weight of the k'th component in dimension d.
%           notice: we assume that the mean of all the guassian is zero.
%
% notice : taking PT
model = struct();
covariance_data = cov(X');
[V, ~] = eig(covariance_data);
model.P = orth(V); % P !

S = model.P' * X; % our s !

[D, M] = size(S);

model.covs = zeros(D, 1, 1, K);
model.mix = zeros(D, K);
model.means = zeros(D, K, 1);
params0 = struct();
params0.means = zeros(K,1);

for i=1:D % for every si in xi
    [modelNew, ll] = learn_GMM(S(i,:), K, params0);
    model.covs(i,:,:,:) = modelNew.covs;
    model.mix(i,:) = modelNew.mix;
	model.means(i,:,:) = modelNew.means;
end
%{
char_mix_of_guassian = struct();
model.var = zeros(D, K);
model.mix = zeros(D, K);
model.means = zeros(D, K);
for i=1:D
    model.var(i,:) = char_mix_of_guassian(i,1).covs;
    model.mix(i,:) = char_mix_of_guassian(i,1).mix;
end
%}

end


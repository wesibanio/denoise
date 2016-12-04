%{
MU1 = [1 2];
SIGMA1 = [2 0; 0 .5];
MU2 = [-3 -5];
SIGMA2 = [1 0; 0 1];
X = transpose([mvnrnd(MU1,SIGMA1,10);mvnrnd(MU2,SIGMA2,10)]);
%[t, ll] = learn_GMM(X, 2);
learn_MVN(X);
%}
function main()
X = load('ims.mat');
train_set = sample_patches(standardize_ims(X.ims.train));
mvn_model = learn_GSM(train_set, 2);
mvn_model.name = 'GSM';
mvn_model.loglikelihood = @(x)GSM_loglikelihood(x,mvn_model);
mvn_model.denoise = @(y, noise)GSM_denoise(y,mvn_model,noise);
[psnr, ll] = test_denoising(X.ims.test,{mvn_model});

end
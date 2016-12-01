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
X = load('../ex1YardenFix/ims.mat');
train_set = sample_patches(X.ims.train);

%{
mvn_model = learn_ICA(train_set, 2);
mvn_model.name = 'ICA';
mvn_model.loglikelihood = @(x)ICA_loglikelihood(x,mvn_model);
mvn_model.denoise = @(y, noise)ICA_denoise(y,mvn_model,noise);
%}

mvn_model = learn_GMM(train_set, 2);
mvn_model.name = 'GMM';
mvn_model.loglikelihood = @(x)GMM_loglikelihood(x,mvn_model);
mvn_model.denoise = @(y, noise)GMM_denoise(y,mvn_model,noise);


[psnr, ll, dur] = test_denoising(X.ims.test,{mvn_model});

end
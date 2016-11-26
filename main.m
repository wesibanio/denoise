MU1 = [1 2];
SIGMA1 = [2 0; 0 .5];
MU2 = [-3 -5];
SIGMA2 = [1 0; 0 1];
X = transpose([mvnrnd(MU1,SIGMA1,10);mvnrnd(MU2,SIGMA2,10)]);
%[t, ll] = learn_GMM(X, 2);
learn_MVN(X);
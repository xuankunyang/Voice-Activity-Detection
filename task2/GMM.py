import numpy as np
from scipy.stats import multivariate_normal

class GaussianMixtureModel():
    def __init__(self, n_compoents: int, max_iter: int = 500, tol: float = 1e-4, init_method: str = 'kmeans', reg_covar: float = 1e-6):
        self.n_compoents = n_compoents
        self.max_iter = max_iter
        self.tol = tol
        self.init_method = init_method
        self.reg_covar = reg_covar

    def fit(self, X: np.ndarray):
        n_samples, n_features = X.shape
        self.means = np.zeros((self.n_compoents, n_features))
        self.covar = np.array([np.identity(n_features)] * self.n_compoents)
        self.weights = np.ones((self.n_compoents, )) / self.n_compoents

        if self.init_method == 'kmeans':
            self._init_kmeans(X)
        if self.init_method == 'random':
            self._init_random(X)

        log_likelihoods = []
        for iter in range(self.max_iter):
            # E step, 计算后验概率
            gamma = self._expectation_step(X)

            # M step, 更新参数
            self._maximization_step(X, gamma)

            log_likelihoods.append(self._log_likelihood(X))

            if iter > 0 and np.abs(log_likelihoods[-1] - log_likelihoods[-2])<self.tol:
                print(f"Converged after {iter} iterations.")
                break
        else:
            print(f"Reached maximum iterations {self.max_iter}.")
        
    def _expectation_step(self, X):
        n_samples, n_features = X.shape   
        gamma = np.zeros((n_samples, self.n_compoents))
        
        for k in range(self.n_compoents):
            pdf = multivariate_normal.pdf(X, mean=self.means[k], cov=self.covar[k])
            gamma[:, k] = self.weights[k] * pdf
        
        gamma = gamma / np.sum(gamma, axis=1, keepdims=True)
        return gamma
    
    def _maximization_step(self, X, gamma):
        n_samples, n_features = X.shape

        Gamma0 = np.sum(gamma, axis=0) # (n_compoents, )
        Gamma1 = gamma.T @ X # (n_compoents, n_features)

        X_expand = X[:, :, np.newaxis] # (n_samples, n_features, 1)
        X_T_expand = X[:, np.newaxis, :] # (n_samples, 1, n_features)
        gamma_expand = gamma[:, :, np.newaxis, np.newaxis] # (n_smaples, n_compoents, 1, 1)

        # NOTE Here needs more adjusting!!!

        X_X_T_expand = X_expand @ X_T_expand # (n_samples, n_features, n_features)
        X_X_T_expand = X_X_T_expand[:, np.newaxis, :, :] # (n_samples, 1, n_features, n_features)
        Gamma2 = np.sum(gamma_expand * X_X_T_expand, axis=0) # (n_compoents, n_features, n_features)

        Gamma0_expand = Gamma0[:, np.newaxis] # (n_compoents, 1)
        self.means = Gamma1 / Gamma0_expand
        mean_expand = self.means[:, :, np.newaxis] # (n_compoents, n_features, 1)
        mean_T_expand = self.means[:, np.newaxis, :] # (n_compoents, 1, n_features)
        Gamma0_expand = Gamma0_expand[:, :, np.newaxis] # (n_compoents, 1, 1)
        reg = np.array([np.identity(n_features) * self.reg_covar] * self.n_compoents)
        self.covar = Gamma2 / Gamma0_expand - mean_expand @ mean_T_expand + reg
        self.weights = Gamma0 / sum(Gamma0)


    def _log_likelihood(self, X):
        n_samples, n_features = X.shape
        pdf = np.zeros((n_samples, self.n_compoents))
        for k in range(self.n_compoents):
            # print(self.covar[k] == self.covar[k].T)
            # e = np.linalg.eigvals(self.covar[k])
            # print(np.min(e))
            pdf[:, k] = multivariate_normal.pdf(X, mean=self.means[k], cov=self.covar[k])
        weights = self.weights[np.newaxis, :]
        p = np.sum(weights * pdf, axis=1)
        log_likelihood = np.sum(np.log(p), axis=0)

        return log_likelihood
    
    def _init_random(self, X):
        n_samples, n_features = X.shape

        random_indices = np.random.choice(n_samples, self.n_compoents, replace=False)
        self.means = X[random_indices]
        # 这里需要考虑是否各个Gaussian的协方差怎么初始化，不做处理就是默认为I

    def _init_kmeans(self, X):
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=self.n_compoents).fit(X)
        self.means = kmeans.cluster_centers_

        # 这里同sklearn一样，只考虑均值的初始化
        # for k in range(self.n_compoents):
        #     points = X[kmeans.labels_ == k]
        #     self.covar = np.cov(points)

    def predict(self, X):
        gamma = self._expectation_step(X)
        return np.argmax(gamma, axis=1)
    
    def score(self, X):
        return self._log_likelihood(X) / X.shape[0]
    

    

            
    
        
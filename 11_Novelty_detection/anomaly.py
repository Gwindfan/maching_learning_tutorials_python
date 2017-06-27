from sklearn.mixture import GaussianMixture

class GaussianMixtureNovelty(GaussianMixture):
    def __init__(self, threshold=0.05, n_components=1, covariance_type='full', tol=1e-3,
                 reg_covar=1e-6, max_iter=100, n_init=1, init_params='kmeans',
                 weights_init=None, means_init=None, precisions_init=None,
                 random_state=None, warm_start=False,
                 verbose=0, verbose_interval=10):
        super().__init__(n_components=n_components, covariance_type=covariance_type, tol=tol,
                         reg_covar=reg_covar, max_iter=max_iter, n_init=n_init, init_params=init_params,
                         weights_init=weights_init, means_init=means_init, precisions_init=precisions_init,
                         random_state=random_state, warm_start=warm_start,
                         verbose=verbose, verbose_interval=verbose_interval)
        
        self.threshold = threshold
    
    def prob_samples(self, test_data):
        scores = self.score_samples(test_data)
        return np.exp(scores)
    
    def predict(self, test_data):
        probs = self.prob_samples(test_data)
        outliers = np.where(probs <= self.threshold)
        result = np.zeros(len(test_data))
        result[outliers] = 1
        return result
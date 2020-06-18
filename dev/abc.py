import numpy as np
from scipy import linalg, stats
from functools import partial

class ABCSampler:
    def __init__(self, prior, candidate_getter, distance=lambda a, b: abs(a - b), statistic=np.mean):
        '''
        prior : stats.rv_continuous or stats.rv_discrete object
        Samples from the prior distribution over the parameters.
        Takes in nothing.
        Returns numpy.ndarray of parameters.

        candidate_getter : function
        Gets candidate samplers of the posterior distribution.
        Takes in parameters, as returned by prior.
        Returns function 'candidate' that samples from the candidate distribution, to be compared to true data, which
            Takes in the number of sample points to return.
            Returns numpy.ndarray of sample points of the desired shape.
        '''
        self.prior = prior
        self.candidate_getter = candidate_getter
        self.distance = distance
        self.statistic = statistic

    def sample(self, data, max_iters=float('inf'), threshold=1e-1, verbose=True):
        '''
        data : numpy.ndarray
        Data that we want to fit.
        Arbitrary shape, but a row should match the return type of candidate.

        max_iters : scalar
        The number of times to try and accept a candidate.

        threshold : scalar
        The maximum value of distance allowable to accept candidate.

        verbose : boolean
        Whether to print information like the distance at each step.
        '''
        num_iters = 0
        while True:
            params = self.prior.rvs()
            candidate = self.candidate_getter(params)
            synthetic = candidate(len(data))
            dis = self.distance(self.statistic(synthetic), self.statistic(data))
            if dis <= threshold:
                return params
            elif verbose:
                print(dis)
            num_iters += 1
            if num_iters > max_iters:
                return None

    def sample_pmc(self, data, thresholds, num_walkers):
        '''
        Carries out population Monte Carlo ABC, based on Beaumont et al. (2009).
        https://arxiv.org/pdf/0805.2256.pdf
        '''
        assert num_walkers > 1, "need at least 2 walkers to establish covariance."
        sampler = partial(self.sample, data=data, max_iters=float('inf'))
        params_matrix = np.array([sampler(threshold=thresholds[0]) for _ in range(num_walkers)])
        weights = np.ones((num_walkers,)) / num_walkers
        print(params_matrix)
        tau = 2 * np.cov(params_matrix.T)
        for thresh in thresholds[1:]:
            new_params_matrix = np.empty_like(params_matrix)
            new_weights = np.empty_like(weights)
            for i in range(num_walkers):
                center = params_matrix[np.random.choice(num_walkers, p=weights)]
                sampling_normal = stats.multivariate_normal(center, tau)
                param_i = sampler(prior=sampling_normal, threshold=thresh)
                new_params_matrix[i] = param_i
                new_weights[i] = self.prior.pdf(param_i) / np.dot(weights, np.prod(stats.norm.pdf(
                    np.linalg.inv(linalg.sqrtm(tau)).dot((param_i - params_matrix).T)), axis=0)) # is the sqrtm needed?
            params_matrix = new_params_matrix
            weights = new_weights / sum(new_weights)
        return np.mean(params_matrix, axis=0) # average the final results




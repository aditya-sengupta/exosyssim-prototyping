import numpy as np
from scipy import linalg, stats
from functools import partial
from matplotlib import pyplot as plt

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

    def sample(self, data, prior=None, max_iters=float('inf'), threshold=1e-1, verbose=True):
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
        if prior is None:
            prior = self.prior
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
                return np.ones_like(params) * np.nan

    def sample_pmc_helper(self, sample, params_matrix, weights, tau, num_to_get, thresh):
        choices = np.random.choice(len(params_matrix), (num_to_get,), p=weights, replace=True)
        centers = params_matrix[choices]
        sampling_normals = [stats.multivariate_normal(center, tau) for center in centers]
        return np.array([sample(prior=p, threshold=thresh, verbose=False, max_iters=5) for p in sampling_normals])

    def sample_pmc(self, data, thresholds, num_walkers, verbose=True):
        '''
        Carries out population Monte Carlo ABC, based on Beaumont et al. (2009).
        https://arxiv.org/pdf/0805.2256.pdf
        '''
        sample = partial(self.sample, data=data, max_iters=float('inf'), verbose=False)
        params_matrix = np.array([self.prior.rvs() for _ in range(num_walkers)])
        weights = np.ones((num_walkers,)) / num_walkers
        tau = 2 * np.cov(params_matrix.T)
        best_distance = min(self.distance(self.statistic(self.candidate_getter(p)(len(data))), self.statistic(data)) for p in params_matrix)
        try:
            k = 0
            thresh = thresholds[0]
            while thresh >= thresholds[-1]:
                print("Attempting to cross threshold {0} on a max of {1}".format(thresh, best_distance))
                new_params = self.sample_pmc_helper(sample, params_matrix, weights, tau, num_walkers, thresh)
                nans = [all(np.isnan(x)) for x in new_params]
                while any(nans):
                    none_inds = np.where(nans)[0]
                    thresh = np.sqrt(thresh * best_distance)
                    print("Raising threshold to {0}".format(thresh))
                    new_params[none_inds] = self.sample_pmc_helper(sample, params_matrix, weights, tau, len(none_inds), thresh)
                    nans = [all(np.isnan(x)) for x in new_params]
                best_distance = thresh
                new_weights = np.empty_like(weights)
                for i in range(num_walkers):
                    new_weights[i] = self.prior.pdf(new_params[i]) / np.dot(weights, np.prod(stats.norm.pdf(
                        np.linalg.inv(linalg.sqrtm(tau)).dot((new_params[i] - params_matrix).T)), axis=0)) # is the sqrtm needed?
                params_matrix = new_params
                weights = new_weights / sum(new_weights)
                tau = 2 * np.cov(params_matrix.T) 
                if np.isclose(thresh, thresholds[k], atol=1e-3):
                    k += 1
                    if k < len(thresholds):
                        thresh = thresholds[k]
                else:
                    thresh = np.sqrt(thresh * thresholds[k])
        except KeyboardInterrupt:
            pass
        return np.mean(params_matrix, axis=0) # average the final results

if __name__ == "__main__":
    prior = stats.multivariate_normal(np.array([40, 11]), np.diag([20, 2]))
    def candidate_getter(p):
        def candidate(size):
            return np.random.normal(p[0], np.abs(p[1]), size)
        return candidate

    gaussian_sampler = ABCSampler(prior, candidate_getter)
    data = np.random.normal(50, 10, (100,))
    print("Sample mean {0}, SD {1}".format(np.mean(data), np.std(data)))
    params = gaussian_sampler.sample_pmc(np.random.normal(50, 10, (100,)), [1, 0.5, 0.1], 7)
    print(params)
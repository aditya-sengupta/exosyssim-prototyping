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
                return None

    def sample_pmc(self, data, thresholds, num_walkers, verbose=True):
        '''
        Carries out population Monte Carlo ABC, based on Beaumont et al. (2009).
        https://arxiv.org/pdf/0805.2256.pdf
        '''
        if verbose:
            print("Attempting to clear threshold", thresholds[0])
        sample = partial(self.sample, data=data, max_iters=float('inf'), verbose=False)
        params_matrix = np.array([sample(threshold=thresholds[0]) for _ in range(num_walkers)])
        weights = np.ones((num_walkers,)) / num_walkers
        tau = 2 * np.cov(params_matrix.T)
        try:
            for k, thresh in enumerate(thresholds[1:]):
                if verbose:
                    print("Attempting to clear threshold", thresh)
                new_weights = np.empty_like(weights)
                best_thresh = thresholds[k]
                while not np.isclose(best_thresh, thresh):
                    print("setting best thresh", best_thresh, thresh)
                    new_params = []
                    for i in range(num_walkers):
                        center = params_matrix[np.random.choice(num_walkers, p=weights)]
                        sampling_normal = stats.multivariate_normal(center, tau)
                        param_i = sample(prior=sampling_normal, threshold=thresh, verbose=False, max_iters=20)
                        new_params.append(param_i)
                    sample_has_none = any([x is None for x in new_params])
                    print('found Nones? {0}'.format(sample_has_none))
                    if sample_has_none:
                        none_inds = np.where([x is None for x in new_params])[0]
                        thresh = np.sqrt(thresh * best_thresh)
                        if verbose:
                            print("Raising threshold to {0}".format(thresh))
                    else:
                        print(thresholds[k + 1], thresh)
                        # if not np.isclose(thresholds[k + 1], thresh):
                        print("hit this point")
                        new_weights[i] = self.prior.pdf(param_i) / np.dot(weights, np.prod(stats.norm.pdf(
                            np.linalg.inv(linalg.sqrtm(tau)).dot((param_i - params_matrix).T)), axis=0)) # is the sqrtm needed?
                        best_thresh = thresh
                        thresh = np.sqrt(thresh * thresholds[k + 1])
                        print("Lowering threshold to {0}".format(thresh))
                        params_matrix = np.array(new_params)
                        weights = new_weights / sum(new_weights)
                        tau = 2 * np.cov(params_matrix.T) 
        except KeyboardInterrupt:
            pass
        print(weights)
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
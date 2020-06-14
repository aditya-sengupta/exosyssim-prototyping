'''
This is a collection of external utility functions, combined with a couple of simple utilities I wrote.
So far, utility functions are only from dfm.io/posts/exopop/ (hence the file name.)
'''

def population_model(theta, period, rp, period_rng, rp_rng):
    lnf0, beta, alpha = theta
    v = np.exp(lnf0) * np.ones_like(period)
    for x, rng, n in zip((period, rp),
                         (period_rng, rp_rng),
                         (beta, alpha)):
        n1 = n + 1
        v *= x**n*n1 / (rng[1]**n1-rng[0]**n1)
    return v

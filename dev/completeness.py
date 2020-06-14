'''
Utility functions for completeness, and a script to generate and save completeness contours.
Mostly directly from dfm.io/posts/exopop/.
'''

import numpy as np
from scipy.stats import gamma
from .dataprocessing import get_stellar_keys

def get_duration(period, aor, e):
    """
    Equation (1) from Burke et al. This estimates the transit
    duration in the same units as the input period. There is a
    typo in the paper (24/4 = 6 != 4).

    :param period: the period in any units of your choosing
    :param aor:    the dimensionless semi-major axis (scaled
                   by the stellar radius)
    :param e:      the eccentricity of the orbit

    """
    return 0.25 * period * np.sqrt(1 - e**2) / aor

def get_a(period, mstar, Go4pi=2945.4625385377644/(4*np.pi*np.pi)):
    """
    Compute the semi-major axis of an orbit in Solar radii.

    :param period: the period in days
    :param mstar:  the stellar mass in Solar masses

    """
    return (Go4pi*period*period*mstar) ** (1./3)

def get_delta(k, c=1.0874, s=1.0187):
    """
    Estimate the approximate expected transit depth as a function
    of radius ratio. There might be a typo here. In the paper it
    uses c + s*k but in the public code, it is c - s*k:
    https://github.com/christopherburke/KeplerPORTs

    :param k: the dimensionless radius ratio between the planet and
              the star

    """
    delta_max = k*k * (c + s*k)
    return 0.84 * delta_max

def get_cdpp():
    cdpp_cols = [k for k in get_stellar_keys() if k.startswith("rrmscdpp")]
    cdpp_vals = np.array([k[-4:].replace("p", ".") for k in cdpp_cols], dtype=float)
    return cdpp_cols, cdpp_vals

def get_mes(star, period, rp, tau, re=0.009171):
    """
    Estimate the multiple event statistic value for a transit.

    :param star:   a pandas row giving the stellar properties
    :param period: the period in days
    :param rp:     the planet radius in Earth radii
    :param tau:    the transit duration in hours

    """
    # Interpolate to the correct CDPP for the duration.
    cdpp_cols, cdpp_vals = get_cdpp()
    cdpp = np.array(star[cdpp_cols], dtype=float)
    sigma = np.interp(tau, cdpp_vals, cdpp)

    # Compute the radius ratio and estimate the S/N.
    k = rp * re / star.radius
    snr = get_delta(k) * 1e6 / sigma

    # Scale by the estimated number of transits.
    ntrn = star.dataspan * star.dutycycle / period
    return snr * np.sqrt(ntrn)

# Pre-compute and freeze the gamma function from Equation (5) in
# Burke et al.

def make_gamma():
    pgam = gamma(4.65, loc=0., scale=0.98)
    mesthres_cols = [k for k in stellar_keys if k.startswith("mesthres")]
    mesthres_vals = np.array([k[-4:].replace("p", ".") for k in mesthres_cols],
                        dtype=float)
    return pgam, mesthres_cols, mesthres_vals

def get_pdet(star, aor, period, rp, e, pgam, mesthres_cols, mesthres_vals):
    """
    Equation (5) from Burke et al. Estimate the detection efficiency
    for a transit.

    :param star:   a pandas row giving the stellar properties
    :param aor:    the dimensionless semi-major axis (scaled
                   by the stellar radius)
    :param period: the period in days
    :param rp:     the planet radius in Earth radii
    :param e:      the orbital eccentricity

    """
    tau = get_duration(period, aor, e) * 24.
    mes = get_mes(star, period, rp, tau)
    mest = np.interp(tau, mesthres_vals,
                     np.array(star[mesthres_cols],
                              dtype=float))
    x = mes - 4.1 - (mest - 7.1)
    return pgam.cdf(x)

def get_pwin(star, period):
    """
    Equation (6) from Burke et al. Estimates the window function
    using a binomial distribution.

    :param star:   a pandas row giving the stellar properties
    :param period: the period in days

    """
    M = star.dataspan / period
    f = star.dutycycle
    omf = 1.0 - f
    pw = 1 - omf**M - M*f*omf**(M-1) - 0.5*M*(M-1)*f*f*omf**(M-2)
    msk = (pw >= 0.0) & (M >= 2.0)
    return pw * msk

def get_pgeom(aor, e):
    """
    The geometric transit probability.

    See e.g. Kipping (2014) for the eccentricity factor
    http://arxiv.org/abs/1408.1393

    :param aor: the dimensionless semi-major axis (scaled
                by the stellar radius)
    :param e:   the orbital eccentricity

    """
    return 1. / (aor * (1 - e*e)) * (aor > 1.0)

def get_completeness(star, period, rp, e, pgam, mesthres_cols, mesthres_vals, with_geom=True):
    """
    A helper function to combine all the completeness effects.

    :param star:      a pandas row giving the stellar properties
    :param period:    the period in days
    :param rp:        the planet radius in Earth radii
    :param e:         the orbital eccentricity
    :param stlr:      the stellar catalog
    :param with_geom: include the geometric transit probability?

    pgam, mesthres_cols, mesthres_vals are here to make these functions pure.
    """
    aor = get_a(period, star.mass) / star.radius
    pdet = get_pdet(star, aor, period, rp, e, pgam, mesthres_cols, mesthres_vals)
    pwin = get_pwin(star, period)
    if not with_geom:
        return pdet * pwin
    pgeom = get_pgeom(aor, e)
    return pdet * pwin * pgeom

def make_comp(stlr, period_grid, rp_grid, name=None):
    pgam, mesthres_cols, mesthres_vals = make_gamma()
    comp = np.zeros_like(period_grid)
    for _, star in stlr.iterrows():
        comp += get_completeness(star, period_grid, rp_grid, 0.0, pgam, mesthres_cols, mesthres_vals, with_geom=True)
    if name:
        np.save('data/comp_{0}.npy'.format(name), comp)
    return comp

def pcomp_vectors(stars, periods, rp, eccs):
    '''
    Self-contained, returns pcomp over matched arrays of planets around stars.
    '''
    c = 1.0874
    s = 1.0187
    Go4pi = 2945.4625385377644/(4*np.pi*np.pi)
    re = 0.009171
    
    mstars = np.array(stars['mass'])
    rstars = np.array(stars['radius'])
    cdpp = np.array(stars[cdpp_cols], dtype=float)
    dataspan = np.array(stars['dataspan'])
    dutycycle = np.array(stars['dutycycle'])
    mesthres_cols_stars = np.array(stars[mesthres_cols], dtype=float)
    
    aor = (Go4pi*periods*periods*mstars) ** (1./3) / rstars
    tau = 6 * periods * np.sqrt(1 - eccs**2) / aor

    # sigma = np.apply_along_axis(np.interp, 0, tau, cdpp_vals, cdpp)
    sigma = np.array([np.interp(tau[i], cdpp_vals, cdpp[i]) for i in range(len(tau))])
    # Compute the radius ratio and estimate the S/N.
    k = rp * re / rstars
    delta = 0.84 * k*k * (c + s*k)
    snr = delta * 1e6 / sigma

    # Scale by the estimated number of transits.
    ntrn = dataspan * dutycycle / periods
    mess = snr * np.sqrt(ntrn)
    mest = np.array([np.interp(tau[i], mesthres_vals, mesthres_cols_stars[i]) for i in range(len(tau))])
    x = mess - 4.1 - (mest - 7.1)
    pdets = pgam.cdf(x)
    
    M = dataspan / periods
    f = dutycycle
    omf = 1.0 - f
    pw = 1 - omf**M - M*f*omf**(M-1) - 0.5*M*(M-1)*f*f*omf**(M-2)
    msk = (pw >= 0.0) & (M >= 2.0)
    pwins = pw * msk
    
    pgeom = 1. / (aor * (1 - eccs*eccs)) * (aor > 1.0)
    
    return pdets * pwins * pgeom

if __name__ == "__main__":
    bins = "hsu" # "hsu" or "dfm"

    if bins == "dfm":
        period_rng = (50, 300)
        rp_rng = (0.75, 2.5)
        period = np.linspace(period_rng[0], period_rng[1], 57)
        rp = np.linspace(rp_rng[0], rp_rng[1], 61)
    elif bins == "hsu":
        period = np.array([0.5, 1.25, 2.5, 5, 10, 20, 40, 80, 160, 320])
        rp = np.array([0.5, 0.75, 1, 1.25, 1.5, 1.75, 2, 2.5, 3, 4, 6, 8, 12, 16])
        period_rng = (min(period), max(period))
        rp_rng = (min(rp), max(rp))

    period_grid, rp_grid = np.meshgrid(period, rp, indexing="ij")

    stlr = get_stellar()
    stlr = stlr[np.isfinite(stlr.mass)]
    stlr = stellar_cuts(stlr, cut_type=bins)

    # kois = get_kois()
    # kois = kois_cuts(kois[kois["kepid"].isin(stlr["kepid"])], period_rng, rp_rng)

    make_comp(stlr, period_grid, rp_grid, name=bins)

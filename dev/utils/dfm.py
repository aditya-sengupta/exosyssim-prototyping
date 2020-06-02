'''
This is a collection of external utility functions, combined with a couple of simple utilities I wrote.
So far, utility functions are only from dfm.io/posts/exopop/ (hence the file name.)
'''

from matplotlib import pyplot as pl
import numpy as np
import pandas as pd
from scipy.stats import gamma

import os
import requests
import pandas as pd
from io import BytesIO
import sys
sys.path.append('..')

def get_catalog(name, basepath="data"):
    fn = os.path.join(basepath, "{0}.h5".format(name))
    if os.path.exists(fn):
        return pd.read_hdf(fn, name)
    if not os.path.exists(basepath):
        os.makedirs(basepath)
    print("Downloading {0}...".format(name))
    url = ("http://exoplanetarchive.ipac.caltech.edu/cgi-bin/nstedAPI/"
           "nph-nstedAPI?table={0}&select=*").format(name)
    r = requests.get(url)
    if r.status_code != requests.codes.ok:
        r.raise_for_status()
    fh = BytesIO(r.content)
    df = pd.read_csv(fh)
    df.to_hdf(fn, name, format="t")
    return df

def stellar_cuts(stlr, cut_to_gk=True):
    m = stlr.kepid >= 0
    if cut_to_gk:
        m = (4200 <= stlr.teff) & (stlr.teff <= 6100)
        m &= stlr.radius <= 1.15

    # Only include stars with sufficient data coverage.
    m &= stlr.dataspan > 365.25*2.
    m &= stlr.dutycycle > 0.6
    m &= stlr.rrmscdpp07p5 <= 1000.

    # Only select stars with mass estimates.
    m &= np.isfinite(stlr.mass)

    base_stlr = pd.DataFrame(stlr)
    stlr = pd.DataFrame(stlr[m])

    print("Selected {0} targets after cuts".format(len(stlr)))
    return stlr

def get_stellar():
    return get_catalog('q1_q16_stellar')

stellar_keys = get_stellar().keys()

def get_kois():
    return get_catalog('q1_q16_koi')

def kois_cuts(kois, period_rng, rp_rng):
    m = kois.koi_pdisposition == "CANDIDATE"
    base_kois = pd.DataFrame(kois[m])
    m &= (period_rng[0] <= kois.koi_period) & (kois.koi_period <= period_rng[1])
    m &= np.isfinite(kois.koi_prad) & (rp_rng[0] <= kois.koi_prad) & (kois.koi_prad <= rp_rng[1])

    kois = pd.DataFrame(kois[m])

    print("Selected {0} KOIs after cuts".format(len(kois)))
    return kois

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
    cdpp_cols = [k for k in stellar_keys if k.startswith("rrmscdpp")]
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
    msk = (pw >= 0.0) * (M >= 2.0)
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

def population_model(theta, period, rp, period_rng, rp_rng):
    lnf0, beta, alpha = theta
    v = np.exp(lnf0) * np.ones_like(period)
    for x, rng, n in zip((period, rp),
                         (period_rng, rp_rng),
                         (beta, alpha)):
        n1 = n + 1
        v *= x**n*n1 / (rng[1]**n1-rng[0]**n1)
    return v

def make_plot(pop_comp, x0, x, y, ax):
    pop = 0.5 * (pop_comp[:, 1:] + pop_comp[:, :-1])
    pop = np.sum(pop * np.diff(y)[None, :, None], axis=1)
    a, b, c, d, e = np.percentile(pop * np.diff(x)[0], [2.5, 16, 50, 84, 97.5], axis=0)

    ax.fill_between(x0, a, e, color="k", alpha=0.1, edgecolor="none")
    ax.fill_between(x0, b, d, color="k", alpha=0.3, edgecolor="none")
    ax.plot(x0, c, "k", lw=1)

def plot_results(samples, kois, koi_periods, koi_rps, period_grid, rp_grid, comp=None):
    if comp is None:
        print("rerunning completeness")
        comp = make_comp()
    period_rng = (min(koi_periods), max(koi_periods))
    rp_rng = (min(koi_rps), max(koi_rps))
    # Loop through the samples and compute the list of population models.
    samples = np.atleast_2d(samples)
    pop = np.empty((len(samples), period_grid.shape[0], period_grid.shape[1]))
    gamma_earth = np.empty((len(samples)))
    for i, p in enumerate(samples):
        pop[i] = population_model(p, period_grid, rp_grid)
        gamma_earth[i] = population_model(p, 365.25, 1.0) * 365.

    fig, axes = pl.subplots(2, 2, figsize=(10, 8))
    fig.subplots_adjust(wspace=0.4, hspace=0.4)

    # Integrate over period.
    dx = 0.25
    x = np.arange(rp_rng[0], rp_rng[1] + dx, dx)
    n, _ = np.histogram(koi_rps, x)

    # Plot the observed radius distribution.
    ax = axes[0, 0]
    make_plot(pop * comp[None, :, :], rp, x, period, ax)
    ax.errorbar(0.5*(x[:-1]+x[1:]), n, yerr=np.sqrt(n), fmt=".k",
                capsize=0)
    ax.set_xlim(rp_rng[0], rp_rng[1])
    ax.set_xlabel("$R_p\,[R_\oplus]$")
    ax.set_ylabel("\# of detected planets")

    # Plot the true radius distribution.
    ax = axes[0, 1]
    make_plot(pop, rp, x, period, ax)
    ax.set_xlim(rp_rng[0], rp_rng[1])
    ax.set_ylim(0, 0.37)
    ax.set_xlabel("$R_p\,[R_\oplus]$")
    ax.set_ylabel("$\mathrm{d}N / \mathrm{d}R$; $\Delta R = 0.25\,R_\oplus$")

    # Integrate over period.
    dx = 31.25
    x = np.arange(period_rng[0], period_rng[1] + dx, dx)
    n, _ = np.histogram(koi_periods, x)

    # Plot the observed period distribution.
    ax = axes[1, 0]
    make_plot(np.swapaxes(pop* comp[None, :, :], 1, 2), period, x, rp, ax)
    ax.errorbar(0.5*(x[:-1]+x[1:]), n, yerr=np.sqrt(n), fmt=".k",
                capsize=0)
    ax.set_xlim(period_rng[0], period_rng[1])
    ax.set_ylim(0, 79)
    ax.set_xlabel("$P\,[\mathrm{days}]$")
    ax.set_ylabel("\# of detected planets")

    # Plot the true period distribution.
    ax = axes[1, 1]
    make_plot(np.swapaxes(pop, 1, 2), period, x, rp, ax)
    ax.set_xlim(period_rng[0], period_rng[1])
    ax.set_ylim(0, 0.27)
    ax.set_xlabel("$P\,[\mathrm{days}]$")
    ax.set_ylabel("$\mathrm{d}N / \mathrm{d}P$; $\Delta P = 31.25\,\mathrm{days}$")

    return gamma_earth

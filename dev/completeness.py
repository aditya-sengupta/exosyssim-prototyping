'''
Utility functions for completeness, and a script to generate and save completeness contours.
Refactored from source at dfm.io/posts/exopop/.
'''

import numpy as np
from scipy.stats import gamma
from .dataprocessing import get_stellar_keys

stellar_keys = get_stellar_keys()

re = 0.009171

# Pre-compute and freeze the gamma function from Equation (5) in
# Burke et al., along with the CDPP and MES values/thresholds.

def get_precomp_params():
    cdpp_cols = [k for k in stellar_keys if k.startswith("rrmscdpp")]
    cdpp_vals = np.array([k[-4:].replace("p", ".") for k in cdpp_cols], dtype=float)
    pgam = gamma(4.65, loc=0., scale=0.98)
    mesthres_cols = [k for k in stellar_keys if k.startswith("mesthres")]
    mesthres_vals = np.array([k[-4:].replace("p", ".") for k in mesthres_cols],
                             dtype=float)
    return cdpp_cols, cdpp_vals, pgam, mesthres_cols, mesthres_vals
    
def get_pdet(periods, rp, eccs, stars=None, derived_params=None, precomp_params=None):
    if precomp_params is None:
        cdpp_cols, cdpp_vals, pgam, mesthres_cols, mesthres_vals = get_precomp_params()
    else:
        cdpp_cols, cdpp_vals, pgam, mesthres_cols, mesthres_vals = precomp_params
        
    if derived_params is None:
        Go4pi = 2945.4625385377644/(4*np.pi*np.pi)
        mstars = stars['mass'].values
        rstars = stars['radius'].values
        cdpp = np.array(stars[cdpp_cols], dtype=float)
        dataspan = stars['dataspan'].values
        dataspan = np.nan_to_num(dataspan)
        dutycycle = stars['dutycycle'].values
        dutycycle = np.nan_to_num(dutycycle)
        mesthres_cols_stars = np.array(stars[mesthres_cols], dtype=float)
        aor = (Go4pi*periods*periods*mstars) ** (1./3) / rstars
    else:
        mstars, rstars, cdpp, dataspan, dutycycle, mesthres_cols_stars, aor = derived_params
        
    c = 1.0874
    s = 1.0187
    tau = 6 * periods * np.sqrt(1 - eccs**2) / aor
    sigma = np.array([np.interp(tau[i], cdpp_vals, cdpp[i]) for i in range(len(tau))])
    sigma = np.nan_to_num(sigma, nan=1)
    
    # Compute the radius ratio and estimate the S/N
    k = rp * re / rstars
    delta = 0.84 * k*k * (c + s*k)
    snr = delta * 1e6 / sigma

    # Scale by the estimated number of transits.
    ntrn = dataspan * dutycycle / periods
    multiple_event_stats = snr * np.sqrt(ntrn)
    
    mesthreshes = np.array([np.interp(tau[i], mesthres_vals, mesthres_cols_stars[i]) for i in range(len(tau))])
    mesthreshes = np.nan_to_num(mesthreshes, nan=7.1)
    x = multiple_event_stats - 4.1 - (mesthreshes - 7.1)
    pdets = pgam.cdf(x)
    return pdets
    
def get_pwin(periods, dataspan, dutycycle):
    M = dataspan / periods
    f = dutycycle
    omf = 1.0 - f
    pw = 1 - omf**M - M*f*omf**(M-1) - 0.5*M*(M-1)*f*f*omf**(M-2)
    msk = (pw >= 0.0) & (M >= 2.0)
    pwins = pw * msk
    return pwins
    
def get_pgeom(cosincls, eccs, periods=None, mstars=None, rstars=None, aor=None):
    '''
    Get geometric probability of a transit. Note that this is the Hsu (2018) version, so outputs are all 0 or 1:
    the planet doesn't or does transit its host star.
    '''
    if aor is None:
        Go4pi = 2945.4625385377644/(4*np.pi*np.pi)
        aor = (Go4pi*periods*periods*mstars) ** (1./3) / rstars
    return np.nan_to_num(aor * cosincls * (1 - eccs * eccs) <= 1)

def get_pcomp(periods, rp, eccs, cosincls, stars, precomp_params=None):
    '''
    Self-contained, returns pcomp over matched arrays of planets around stars.
    
    Arguments
    ---------
    stars : pd.DataFrame
    A sample of the stellar catalog whose rows match the planets.
    
    periods, rp, eccs, cosincls : numpy.ndarray
    Periods, radii, eccentricities, cosine-inclinations of the planets in vectors.

    precomp_params : list or None
    The precomputed parameters indicating the CDPP values, the MES threshold values, and the gamma distribution.
    '''
    if precomp_params is None:
        cdpp_cols, cdpp_vals, pgam, mesthres_cols, mesthres_vals = get_precomp_params()
    else:
        cdpp_cols, cdpp_vals, pgam, mesthres_cols, mesthres_vals = precomp_params
        
    Go4pi = 2945.4625385377644/(4*np.pi*np.pi)
    mstars = stars['mass'].values
    rstars = stars['radius'].values
    cdpp = np.array(stars[cdpp_cols], dtype=float)
    dataspan = stars['dataspan'].values
    dataspan = np.nan_to_num(dataspan)
    dutycycle = stars['dutycycle'].values
    dutycycle = np.nan_to_num(dutycycle)
    mesthres_cols_stars = np.array(stars[mesthres_cols], dtype=float)
    aor = (Go4pi*periods*periods*mstars) ** (1./3) / rstars
    
    derived_params = [mstars, rstars, cdpp, dataspan, dutycycle, mesthres_cols_stars, aor]
    
    pdets = get_pdet(periods, rp, eccs, derived_params=derived_params)
    pwins = get_pwin(periods, dataspan, dutycycle)
    pgeom = get_pgeom(cosincls, eccs, aor=aor)
    
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

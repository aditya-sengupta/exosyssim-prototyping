'''
Utility functions for completeness, and a script to generate and save completeness contours.
Refactored from source at dfm.io/posts/exopop/.
'''

import numpy as np
from scipy.stats import gamma
from utils import get_snr, get_tau, cdpp_cols, mesthres_cols, mesthres_vals
from constants import pgam, re, Go4pi, c, s

# Pre-compute and freeze the gamma function from Equation (5) in
# Burke et al., along with the CDPP and MES values/thresholds.
    
def get_pdet(periods, rp, eccs, stars=None, derived_params=None):        
    if derived_params is None:
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
        
    tau = get_tau(periods, eccs, aor)
    snr = get_snr(rp, rstars, cdpp, tau=tau)

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
        aor = (Go4pi*periods*periods*mstars) ** (1./3) / rstars
    return np.nan_to_num(aor * cosincls * (1 - eccs * eccs) <= 1)

def get_pcomp(periods, rp, eccs, cosincls, stars):
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

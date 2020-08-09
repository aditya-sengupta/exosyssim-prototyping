'''
Data processing utilities, mostly directly from dfm.io/posts/exopop/.
'''

from io import BytesIO
import numpy as np
import pandas as pd
import os
import requests

from .constants import re, c, s
from .constants import cdpp_cols, cdpp_vals, mesthres_cols, mesthres_vals

def get_catalog(name, basepath="../data", **kwargs):
    fn = os.path.join(basepath, "{0}.h5".format(name))
    if os.path.exists(fn):
        return pd.read_hdf(fn, name, **kwargs)
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

def get_kepler_stellar_keys():
    return get_catalog('q1_q16_stellar', start=0, stop=0).keys()

def stellar_cuts(stlr, cut_type="dfm"):
    m = stlr.kepid >= 0
    if cut_type == "dfm":
        m = (4200 <= stlr.teff) & (stlr.teff <= 6100)
        m &= stlr.radius <= 1.15
        m &= stlr.dataspan > 365.25*2.
        m &= stlr.dutycycle > 0.6
        m &= stlr.rrmscdpp07p5 <= 1000.
    elif cut_type == "hsu":
        m = (4000 <= stlr.teff) & (stlr.teff <= 7000)
        m &= stlr.logg >= 4.0
        m &= stlr.dataspan > 365.25 / 4
        m &= np.isfinite(stlr.radius)
        m &= np.isfinite(stlr.rrmscdpp07p5)
        m &= np.isfinite(stlr.dens)

    # Only select stars with mass estimates.
    m &= np.isfinite(stlr.mass)
    stlr = pd.DataFrame(stlr[m])

    print("Selected {0} targets after cuts".format(len(stlr)))
    return stlr

def get_kepler_stellar():
    return get_catalog('q1_q16_stellar')

def get_kois():
    return get_catalog('q1_q16_koi')

def get_tess_catalog():
    # tbd: API for this https://exoplanetarchive.ipac.caltech.edu/cgi-bin/TblView/nph-tblView?app=ExoTbls&config=TOI
    return pd.read_csv('../data/TOI_2020.06.30_10.44.38.csv', comment='#')

def kois_cuts(kois, period_rng, rp_rng):
    m = kois.koi_pdisposition == "CANDIDATE"
    m &= (period_rng[0] <= kois.koi_period) & (kois.koi_period <= period_rng[1])
    m &= np.isfinite(kois.koi_prad) & (rp_rng[0] <= kois.koi_prad) & (kois.koi_prad <= rp_rng[1])

    kois = pd.DataFrame(kois[m])

    print("Selected {0} KOIs after cuts".format(len(kois)))
    return kois

def get_paired_kepler_catalogs():
    kois = get_kois()
    stellar = get_kepler_stellar()
    kois = kois[kois["kepid"].isin(stellar["kepid"])]
    kois = kois[np.isfinite(kois["koi_prad"])]
    stellar = stellar[np.isfinite(stellar.mass)]
    return kois, stellar

def get_snr(rp, rstar, cdpp, period=None, ecc=None, aor=None, tau=None):
    if tau is None:
        tau = get_tau(period, ecc, aor)
    if isinstance(tau, float) or isinstance(tau, int):
        sigma = np.interp(tau, cdpp_vals, cdpp)
    else:
        assert len(cdpp) == len(tau), "cdpp axis 0 must be length of sample"
        sigma = np.array([np.interp(tau[i], cdpp_vals, cdpp[i]) for i in range(len(tau))])
    sigma = np.nan_to_num(sigma, nan=1)
    k = rp * re / rstar
    d = 0.84 * k*k * (c + s*k)
    return d * 1e6 / sigma

def get_tau(period, ecc, aor):
    return 6 * period * np.sqrt(1 - ecc**2) / aor

def get_bins(bins):
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

    return period, period_rng, rp, rp_rng

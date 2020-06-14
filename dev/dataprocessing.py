'''
Data processing utilities, mostly directly from dfm.io/posts/exopop/.
'''

from io import BytesIO
import pandas as pd
import os
import requests
import pandas as pd
import sys
sys.path.append('..')

def get_catalog(name, basepath="../data"):
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
    
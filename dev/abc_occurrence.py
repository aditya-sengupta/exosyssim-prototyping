import numpy as np
import scipy.stats as stats
import pandas as pd

from abc_generic import ABCSampler
from completeness import get_pcomp
from utils import get_paired_kepler_catalogs, get_snr, cdpp_cols, cdpp_vals
from constants import minmeanmax, Go4pi, re, eps
from covariance import make_cov

def log_with_extra(bins):
    log_bins = np.log(bins)
    return np.append(log_bins, max(log_bins) + 1) # arbitrary finite number
    # this is inelegant and should be changed later

def generate_planet_catalog(f, stars, log_p_bins, log_r_bins, add_noise=True):
    '''
    Takes in the occurrence rate matrix f and generates `num_planets' planets, 
    represented as a matrix of state (row) vectors.
    
    Each row has: period in days; radius in Earth radii; eccentricity; cosine of inclination; impact parameter;
    longitude of periastron; fractional transit depth; transit duration in days; is_detected (bool).
    
    Arguments
    ---------
    f : np.ndarray
    f = f(r, p); occurrence rate prior that we sample p and r from.
    
    stars : pd.dataframe
    A sample of the stellar dataframe.
    '''

    planet_params = ['kepid', 'period', 'prad', 'ecc', 'cosincl', 'b', 'omegas', 'd', 'D']
    lam = np.sum(f)
    nums_planets = np.minimum(stats.poisson(lam).rvs((len(stars),)), 10)
    num_planets = sum(nums_planets)
    stellar_catalog = pd.DataFrame(np.repeat(stars.values, nums_planets, axis=0))
    stellar_catalog.columns = stars.columns
    stellar_catalog = stellar_catalog.astype(stars.dtypes)
    
    flat_f = f.flatten()
    buckets = np.random.choice(f.size, p = flat_f / lam, size=(num_planets,))
    p_left_inds = buckets // f.shape[1]
    p_right_inds = p_left_inds + 1
    r_left_inds = buckets % f.shape[1]
    r_right_inds = r_left_inds + 1
    periods = np.exp(np.random.uniform(log_p_bins[p_left_inds], log_p_bins[p_right_inds]))
    pl_rads = np.exp(np.random.uniform(log_r_bins[r_left_inds], log_r_bins[r_right_inds]))
    eccens  = stats.rayleigh(scale=0.03).rvs(size=(num_planets,))
    cosincl = np.random.uniform(0, 1, size=(num_planets,))
    impacts = np.random.uniform(0, 1, size=(num_planets,))
    omegas = np.random.uniform(0, 2 * np.pi, size=(num_planets,))
    ror = pl_rads * re / stellar_catalog.radius.values
    depths = ror ** 2 
    aor = (Go4pi * periods * periods * stellar_catalog.mass.values) ** (1./3) / stellar_catalog.radius.values
    arcsin_args = np.sqrt((1 + ror) ** 2 - impacts ** 2) / aor
    problems = np.where(arcsin_args > 1)[0]
    impacts[problems] = np.minimum(1 - eps, np.sqrt((1 + ror[problems]) ** 2 - (aor[problems]) ** 2))
    # some impacts are 1 + eps
    arcsin_args[problems] = 1
    D = (periods / np.pi) * np.arcsin(arcsin_args)
    tau0 = periods * impacts / (2 * np.pi * cosincl * np.sqrt(1 - eccens ** 2)) * 1 / (aor ** 2)
    T = 2 * tau0 * np.sqrt(1 - impacts ** 2)
    tau = 2 * tau0 * np.divide(ror, np.sqrt(1 - impacts ** 2))
    sigma = 1 # 'model uncertainty'
    nums_transits = stellar_catalog['dataspan'].values * stellar_catalog['dutycycle'].values / periods
    snr = get_snr(pl_rads, stellar_catalog.radius.values, stellar_catalog[cdpp_cols].values, tau=tau)

    if add_noise:
        cov = make_cov(depths, T, tau, periods, nums_transits, snr, sigma, diagonal=True)
        noise = np.array([np.random.multivariate_normal(np.zeros(len(cov[i]),), cov[i]).T for i in range(num_planets)]).T
        periods += noise[1] 
        omegas += noise[0] / periods * 2 * np.pi # I think
        D += noise[2]
        # depths += noise[3] # WARNING: transit depth noise is weird (check units?)
        pl_rads = np.sqrt(depths) * stellar_catalog.radius / re

    # [kepid, period, radius, ecc, cosincl, impact param b, long of periastron omega, transit depth d, transit duration D, is_detected]
    planets_matrix = np.vstack((stellar_catalog.kepid, periods, pl_rads, eccens, cosincl, impacts, omegas, depths, D)).T
    planetary_catalog = pd.DataFrame(planets_matrix, columns=planet_params)
    probs = get_pcomp(periods, pl_rads, eccens, cosincl, stellar_catalog)
    probs = np.array(probs)
    probs = np.nan_to_num(probs)
    if np.allclose(probs, 0):
        print("all probabilities of detection are 0")
    detected = np.random.binomial(1, np.nan_to_num(probs))
    planetary_catalog['is_detected'] = detected
    return stellar_catalog, planetary_catalog.astype({'kepid': np.int64, 'is_detected': bool})

def kois_to_synth_catalog(period_bins, rp_bins, kois=None, stellar=None):
    if kois is None or stellar is None:
        kois, stellar = get_paired_kepler_catalogs()
    f = np.histogram2d(kois.koi_period, kois.koi_prad, bins=[period_bins, rp_bins])[0] / len(stellar)
    stellar_catalog_abc = pd.DataFrame(np.vstack([stellar[stellar.kepid == k].to_numpy() for k in kois.kepid]))
    stellar_catalog_abc.columns = stellar.columns
    stellar_catalog_abc = stellar_catalog_abc.astype(stellar.dtypes)
    kois_catalog_abc = kois.rename(columns={'koi_period': 'period', 'koi_prad': 'prad'})
    kois_catalog_abc["is_detected"] = kois_catalog_abc["koi_disposition"] == "CONFIRMED"
    kois_abc = [SyntheticCatalog(f, stellar_catalog_abc, period_bins, rp_bins, planets=kois_catalog_abc)]
    return kois_abc

class SyntheticCatalog:
    '''
    Quick encapsulation for paired stellar/planetary catalogs.
    '''
    def __init__(self, f, stars, period_bins, rp_bins, planets=None):
        self.f = f
        self.period_bins = period_bins
        self.rp_bins = rp_bins
        if planets is not None:
            self.planets = planets
            self.stars = stars
        else:
            self.stars, self.planets = generate_planet_catalog(f, stars, log_with_extra(period_bins), log_with_extra(rp_bins))

class OccurrencePrior(stats.rv_continuous):
    def __init__(self, period_bins, rp_bins):
        self.period_bins = period_bins
        self.rp_bins = rp_bins

    def pdf(self, f):
        return np.prod(np.exp(-f))
    
    def rvs(self):
        return stats.expon.rvs(size=(len(self.period_bins) * len(self.rp_bins),))

class OccurrenceABCSampler(ABCSampler):
    def __init__(self, period_bins, rp_bins, stellar_sample):
        self.prior = OccurrencePrior(period_bins, rp_bins)
        self.stellar_sample = stellar_sample
        self.period_bins, self.rp_bins = period_bins, rp_bins
        self.log_p_bins = log_with_extra(self.period_bins)
        self.log_r_bins = log_with_extra(self.rp_bins)
    
    def distance(self, s1, s2):
        return np.sum((s1 - s2) ** 2)

    def statistic(self, catalogs):
        Nstars = 0
        N = np.zeros((len(self.log_p_bins) - 1, len(self.log_r_bins) - 1))
        for catalog in catalogs:
            Nstars += len(catalog.stars)
            detected = catalog.planets[catalog.planets.is_detected]
            period_counts = np.digitize(np.log(detected.period.values), self.log_p_bins)
            rp_counts = np.digitize(np.log(detected.prad.values), self.log_r_bins)
            for i, j in zip(period_counts, rp_counts):
                if i < N.shape[0] and j < N.shape[1]: # exclude some on the edges
                    N[i][j] += 1
        return N / Nstars

    def candidate_getter(self, f):
        def candidate(size=1):
            return [
                SyntheticCatalog(
                    f.reshape(len(self.period_bins), len(self.rp_bins)), 
                    self.stellar_sample, 
                    self.period_bins,
                    self.rp_bins)
                for _ in range(size)
            ]
        return candidate

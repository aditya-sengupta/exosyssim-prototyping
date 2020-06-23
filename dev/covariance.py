'''
Implements the covariance matrix from Price and Rogers, 2014.
Adapted from Julia source at https://github.com/ExoJulia/ExoplanetsSysSim.jl/blob/master/src/transit_observations.jl.
'''

import numpy as np

from constants import kepler_texp as texp
from constants import LC_rate
from utils import get_catalog, get_snr
import warnings

def make_pd(matrix):
	'''
	Performs ridge regression to make a matrix positive definite.

	Arguments
	---------
	matrix : np.ndarray
	The matrix to change.

	Returns
	-------
	matrix_pd : np.ndarray
	The matrix with the smallest eigenvalue shifted.
	'''
	eps = 1e-3
	smallest_eig = np.min(np.linalg.eigvals(matrix))
	if smallest_eig > 0:
		return matrix
	else:
		return matrix + (1 + eps) * abs(smallest_eig) * np.identity(matrix.shape[0])

def make_cov(delta, T, tau, period, num_tr, snr, sigma, diagonal=False):
	'''
	Makes the covariance matrix as in Equation A8 from Price and Rogers, 2014.
	cov is the covariance matrix of {t_c, tau, T, delta}.

	Arguments
	---------
	delta, T, tau, period, num_tr, snr : scalar
	Transit parameters. 
	delta - transit depth
	T - FWHM transit duration
	tau - ingress/egress duration
	period - period of transit
	num_tr - number of transits
	snr - signal-to-noise ratio of the transit

	sigma : scalar
	Noise scale factor.

	diagonal : boolean
	Whether to keep only diagonal elements.
	'''
	gamma = LC_rate * num_tr

	tau3 = tau ** 3
	texp3 = texp ** 3
	a1 = (10*tau3+2*texp ** 3-5*tau*texp ** 2)/tau3
	a2 = (5*tau3+texp3-5*tau*tau*texp)/tau3
	a3 = (9*texp ** 5*period-40*tau3*texp*texp*period+120*tau ** 4*texp*(3*period-2*tau))/tau ** 6
	a4 = (a3*tau ** 5+texp ** 4*(54*tau-35*period)-12*tau*texp3*(4*tau+period)+360*tau ** 4*(tau-period))/tau ** 5
	a5 = (a2*(24*T*T*(texp-3*tau)-24*T*period*(texp-3*tau))+tau3*a4)/tau3
	a6 = (3*tau*tau+T*(texp-3*tau))/(tau*tau)
	a7 = (-60*tau ** 4+12*a2*tau3*T-9*texp ** 4+8*tau*texp3+40*tau3*texp)/(tau ** 4)
	a8 = (2*T-period)/tau
	a9 = (-3*tau*tau*texp*(-10*T*T+10*T*period+texp*(2*texp+5*period))-texp ** 4*period+8*tau*texp3*period)/(tau ** 5)
	a10 = ((a9+60)*tau*tau+10*(-9*T*T+9*T*period+texp*(3*texp+period))-75*tau*period)/(tau*tau)
	a11 = (texp*period-3*tau*(period-2*tau))/(tau*tau)
	a12 = (-360*tau ** 5-24*a2*tau3*T*(texp-3*tau)+9*texp ** 5-35*tau*texp ** 4-12*tau*tau*texp3-40*tau3*texp*texp+360*tau ** 4*texp)/(tau ** 5)
	a13 = (-3*texp3*(8*T*T-8*T*period+3*texp*period)+120*tau*tau*T*texp*(T-period)+8*tau*texp3*period)/tau ** 5
	a14 = (a13*tau*tau+40*(-3*T*T+3*T*period+texp*period)-60*tau*period)/(tau*tau)
	a15 = (2*texp-6*tau)/tau
	b1  = (6*texp*texp-3*texp*period+tau*period)/(texp*texp)
	b2  = (tau*T+3*texp*(texp-T))/(texp*texp)
	b3 = (tau3-12*T*texp*texp+8*texp3+20*tau*texp*texp-8*tau*tau*texp)/texp3
	b4 = (6*T*T-6*T*period+texp*(5*period-4*texp))/(texp*texp)
	b5 = (10*texp-3*tau)/texp
	b6 = (12*b4*texp3+4*tau*(-6*T*T+6*T*period+texp*(13*period-30*texp)))/texp3
	b7 = (b6*texp ** 5+4*tau*tau*texp*texp*(12*texp-11*period)+tau3*texp*(11*period-6*texp)-tau ** 4*period)/texp ** 5
	b8 = (3*T*T-3*T*period+texp*period)/(texp*texp)
	b9 = (8*b8*texp ** 4+20*tau*texp*texp*period-8*tau*tau*texp*period+tau3*period)/texp ** 4
	b10 =  (-tau ** 4+24*T*texp*texp*(tau-3*texp)+60*texp ** 4+52*tau*texp3-44*tau*tau*texp*texp+11*tau3*texp)/texp ** 4
	b11 =  (-15*b4*texp3+10*b8*tau*texp*texp+15*tau*tau*(2*texp-period))/texp3
	b12 =  (b11*texp ** 5+2*tau3*texp*(4*period-3*texp)-tau ** 4*period)/texp ** 5
	b13 =  (period-2*T)/texp
	b14 =  (6*texp-2*tau)/texp

	Q = snr/np.sqrt(num_tr)
	sigma_t0 = np.sqrt(0.5*tau*T/(1-texp/(3*tau)))/Q if tau >= texp else np.sqrt(0.5*texp*T/(1-tau/(3*texp)))/Q
	sigma_period = sigma_t0/np.sqrt(num_tr)
	sigma_duration = sigma*np.sqrt(abs(6*tau*a14/(delta*delta*a5)) / gamma) if tau>=texp else sigma*np.sqrt(abs(6*texp*b9/(delta*delta*b7)) / gamma)
	sigma_depth = sigma*np.sqrt(abs(-24*a11*a2/(tau*a5)) / gamma) if tau>=texp else sigma*np.sqrt(abs(24*b1/(texp*b7)) / gamma)

	if diagonal:     # Assume uncertainties uncorrelated (Diagonal)
		return np.diag([sigma_t0, sigma_period, sigma_duration, sigma_depth])
	else:
		warnings.warn("this is currently not the same covariance as when diagonal=True: covariance is of {tau, T, delta, f0}")
		cov = np.zeros(4,4)
		if tau>=texp:
			cov[0,0] = 24*tau*a10/(delta*delta*a5)
			cov[0,1] = 36*a8*tau*a1/(delta*delta*a5)
			cov[1,0] = cov[0,1]
			cov[0,2] = -12*a11*a1/(delta*a5)
			cov[2,0] = cov[0,2]
			cov[0,3] = -12*a6*a1/(delta*a5)
			cov[3,0] = cov[0,3]
			cov[1,1] = 6*tau*a14/(delta*delta*a5)
			cov[1,2] = 72*a8*a2/(delta*a5)
			cov[2,1] = cov[1,2]
			cov[1,3] = 6*a7/(delta*a5)
			cov[3,1] = cov[1,3]
			cov[2,2] = -24*a11*a2/(tau*a5)
			cov[2,3] = -24*a6*a2/(tau*a5)
			cov[3,2] = cov[2,3]
			cov[3,3] = a12/(tau*a5)
		else:
			cov[0,0] = -24*texp*texp*b12/(delta*delta*tau*b7)
			cov[0,1] = 36*texp*b13*b5/(delta*delta*b7)
			cov[1,0] = cov[0,1]
			cov[0,2] = 12*b5*b1/(delta*b7)
			cov[0,2] = cov[2,0]
			cov[0,3] = 12*b5*b2/(delta*b7)
			cov[3,0] = cov[0,3]
			cov[1,1] = 6*texp*b9/(delta*delta*b7)
			cov[1,2] = 72*b13/(delta*b7)
			cov[2,1] = cov[1,2]
			cov[1,3] = 6*b3/(delta*b7)
			cov[3,1] = cov[1,3]
			cov[2,2] = 24*b1/(texp*b7)
			cov[2,3] = 24*b2/(texp*b7)
			cov[3,2] = cov[3,2]
			cov[3,3] = b10/(texp*b7)
		cov *= sigma*sigma/gamma
		cov = make_pd(cov)

	return cov

if __name__ == "__main__":
	# for the sake of making this quick demo independent of the other files, I'll repeat a couple of functions
	import os
	import pandas as pd
	import scipy.stats as stats
	import sys
	sys.path.append('..')

	re = 0.009158

	def get_a(period, mstar, Go4pi=2945.4625385377644/(4*np.pi*np.pi)):
		return (Go4pi*period*period*mstar) ** (1./3)

	kois = get_catalog('q1_q16_koi')
	stellar = get_catalog('q1_q16_stellar')
	kois = kois[kois["kepid"].isin(stellar["kepid"])]
	kois = kois[np.isfinite(kois["koi_prad"])]
	stellar = stellar[np.isfinite(stellar.mass)]
	combined = pd.merge(kois, stellar, on='kepid')

	stellar_keys = stellar.keys()
	cdpp_cols = [k for k in stellar_keys if k.startswith("rrmscdpp")]
	cdpp_vals = np.array([k[-4:].replace("p", ".") for k in cdpp_cols], dtype=float)
	mesthres_cols = [k for k in stellar_keys if k.startswith("mesthres")]
	mesthres_vals = np.array([k[-4:].replace("p", ".") for k in mesthres_cols], dtype=float)

	i = 0 # index of the KOI in question
	p = combined['koi_period'][i]
	r = combined['koi_prad'][i]
	rstar = combined['radius'][i]
	ror = r / rstar * re
	d = ror ** 2
	b = combined['koi_impact'][i]
	num_tr = combined['dataspan'][i] * combined['dutycycle'][i] / p
	
	cdpp = np.array(stellar[cdpp_cols], dtype=float)[i]

	cosincl = np.cos(combined['koi_incl'][i] * np.pi / 180)
	ecc = stats.rayleigh(scale=0.03).rvs() * 0
	aor = get_a(p, combined['mass'][i]) / rstar
	
	tau0 = p * b / (2 * np.pi * cosincl * np.sqrt(1 - ecc ** 2)) * 1 / (aor ** 2)
	T = 2 * tau0 * np.sqrt(1 - b ** 2)
	D = (p / np.pi) * np.arcsin(np.sqrt((1 + p / rstar) ** 2 - b ** 2) / aor)
	tau0 = p * b / (2 * np.pi * cosincl * np.sqrt(1 - ecc ** 2)) * 1 / (aor ** 2)
	T = 2 * tau0 * np.sqrt(1 - b ** 2)
	tau = 2 * tau0 * r / np.sqrt(1 - b ** 2)
	snr = get_snr(r, rstar, cdpp, period=p, ecc=ecc, aor=aor)

	sigma = 1 # 'model uncertainty'
	price_uncertainty_params = [d, T, tau, p, num_tr, snr, sigma]
	print(make_cov(*price_uncertainty_params, diagonal=True))

'''
Constants, and simple debugging utilities.
'''

from numpy import nanmean, isnan, pi
from scipy.stats import gamma

Go4pi = 2945.4625385377644/(4 * pi * pi)
re = 0.009171
c = 1.0874
s = 1.0187
pgam = gamma(4.65, loc=0., scale=0.98)
kepler_exp_time_internal =  6.019802903/(24*60*60)    # https://archive.stsci.edu/kepler/manuals/archive_manual.pdf
kepler_read_time_internal = 0.5189485261/(24*60*60)

num_exposures_per_LC = 270
num_exposures_per_SC = 9
kepler_texp = kepler_exp_time_internal * num_exposures_per_LC
LC_duration = kepler_texp + kepler_read_time_internal * num_exposures_per_LC
LC_rate = 1. / LC_duration

eps = 1e-6 # small number for convenience to avoid div-by-zeros and other numerical problems.

cdpp_cols = ['rrmscdpp01p5', 'rrmscdpp02p0', 'rrmscdpp02p5', 'rrmscdpp03p0', 'rrmscdpp03p5', 'rrmscdpp04p5', 'rrmscdpp05p0', 'rrmscdpp06p0', 'rrmscdpp07p5', 'rrmscdpp09p0', 'rrmscdpp10p5', 'rrmscdpp12p0', 'rrmscdpp12p5', 'rrmscdpp15p0']
cdpp_vals = [1.5, 2., 2.5, 3., 3.5, 4.5, 5., 6., 7.5, 9., 10.5, 12., 12.5, 15.]
mesthres_cols = ['mesthres01p5', 'mesthres02p0', 'mesthres02p5', 'mesthres03p0', 'mesthres03p5', 'mesthres04p5', 'mesthres05p0', 'mesthres06p0', 'mesthres07p5', 'mesthres09p0', 'mesthres10p5', 'mesthres12p0', 'mesthres12p5', 'mesthres15p0']
mesthres_vals = [1.5, 2., 2.5, 3., 3.5, 4.5, 5., 6., 7.5, 9., 10.5, 12., 12.5, 15.]

# debugging utility
def minmeanmax(statistic, name):
    print("{0} min-mean-max-hasnan".format(name), min(statistic), nanmean(statistic), max(statistic), any(isnan(statistic)))

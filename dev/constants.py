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


# debugging utility
def minmeanmax(statistic, name):
    print("{0} min-mean-max-hasnan".format(name), min(statistic), nanmean(statistic), max(statistic), any(isnan(statistic)))

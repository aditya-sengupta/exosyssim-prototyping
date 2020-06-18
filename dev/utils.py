'''
Pure utility functions.
'''

from numpy import nanmean, isnan

# debugging utility
def minmeanmax(statistic, name):
    print("{0} min-mean-max-hasnan".format(name), min(statistic), nanmean(statistic), max(statistic), any(isnan(statistic)))


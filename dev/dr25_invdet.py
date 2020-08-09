import numpy as np
from matplotlib import pyplot as plt
import sys
import os
sys.path.append("..")

invdets = open("../data/dr25_fgk_invdet.txt").read().split('\n')
period_bins = [0.1, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0, 500.0]
radius_bins = [0.1, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0, 4.0, 6.0, 8.0, 12.0, 16.0]

occurrence = []
occ_err_pl = []
occ_err_mn = []

for entry in invdets:
    if '+' in entry:
        occurrence.append(float(entry[:entry.index('+')]))
        occ_err_pl.append(float(entry[entry.index('+') + 1:entry.index('-')]))
        occ_err_mn.append(float(entry[entry.index('-') + 1:-1]))

occurrence = np.array(occurrence[::-1]).reshape((len(period_bins)-1, len(radius_bins)-1))
plt.imshow(occurrence, origin='lower')
plt.colorbar()
_ = plt.xticks(list(range(len(radius_bins)-1)), radius_bins[:-1])
plt.xlabel(r"Radius ($R_E$)")
_ = plt.yticks(list(range(len(period_bins)-1)), period_bins[:-1])
plt.ylabel("Period (days)")
plt.show()
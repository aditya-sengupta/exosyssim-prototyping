# from Danley Hsu

import pylab as plt
#import matplotlib.colors as colors
#import matplotlib.cm as cm
#import matplotlib.ticker
import numpy as np
import math
import pandas as pd
from scipy import stats, special
from statistics import median, mean
from scipy import optimize

gamma_coeff = [[27.8185, 0.32432, 0.945075],
               [32.6448, 0.268898, 0.889724],
               [31.6342, 0.279425, 0.886144],
               [30.1906, 0.294688, 0.875042],
               [30.9919, 0.286979, 0.859865],
               [31.5196, 0.282741, 0.833673],
               [32.886, 0.269577, 0.768366],
               [33.3884, 0.264472, 0.699093]]

ntr_lim = [36, 18, 9, 6, 5, 4, 3]

gamma_ind = 0
min_mes = 5
max_mes = 20
num_bin = 10

def gamma_cdf(x, a, b, c):
    return c*stats.gamma.cdf(x, a, scale=b)

inj_in = pd.read_csv('kplr_dr25_inj1_input_mdwarf.csv', sep=',')
inj_out = pd.read_csv('kplr_dr25_inj1_robovetter_output_clean.txt', delim_whitespace=True)
inj_out['kepid'] = list(map(int, inj_out.TCE_ID.str.slice(0,9)))

inj_df = pd.merge(inj_in, inj_out, how='left', on='kepid')

if gamma_ind == 0:
    inj_sub = inj_df[inj_df.ntr > ntr_lim[0]]
elif gamma_ind == 7:
    inj_sub = inj_df[inj_df.ntr <= ntr_lim[6]]
else:
    inj_sub = inj_df[(inj_df.ntr > ntr_lim[gamma_ind]) & (inj_df.ntr <= ntr_lim[gamma_ind-1])]
#inj_sub = inj_df

inj_sub = inj_sub.sort_values('Expected_MES')
num_per_bin = int(len(inj_sub)/num_bin)
inj_x_arr = []
inj_y_arr = []
inj_yerrlow_arr = []
inj_yerrup_arr = []

print("# per bin = ", num_per_bin)
for i in range(num_bin):
    inj_mes_sub = inj_sub.iloc[i*num_per_bin:(i+1)*num_per_bin]
    inj_x_arr.append(median(inj_mes_sub.Expected_MES))
    k = len(inj_mes_sub[inj_mes_sub.Disposition == 'PC'])
    n = len(inj_mes_sub)
    #print(k, n)
    p = k/n
    inj_y_arr.append(p)
    def wilson(z):
        return (p + z*z/(2*n) + z*np.sqrt(p*(1-p)/n + (z*z)/(4*n*n)))/(1+(z*z/n))
    # def bin_p_low(p):
    #     return np.log(stats.beta.cdf(p, k+1, n-k+1))-np.log(0.16)
    # def bin_p_up(p):
    #     return np.log(stats.beta.cdf(p, k+1, n-k+1))-np.log(0.84)
    #yerrlow = optimize.root(bin_p_low, 0.5, method='lm')
    #yerrup = optimize.root(bin_p_up, 0.5, method='lm')

    inj_yerrlow_arr.append(p - wilson(-1))
    inj_yerrup_arr.append(wilson(1) - p)

a = gamma_coeff[gamma_ind][0]
b = gamma_coeff[gamma_ind][1]
c = gamma_coeff[gamma_ind][2]
gamma_x_arr = np.linspace(min_mes, max_mes, 1000)
fit_arr, fit_cov = optimize.curve_fit(gamma_cdf, inj_x_arr, inj_y_arr, [a, b, c])
#gamma_y_arr = np.array([gamma_cdf(x, a, b, c) for x in gamma_x_arr])
gamma_y_arr = np.array([gamma_cdf(x, fit_arr[0],fit_arr[1],fit_arr[2]) for x in gamma_x_arr])
print("FGK = ", [a, b, c], " / M = ", fit_arr)
    
fig = plt.figure()
ax = plt.axes()

ax.plot(gamma_x_arr, gamma_y_arr, label="FGK Gamma Detection Efficiency", color="black")
ax.errorbar(inj_x_arr, inj_y_arr, yerr=[inj_yerrlow_arr, inj_yerrup_arr], label="M Injection Tests", markeredgewidth=1, capsize=5, color="red")

plt.xlabel("Expected MES")
plt.ylabel("Detection Probability")
plt.ylim((0, 1))
plt.legend()
plt.tight_layout()
plt.savefig("inj_det_curve.png")
plt.savefig("inj_det_curve.eps")
plt.show()

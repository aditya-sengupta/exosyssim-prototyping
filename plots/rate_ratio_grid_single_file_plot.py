# from Danley Hsu

import pylab as plt
import matplotlib.colors as colors
import matplotlib.cm as cm
import numpy as np
import math
import copy

limit_type = "abc"
calc_type = ["abc","abc"]
noplan_file = ["dr25_gaia_fgk_mgrid", "dr25_gaia_m"]
type_label = ['FGK','M']

in_file = ["abc_rates/dr25_gaia_mfgk_ratio_dirichlet"]
out_file = "dr25_gaia_m_fgk_ratio_dirichlet"
out_eps = True
color_lim = 12

if limit_type == "abc":
    # Christiansen limits
    # limitP = np.array([0.5, 1.25, 2.5, 5., 10., 20., 40., 80., 160., 320.])
    # limitR = np.array([0.5, 0.75, 1., 1.25, 1.5, 1.75, 2., 2.5, 3., 4., 6., 8., 12., 16.])
    limitP = np.array([0.5, 1., 2., 4., 8., 16., 32., 64., 128., 256.])#, 500.])
    #limitP = np.array([2., 4., 8., 16., 32., 64., 128., 256., 500.])
    #limitR = np.array([0.5, 0.75, 1., 1.25, 1.5, 1.75, 2., 2.5, 3., 4., 6., 8., 12., 16.])
    limitR = np.array([0.5, 1., 1.5, 2., 2.5, 3., 4.])

elif limit_type == "sag":
    # SAG13 limits
    limitP = np.array([10., 20., 40., 80., 160., 320., 640.])
    limitR = np.array([1./1.5, 1., 1.5, 1.5**2, 1.5**3, 1.5**4, 1.5**5, 1.5**6, 1.5**7])

with open("./noplan_bins/"+noplan_file[0]+".txt") as f:
    na_bins = [int(x) for x in f.readline().split(',')]
if len(noplan_file) == 2:
    with open("./noplan_bins/"+noplan_file[1]+".txt") as f:
        noplan_bins = [int(x) for x in f.readline().split(',')]
else:
    noplan_bins = na_bins
        
in_file = open("./data/"+in_file[0]+".txt", "r")

data_array = np.zeros((len(limitR)-1,len(limitP)-1))
errup_array = np.zeros((len(limitR)-1,len(limitP)-1))
errdn_array = np.zeros((len(limitR)-1,len(limitP)-1))
color_array = np.zeros((len(limitR)-1,len(limitP)-1))

for i in range(0,len(limitP)-1):
    for j in range(0,len(limitR)-1):
        lin_arr = in_file.readline().split()
        bin_mean = float(lin_arr[0])
        bin_1quart = float(lin_arr[0])-float(lin_arr[4])
        bin_3quart = float(lin_arr[2])+float(lin_arr[0])
        if (i*(len(limitR)-1)+j+1) in na_bins:
            data_array[j,i] = None
            errup_array[j,i] = None
            errdn_array[j,i] = None
            color_array[j,i] = None
        elif (i*(len(limitR)-1)+j+1) in noplan_bins:
            data_array[j,i] = bin_3quart
            errup_array[j,i] = None
            errdn_array[j,i] = None 
            color_array[j,i] = None
        else:
            data_array[j,i] = bin_mean
            errup_array[j,i] = float(lin_arr[2])
            errdn_array[j,i] = float(lin_arr[4])
            color_array[j,i] = bin_mean

print(np.amin(color_array[~np.isnan(color_array)]), np.amax(color_array[~np.isnan(color_array)]))
color_array = np.ma.masked_invalid(color_array)            
fig = plt.figure()
fig.set_size_inches(16.5,8.5)
#ax = fig.add_subplot(111)

#temp_cm = copy.copy(plt.cm.rainbow)
temp_cm = copy.copy(plt.cm.plasma)
temp_cm.set_over('white')
temp_cm.set_under('blue')
temp_cm.set_bad('0.7')
tab_heatmap = plt.pcolormesh(color_array, cmap=temp_cm, edgecolors='black', linewidths=1, vmin=-color_lim, vmax=color_lim)
tab_plot = plt.colorbar(tab_heatmap)
tab_plot.set_label('$f_{\mathrm{'+type_label[1]+'}}/f_{\mathrm{'+type_label[0]+'}}$', rotation=270, fontsize=24, labelpad=35)
tab_plot.ax.tick_params(labelsize=18, length=10, width=2) 

if limit_type=="abc":
    p_lim_range = data_array.shape[1]
elif limit_type=="sag":
    p_lim_range = data_array.shape[1]-1
    
for i in range(p_lim_range):
    for j in range(data_array.shape[0]):
        if ((i*(len(limitR)-1)+j+1) in na_bins):# or (data_array[j,i] is np.ma.masked):
            plt.text(i + 0.5, j + 0.5, 'N/A',
                     horizontalalignment='center',
                     verticalalignment='center', fontsize=17)
        elif ((i*(len(limitR)-1)+j+1) in noplan_bins):
            if data_array[j,i] >= 100:
                 plt.text(i + 0.5, j + 0.5, '<%3d' %
                     (round(data_array[j,i])),
                     horizontalalignment='center',
                     verticalalignment='center', fontsize=17)
            else:
                plt.text(i + 0.5, j + 0.5, '<%.2g' %
                         (data_array[j,i]),
                         horizontalalignment='center',
                         verticalalignment='center', fontsize=17)
        else:
            plt.text(i + 0.5, j + 0.5, '$%.2g^{+%.2g}_{-%.2g}$' %
                     (data_array[j,i],errup_array[j,i],errdn_array[j,i]),
                     horizontalalignment='center',
                     verticalalignment='center', fontsize=17)

plt.ylabel("Planet Radius (R$_{\oplus}$)", fontsize=24)
plt.xlabel("Orbital Period (day)", fontsize=24)

plt.ylim(0, len(limitR)-1)
if limit_type == "sag":
    plt.xlim(0, len(limitP)-2)

x = range(len(limitP))
y = range(len(limitR))
xlabels = [str(i) for i in limitP]
ylabels = [str(i) for i in limitR]
# if limit_type == "abc":
#     # x = [0,1,2,3,4,5,6,7,8,9]
#     # y = [0,1,2,3,4,5,6,7,8,9,10,11,12,13]
#     # xlabels = ['0.5', '1.25', '2.5', '5', '10', '20', '40', '80', '160', '320']
#     # ylabels = ['0.5', '0.75', '1', '1.25', '1.5', '1.75', '2', '2.5', '3', '4', '6', '8', '12', '16']
#     x = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]#, 10]
#     y = [0, 1, 2, 3, 4, 5, 6]
#     xlabels = ['0.5', '1', '2', '4', '8', '16', '32', '64', '128', '256']#, '500']
#     #ylabels = ['0.5', '0.75', '1', '1.25', '1.5', '1.75', '2', '2.5', '3', '4', '6', '8', '12', '16']
#     ylabels = ['0.5', '1', '1.5', '2', '2.5', '3', '4']
# elif limit_type == "sag":
#     x = [0,1,2,3,4,5]
#     y = [0,1,2,3,4,5,6,7,8]
#     xlabels = ['10', '20', '40', '80', '160', '320']
#     ylabels = ['0.67', '1', '1.5', '2.3', '3.4', '5.1', '7.6', '11', '17']
    
plt.xticks(x, xlabels)
plt.yticks(y, ylabels)
plt.tick_params(labelsize=20)

plt.tight_layout()
if out_eps:
    plt.savefig("./plots/"+out_file+".eps")
plt.savefig("./plots/"+out_file+".png")
plt.show()

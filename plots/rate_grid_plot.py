# from Danley Hsu

import pylab as plt
import matplotlib.colors as colors
import matplotlib.cm as cm
import matplotlib.ticker
import numpy as np
import math
import copy

limit_type = "abc"
calc_type = "abc"
#noplan_file = "dr25_report"
noplan_file = "dr25_gaia_m"

in_file = "abc_rates/dr25_gaia_m_fgkvet_uniform"
out_file = "dr25_gaia_m_fgkvet_uniform"
#in_file = "abc_rates/dr25_report_baseline_vet"
#out_file = "dr25report_baseline_m_empty"#vet_uniform_mcmc"
integ_gap_plot = False#True
out_eps = True
empty_grid = False

colormax = 15.
np.set_printoptions(precision=2)

if limit_type == "abc":
    # Christiansen limits
#    limitP = np.array([0.5, 1.25, 2.5, 5., 10., 20., 40., 80., 160., 320.])
#    if qtr_num == "q1q17dr25":
#        limitP = np.append(limitP, 500.)
#    limitR = np.array([0.5, 0.75, 1., 1.25, 1.5, 1.75, 2., 2.5, 3., 4., 6., 8., 12., 16.])
    limitP = np.array([0.5, 1., 2., 4., 8., 16., 32., 64., 128., 256.])#, 500.])
    #limitR = np.array([0.5, 0.75, 1., 1.25, 1.5, 1.75, 2., 2.5, 3., 4., 6., 8., 12., 16.])
    limitR = np.array([0.5, 1., 1.5, 2., 2.5, 3., 4.])
elif limit_type == "sag":
    # SAG13 limits
    limitP = np.array([10., 20., 40., 80., 160., 320., 640.])
    limitR = np.array([1./1.5, 1., 1.5, 1.5**2, 1.5**3, 1.5**4, 1.5**5, 1.5**6, 1.5**7])

with open("./noplan_bins/"+noplan_file+".txt") as f:
    noplan_bins = [int(x) for x in f.readline().split(',')]

table_file = open("./data/"+in_file+".txt", "r")

data_array = np.zeros((len(limitR)-1,len(limitP)-1))
color_array = np.zeros((len(limitR)-1,len(limitP)-1))
error_array = np.zeros((len(limitR)-1, len(limitP)-1))
error_up = np.zeros((len(limitR)-1, len(limitP)-1))
error_down = np.zeros((len(limitR)-1, len(limitP)-1))

for i in range(0,len(limitP)-1):
    for j in range(0,len(limitR)-1):
        lin_arr = table_file.readline().split()

        if calc_type == "invdet":
            if (lin_arr[0] == "Inf") or (lin_arr[2] == "N/A") or ((i*(len(limitR)-1)+j+1) in noplan_bins):
                data_array[j,i] = None
                color_array[j,i] = None
                error_array[j,i] = None
                print("Individual-Period:{0}/Radius:{1}= N/A".format(limitP[i:i+2], limitR[j:j+2]))
            else:
                data_array[j,i] = float(lin_arr[0])
                color_array[j,i] = float(lin_arr[0])/100./(np.log(limitP[i+1])-np.log(limitP[i]))/(np.log(limitR[j+1])-np.log(limitR[j]))
                error_array[j,i] = float(lin_arr[2])
                print("Individual-Period:{0}/Radius:{1}= {2} +/- {3}".format(limitP[i:i+2], limitR[j:j+2], data_array[j,i]/100., error_array[j,i]/100.))
        else:
            bin_mean = float(lin_arr[0])
            bin_1quart = float(lin_arr[0])-float(lin_arr[4])
            bin_3quart = float(lin_arr[2])+float(lin_arr[0])
            error_up[j,i] = bin_3quart-bin_mean
            error_down[j,i] = bin_mean-bin_1quart
            if calc_type == "abc":
                bin_mean *= 100.
                bin_1quart *= 100.
                bin_3quart *= 100.
            print("Individual-Period:{0}/Radius:{1}= 16% / Mean / 84%: -{2} / {3} / +{4}".format(limitP[i:i+2], limitR[j:j+2], (bin_mean-bin_1quart)/100., bin_mean/100., (bin_3quart-bin_mean)/100.))

            if (i*(len(limitR)-1)+j+1) in noplan_bins:
                data_array[j,i] = bin_3quart
                color_array[j,i] = None
                error_array[j,i] = bin_3quart-bin_mean
            else:
                data_array[j,i] = bin_mean
                color_array[j,i] = bin_mean/100./(np.log(limitP[i+1])-np.log(limitP[i]))/(np.log(limitR[j+1])-np.log(limitR[j]))
                error_array[j,i] = max(bin_mean-bin_1quart, bin_3quart-bin_mean)

#limit_type = "gap"

color_array = np.ma.masked_invalid(color_array)
print("Min = ", np.amin(color_array), " / Max = ", np.amax(color_array))
fig = plt.figure()
color_min = 0.01#np.amin(color_array)
color_max = 3.0#np.amax(color_array)
fig.set_size_inches(16.5,8.5)
#ax = fig.add_subplot(111)

#temp_cm = copy.copy(plt.cm.rainbow)
temp_cm = copy.copy(plt.cm.plasma)
temp_cm.set_over('orange')
temp_cm.set_bad('0.7')
if empty_grid:
    color_array = np.zeros((len(limitR)-1,len(limitP)-1))
    tab_plot = plt.pcolor(color_array, cmap=plt.cm.binary, edgecolors='black', linewidths=1, vmin=0.0, vmax=1.0)
else:
    if limit_type == "gap":
        tab_heatmap = plt.pcolormesh(color_array[:10,-5:], cmap=temp_cm, vmax=min(np.amax(color_array[:10,-5:]), colormax), edgecolors='black', linewidths=1)
    else:
        tab_heatmap = plt.pcolormesh(color_array, cmap=temp_cm, norm=colors.LogNorm(vmin=color_min, vmax=color_max), edgecolors='black', linewidths=1)
    tab_plot = plt.colorbar(tab_heatmap, pad=0.02)
    tab_plot.set_label('$\\frac{d^{2}f_{\\mathrm{M}}}{d ( \ln{R_p} ) d (\ln{P})}$', rotation=270, fontsize=25, labelpad=55)
    tab_plot.ax.tick_params(which='minor', width=2, length=6)
    #tab_plot.set_label('$\mathrm{Frequency}\//\/[(d\log{R_p})(d\log{P})]$', rotation=270, fontsize=24, labelpad=35)
    tab_plot.ax.tick_params(labelsize=18, length=10, width=2) 

# if limit_type=="abc" or limit_type == "poisson":
#    p_lim_range = data_array.shape[1]
#elif limit_type=="sag":
#    p_lim_range = data_array.shape[1]-1
#else:
p_lim_range = data_array.shape[1]

if not empty_grid:
    for i in range(p_lim_range):
        for j in range(data_array.shape[0]):
            if calc_type == "invdet":
                if ((i*(len(limitR)-1)+j+1) in noplan_bins) or (np.isnan(data_array[j,i])):
                    plt.text(i + 0.5, j + 0.5, 'N/A',
                             horizontalalignment='center',
                             verticalalignment='center', fontsize=17)
                else:
                    if np.floor(np.log(data_array[j,i])) == np.floor(np.log(error_array[j,i])):
                        plt.text(i + 0.5, j + 0.5, '%.2g + \n %.2g - \n %.2g %%' %
                                 (data_array[j,i], error_up[j,i]*100, error_down[j,i]*100),
                                 horizontalalignment='center',
                                 verticalalignment='center', fontsize=17)
                    else:
                        plt.text(i + 0.5, j + 0.5, '%.3g + \n %.2g - \n %.2g %%' %
                                 (data_array[j,i], error_up[j,i]*100, error_down[j,i]*100),
                                 horizontalalignment='center',
                                 verticalalignment='center', fontsize=17)
            else:
                if (i*(len(limitR)-1)+j+1) in noplan_bins:
                    if limit_type=="gap" and (i > 3 and j < 10):
                        if data_array[j,i] > 100.:
                            plt.text(i - 4 + 0.5, j + 0.5, '<%2d%%' %
                                     (data_array[j,i]),
                                     horizontalalignment='center',
                                     verticalalignment='center', fontsize=17, color='black')
                        else:
                            plt.text(i - 4 + 0.5, j + 0.5, '<%.2g%%' % data_array[j,i],
                                     horizontalalignment='center',
                                     verticalalignment='center', fontsize=17, color='black')
                    elif limit_type != "gap":
                        if data_array[j,i] > 100.:
                            plt.text(i + 0.5, j + 0.5, '<%2d%%' %
                                     (data_array[j,i]),
                                     horizontalalignment='center',
                                     verticalalignment='center', fontsize=17, color='black')
                        else:
                            plt.text(i + 0.5, j + 0.5, '<%.2g%%' % data_array[j,i],
                                     horizontalalignment='center',
                                     verticalalignment='center', fontsize=17, color='black')
                else:
                    round_pow = -1*int(np.floor(max(np.log10(data_array[j,i]), np.log10(error_up[j,i]*100),np.log10(error_down[j,i]*100))))+1
                    tmp_data = np.round(data_array[j,i], round_pow)
                    tmp_errorup = np.round(error_up[j,i]*100, round_pow)
                    tmp_errordn = np.round(error_down[j,i]*100, round_pow)
                    plt.text(i + 0.5, j + 0.5, '$%.2g^{+%.2g}_{-%.2g}$%%' %
                             (tmp_data, tmp_errorup, tmp_errordn),
                             horizontalalignment='center',
                             verticalalignment='center', fontsize=17, color='black')
                    
                    # if np.floor(np.log10(data_array[j,i])) == np.floor(np.log10(error_up[j,i]*100)) or np.floor(np.log10(data_array[j,i])) == np.floor(np.log10(error_down[j,i]*100)):
                    #     if limit_type == "gap" and (i > 3 and j < 10):
                    #         plt.text(i - 4 + 0.5, j + 0.5, '%.2g + \n %.2g - \n %.2g %%' %
                    #                  (data_array[j,i], error_up[j,i]*100, error_down[j,i]*100),
                    #                  horizontalalignment='center',
                    #                  verticalalignment='center', fontsize=17, color='black')
                    #     elif limit_type != "gap":
                    #         plt.text(i + 0.5, j + 0.5, '%.2g + \n %.2g - \n %.2g %%' %
                    #                  (data_array[j,i], error_up[j,i]*100, error_down[j,i]*100),
                    #                  horizontalalignment='center',
                    #                  verticalalignment='center', fontsize=17, color='black')
                    # else:
                    #     if limit_type == "gap" and (i > 3 and j < 10):
                    #         plt.text(i - 4 + 0.5, j + 0.5, '%.3g + \n %.2g - \n %.2g %%' %
                    #                  (data_array[j,i], error_up[j,i]*100, error_down[j,i]*100),
                    #                  horizontalalignment='center',
                    #                  verticalalignment='center', fontsize=17, color='black')
                    #     elif limit_type != "gap":
                    #         plt.text(i + 0.5, j + 0.5, '%.3g + \n %.2g - \n %.2g %%' %
                    #                  (data_array[j,i], error_up[j,i]*100, error_down[j,i]*100),
                    #                  horizontalalignment='center',
                    #                  verticalalignment='center', fontsize=17, color='black')
                  
plt.ylabel("Planet Radius (R$_{\oplus}$)", fontsize=24)
plt.xlabel("Orbital Period (day)", fontsize=24)

plt.ylim(0, len(limitR)-1)
#if limit_type == "sag":
#    plt.xlim(0, len(limitP)-2)
if limit_type == "gap":
    plt.ylim(0, 8)
    plt.xlim(0, 5)

x = list(range(len(limitP)))
y = list(range(len(limitR)))
xlabels = [str(x) for x in limitP]
ylabels = [str(x) for x in limitR]
    
# if limit_type == "abc":
#     # x = [0,1,2,3,4,5,6,7,8,9]
#     # y = [0,1,2,3,4,5,6,7,8,9,10,11,12,13]
#     # xlabels = ['0.5', '1.25', '2.5', '5', '10', '20', '40', '80', '160', '320']
#     # ylabels = ['0.5', '0.75', '1', '1.25', '1.5', '1.75', '2', '2.5', '3', '4', '6', '8', '12', '16']
    
#     x = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
#     y = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
    
#     xlabels = ['0.5', '1', '2', '4', '8', '16', '32', '64', '128', '256', '500']
#     ylabels = ['0.5', '0.75', '1', '1.25', '1.5', '1.75', '2', '2.5', '3', '4', '6', '8', '12', '16']
# elif limit_type == "sag":
#     x = [0,1,2,3,4,5,6]
#     y = [0,1,2,3,4,5,6,7,8]
#     xlabels = ['10', '20', '40', '80', '160', '320', '640']
#     ylabels = ['0.67', '1', '1.5', '2.3', '3.4', '5.1', '7.6', '11', '17']
# elif limit_type == "gap":
#     x = [0,1,2,3,4,5]
#     y = [0,1,2,3,4,5,6,7,8,9,10]
#     xlabels = ['10', '20', '40', '80', '160', '320']
#     ylabels = ['0.5', '0.75', '1', '1.25', '1.5', '1.75', '2', '2.5', '3', '4', '6']
    
plt.xticks(x, xlabels)
plt.yticks(y, ylabels)
plt.tick_params(labelsize=20)

plt.tight_layout()
if out_eps:
    plt.savefig("./plots/"+out_file+".eps")
plt.savefig("./plots/"+out_file+".png")
plt.show()

if integ_gap_plot:
    integ_array1 = np.zeros(len(limitR)-1)
    integ_array2 = np.zeros(len(limitR)-1)
    integ_array3 = np.zeros(len(limitR)-1)
    integ_rad = np.zeros(len(limitR)-1)
    integ_errup1 = np.zeros(len(limitR)-1)
    integ_errup2 = np.zeros(len(limitR)-1)
    integ_errdown1 = np.zeros(len(limitR)-1)
    integ_errdown2 = np.zeros(len(limitR)-1)
    # integ_err3 = np.zeros(len(limitR)-1)
    integ_errup3 = np.zeros(len(limitR)-1)
    integ_errdown3 = np.zeros(len(limitR)-1)
    for j in range(0, len(limitR)-1):
        integ_rad[j] = 2.0**((np.log2(limitR[j]) + np.log2(limitR[j+1]))/2.0)
        # for i in range(0, 4):
        #     if (i*(len(limitR)-1)+j+1) in noplan_bins:
        #         integ_array1[j] += (data_array[j,i]-error_array[j,i])
        #     else:
        #         integ_array1[j] += data_array[j,i]
        #     integ_errup1[j] += error_up[j,i]
        #     integ_errdown1[j] += error_down[j,i]
        # for i in range(5, 8):
        #     if (i*(len(limitR)-1)+j+1) in noplan_bins:
        #         integ_array2[j] += (data_array[j,i]-error_array[j,i])
        #     else:
        #         integ_array2[j] += data_array[j,i]
        #     integ_errup2[j] += error_up[j,i]
        #     integ_errdown2[j] += error_down[j,i]
        for i in range(0, 8):
            if (i*(len(limitR)-1)+j+1) in noplan_bins:
                integ_array3[j] += (data_array[j,i]-error_array[j,i])
            else:
                integ_array3[j] += data_array[j,i]
            integ_errup3[j] += error_up[j,i]
            integ_errdown3[j] += error_down[j,i]
        # print("{0},{1},{2},{3},{4}".format(limitR[j],limitR[j+1],integ_array3[j]/100., integ_errup3[j], integ_errdown3[j]))
        # for i in range(5, 8):
        #     if (i*(len(limitR)-1)+j+1) in noplan_bins:
        #         integ_array3[j] += (data_array[j,i]-error_array[j,i])
        #     else:
        #         integ_array3[j] += data_array[j,i]
        #     integ_err3[j] += error_array[j,i]

    integ_array1 = np.append(integ_array1, 0.)
    integ_array2 = np.append(integ_array2, 0.)
    integ_array3 = np.append(integ_array3, 0.)
        
    fig = plt.figure()
    fig.set_size_inches(11.5,8.5)
    ax = fig.add_subplot(111)
    
    bin_centers = 0.5*(limitR[1:]+limitR[:-1])
    # ax.step(limitR, integ_array1/100., color='k', where='post')
    # ax.errorbar(bin_centers, integ_array1[:-1]/100., yerr=np.vstack((integ_errdown1, integ_errup1)), fmt='o', color='k', capsize=8, label=r'$P=0.5-8$d')
    # ax.step(limitR, integ_array2/100., color='red', where='post')
    # ax.errorbar(bin_centers, integ_array2[:-1]/100., yerr=np.vstack((integ_errdown2, integ_errup2)), fmt='o', color='red', capsize=8, label=r'$P=16-128$d')
    ax.step(limitR, integ_array3/100., where='post', color='green')
    ax.errorbar(bin_centers, integ_array3[:-1]/100., yerr=np.vstack((integ_errdown3, integ_errup3)), fmt='o', color='green', capsize=8, label=r'$P=0.5-128$d')
        
    plt.xlabel("Planet Radius (R$_{\oplus}$)", fontsize=32)
    plt.ylabel("Occurrence Rate ($f$)", fontsize=32)
    #plt.ylabel('$\\frac{d^{2}f}{d ( \log_2{R_p} ) d (\log_2{P})}$', fontsize=24)

    #plt.xticks(x, xlabels)
    #plt.yticks(y, ylabels)
    ax.set_yscale('log', basey=10)
    ax.set_xscale('log')
    #ax.set_yscale('log')
    ax.set_xlim([min(limitR), max(limitR)])
    ax.set_ylim([1e-4, 3.0])
    #ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%d'))
    ax.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%d'))
    plt.tick_params(labelsize=28, width=3, length=12)
    plt.tick_params(which='minor', width=3, length=8, labelsize=24)
    plt.legend(fontsize=20)

    plt.tight_layout()
    if out_eps:
        plt.savefig("./plots/"+out_file+"_integ_gap.eps")
    plt.savefig("./plots/"+out_file+"_integ_gap.png")
    plt.show()

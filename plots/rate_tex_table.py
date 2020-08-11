# from Danley Hsu

import numpy as np
import math

calc_type = ["abc", "abc"]#, "abc"]

in_file = ["abc_rates/dr25_gaia_m_fgkvet_dirichlet", "abc_rates/dr25_gaia_m_fgkvet_uniform"]#, "abc_rates/dr25_report_baseline_mulders"]
noplan_file = ["dr25_gaia_m.txt", "dr25_gaia_m.txt"]#, "dr25_report.txt"]
ratio_in_file = ["abc_rates/dr25_gaia_mfgk_ratio_dirichlet","abc_rates/dr25_gaia_mfgk_ratio_uniform"]
ratio_noplan = ["dr25_gaia_fgk_mgrid.txt","dr25_gaia_fgk_mgrid.txt"]
planbin_file = ["dr25_gaia_m_planbin.txt"]
limit_type = "abc"

# in_file = ["abc_rates/dr25_report_exopag_typef", "abc_rates/dr25_report_exopag_typeg", "abc_rates/dr25_report_exopag_typek"]
# noplan_file = ["dr25_report_sag_typef.txt", "dr25_report_sag_typeg.txt", "dr25_report_sag_typek.txt"]
# limit_type = "sag"

out_file = "baseline_m_tex_table.txt"

np.set_printoptions(precision=2)

if limit_type == "abc":
    # Christiansen limits
    # limitP = np.array([0.5, 1.25, 2.5, 5., 10., 20., 40., 80., 160., 320.])
    # limitR = np.array([0.5, 0.75, 1., 1.25, 1.5, 1.75, 2., 2.5, 3., 4., 6., 8., 12., 16.])
    #limitP = np.array([0.5, 1., 2., 4., 8., 16., 32., 64., 128., 256., 500.])
    #limitR = np.array([0.5, 0.75, 1., 1.25, 1.5, 1.75, 2., 2.5, 3., 4., 6., 8., 12., 16.])
    limitP = np.array([0.5, 1., 2., 4., 8., 16., 32., 64., 128., 256.])
    limitR = np.array([0.5, 1., 1.5, 2., 2.5, 3., 4.])
elif limit_type == "sag":
    # SAG13 limits
    limitP = np.array([10., 20., 40., 80., 160., 320., 640.])
    limitR = np.array([1./1.5, 1., 1.5, 1.5**2, 1.5**3, 1.5**4, 1.5**5, 1.5**6, 1.5**7])

p_lim_range = len(limitP)-1
text_array = []

for i in range(0,len(limitP)-1):
    text_array.append([])
    for j in range(0,len(limitR)-1):
        text_array[i].append("$")
        
        if limitP[i] < 10:
            text_array[i][j] += "\\"+"phn"
        if limitP[i] < 100:
            text_array[i][j] += "\\"+"phn"
        text_array[i][j] += '{:5.2f}'.format(limitP[i]) + '-'
        if limitP[i+1] < 10:
            text_array[i][j] += "\\"+"phn"
        if limitP[i+1] < 100:
            text_array[i][j] += "\\"+"phn"
        text_array[i][j] += '{:5.2f}'.format(limitP[i+1]) + '$&$'

        if limitR[j] < 10:
            text_array[i][j] += "\\"+"phn"
        text_array[i][j] += '{:4.2f}'.format(limitR[j]) + '-'
        if limitR[j+1] < 10:
            text_array[i][j] += "\\"+"phn"
        text_array[i][j] += '{:4.2f}'.format(limitR[j+1]) + '$&'

if len(planbin_file) > 0:
    planbin_arr = [[] for x in range(len(limitR)-1)]
    with open(planbin_file[0],"r") as f:
        for j in range(len(limitR)-1):
            planbin_arr[j] = [int(x) for x in f.readline().split('\t')]
    planbin_arr = planbin_arr[::-1]
    for i in range(len(limitP)-1):
        for j in range(len(limitR)-1):
            text_array[i][j] += '$' + str(planbin_arr[j][i]) + '$&'

for k in range(0, len(calc_type)):
    table_file = open("./data/"+in_file[k]+".txt", "r")

    with open("./noplan_bins/"+noplan_file[k]) as f:
        noplan_bins = [int(x) for x in f.readline().split(',')]

    data_array = np.zeros((len(limitR)-1,len(limitP)-1))
    uperror_array = np.zeros((len(limitR)-1, len(limitP)-1))
    dnerror_array = np.zeros((len(limitR)-1, len(limitP)-1))

    if len(ratio_in_file) > 0:
        ratio_file = open("./data/"+ratio_in_file[k]+".txt", "r")
        ratio_array = np.zeros((len(limitR)-1,len(limitP)-1))
        ratio_uperror = np.zeros((len(limitR)-1,len(limitP)-1))
        ratio_dnerror = np.zeros((len(limitR)-1,len(limitP)-1))
        with open("./noplan_bins/"+ratio_noplan[k]) as f:
            ratio_noplanbin = [int(x) for x in f.readline().split(',')]

    for i in range(0,len(limitP)-1):
        for j in range(0,len(limitR)-1):
            if calc_type[k] == "abc":
                lin_arr = table_file.readline().split()
        
                bin_mean = float(lin_arr[0])
                bin_1quart = (float(lin_arr[0])-float(lin_arr[4]))
                bin_3quart = (float(lin_arr[2])+float(lin_arr[0]))

                if len(ratio_in_file) > 0:
                    ratio_lin = ratio_file.readline().split()
        
                    ratio_mean = float(ratio_lin[0])
                    ratio_1quart = (float(ratio_lin[0])-float(ratio_lin[4]))
                    ratio_3quart = (float(ratio_lin[2])+float(ratio_lin[0]))

                if (i*(len(limitR)-1)+j+1) in noplan_bins:
                    data_array[j,i] = bin_3quart
                    uperror_array[j,i] = None
                    dnerror_array[j,i] = None
                    if len(ratio_in_file) > 0:
                        if (i*(len(limitR)-1)+j+1) in ratio_noplanbin:
                            ratio_array[j,i] = None
                            ratio_uperror[j,i] = None
                            ratio_dnerror[j,i] = None
                        else:
                            ratio_array[j,i] = ratio_3quart
                            ratio_uperror[j,i] = None
                            ratio_dnerror[j,i] = None
                else:
                    data_array[j,i] = bin_mean
                    dnerror_array[j,i] = bin_mean-bin_1quart
                    uperror_array[j,i] = bin_3quart-bin_mean
                    if len(ratio_in_file) > 0:
                        if (i*(len(limitR)-1)+j+1) in ratio_noplanbin:
                            print(i*(len(limitR)-1)+j+1)
                            ratio_array[j,i] = None
                            ratio_uperror[j,i] = None
                            ratio_dnerror[j,i] = None
                        else:
                            ratio_array[j,i] = ratio_mean
                            ratio_dnerror[j,i] = ratio_mean-ratio_1quart
                            ratio_uperror[j,i] = ratio_3quart-ratio_mean

            elif calc_type[k] == "invdet":
                bin_data = table_file.readline().split(' ')
                if (bin_data[0] == "Inf") or (bin_data[2] == "N/A") or ((i*(len(limitR)-1)+j+1) in noplan_bins):
                    data_array[j,i] = None
                    uperror_array[j,i] = None
                    dnerror_array[j,i] = None
                else:
                    data_array[j,i] = float(bin_data[0])/100.
                    uperror_array[j,i] = float(bin_data[2])/100.
                    dnerror_array[j,i] = float(bin_data[2])/100.

            elif calc_type[k] == "poisson":
                bin_data = table_file.readline().split(' ')
                data_array[j,i] = float(bin_data[0])/100.
                uperror_array[j,i] = float(bin_data[2])/100.
                dnerror_array[j,i] = float(bin_data[4])/100.
                
    for i in range(p_lim_range):
        for j in range(data_array.shape[0]):
            tmp_pow = 0
            if data_array[j,i] != None:
                while data_array[j,i] < 1.:
                    data_array[j,i] *= 10.
                    uperror_array[j,i] *= 10.
                    dnerror_array[j,i] *= 10.
                    tmp_pow -= 1
                while data_array[j,i] > 10.:
                    data_array[j,i] /= 10.
                    uperror_array[j,i] /= 10.
                    dnerror_array[j,i] /= 10.
                    tmp_pow += 1

            if np.isnan(data_array[j,i]):
                text_array[i][j] += "N/A"
                    
            elif (i*(len(limitR)-1)+j+1) in noplan_bins:
                text_array[i][j] +=  '$<{:3.1f}'.format(data_array[j,i]) + "\\" + "times" + "10^{" + str(tmp_pow) + "}$"
            else:
                if calc_type[k] == "invdet":
                    if uperror_array[j,i] < 0.1:
                        text_array[i][j] +=  '$({:5.3f}'.format(data_array[j,i]) + "\\"+ "pm" + '{:5.3f})'.format(uperror_array[j,i])+ "\\" + "times" + "10^{" + str(tmp_pow) + "}$"
                    elif uperror_array[j,i] < 1.:
                        text_array[i][j] +=  '$({:4.2f}'.format(data_array[j,i]) + "\\" + "pm" + '{:4.2f})'.format(uperror_array[j,i])+ "\\" + "times" + "10^{" + str(tmp_pow) + "}$"
                    else:
                        text_array[i][j] +=  '$({:3.1f}'.format(data_array[j,i]) + "\\"+ "pm" + '{:3.1f})'.format(uperror_array[j,i])+ "\\" + "times" + "10^{" + str(tmp_pow) + "}$"
                else:
                    if uperror_array[j,i] < 0.1 or dnerror_array[j,i] < 0.1:
                        text_array[i][j] +=  '${:5.3f}'.format(data_array[j,i]) + "^{+"+'{:5.3f}'.format(uperror_array[j,i])+ "}_{-" + '{:5.3f}'.format(dnerror_array[j,i]) +"}\\" + "times" + "10^{" + str(tmp_pow) + "}$"
                    elif uperror_array[j,i] < 1. or dnerror_array[j,i] < 1.:
                        text_array[i][j] +=  '${:4.2f}'.format(data_array[j,i]) + "^{+"+'{:4.2f}'.format(uperror_array[j,i])+ "}_{-" + '{:4.2f}'.format(dnerror_array[j,i]) +"}\\" + "times" + "10^{" + str(tmp_pow) + "}$"
                    else:
                        text_array[i][j] +=  '${:3.1f}'.format(data_array[j,i]) + "^{+"+'{:3.1f}'.format(uperror_array[j,i])+ "}_{-" + '{:3.1f}'.format(dnerror_array[j,i]) +"}\\" + "times" + "10^{" + str(tmp_pow) + "}$"

            if len(ratio_in_file) > 0:
                text_array[i][j] += "&"
                # tmp_pow = 0
                # if ratio_array[j,i] != None:
                #     while ratio_array[j,i] < 1.:
                #         ratio_array[j,i] *= 10.
                #         ratio_uperror[j,i] *= 10.
                #         ratio_dnerror[j,i] *= 10.
                #         tmp_pow -= 1
                        
                #     while ratio_array[j,i] > 10.:
                #         ratio_array[j,i] /= 10.
                #         ratio_uperror[j,i] /= 10.
                #         ratio_dnerror[j,i] /= 10.
                #         tmp_pow += 1

                if np.isnan(ratio_array[j,i]):
                    text_array[i][j] += "N/A"
                    
                elif np.isnan(ratio_uperror[j,i]):
                    text_array[i][j] +=  '$<{:3.1f}'.format(ratio_array[j,i]) + "$"
                    # text_array[i][j] +=  '$<{:3.1f}'.format(ratio_array[j,i]) + "\\" + "times" + "10^{" + str(tmp_pow) + "}$"

                else:
                    # if ratio_uperror[j,i] < 0.1 or ratio_dnerror[j,i] < 0.1:
                    #     text_array[i][j] +=  '${:5.3f}'.format(ratio_array[j,i]) + "^{+"+'{:5.3f}'.format(ratio_uperror[j,i])+ "}_{-" + '{:5.3f}'.format(ratio_dnerror[j,i]) +"}$"
                    #     # text_array[i][j] +=  '${:5.3f}'.format(ratio_array[j,i]) + "^{+"+'{:5.3f}'.format(ratio_uperror[j,i])+ "}_{-" + '{:5.3f}'.format(ratio_dnerror[j,i]) +"}\\" + "times" + "10^{" + str(tmp_pow) + "}$"
                    # if ratio_uperror[j,i] < 1. or ratio_dnerror[j,i] < 1.:
                    #     text_array[i][j] +=  '${:4.2f}'.format(ratio_array[j,i]) + "^{+"+'{:4.2f}'.format(ratio_uperror[j,i])+ "}_{-" + '{:4.2f}'.format(ratio_dnerror[j,i]) +"}$"
                    #     # text_array[i][j] +=  '${:4.2f}'.format(ratio_array[j,i]) + "^{+"+'{:4.2f}'.format(ratio_uperror[j,i])+ "}_{-" + '{:4.2f}'.format(ratio_dnerror[j,i]) +"}\\" + "times" + "10^{" + str(tmp_pow) + "}$"
                    # else:
                    text_array[i][j] +=  '${:3.1f}'.format(ratio_array[j,i]) + "^{+"+'{:3.1f}'.format(ratio_uperror[j,i])+ "}_{-" + '{:3.1f}'.format(ratio_dnerror[j,i]) +"}$"
                        # text_array[i][j] +=  '${:3.1f}'.format(ratio_array[j,i]) + "^{+"+'{:3.1f}'.format(ratio_uperror[j,i])+ "}_{-" + '{:3.1f}'.format(ratio_dnerror[j,i]) +"}\\" + "times" + "10^{" + str(tmp_pow) + "}$"
            
            if k != len(calc_type)-1:
                text_array[i][j] += "&"

f_out = open(out_file, "w")

f_out.write("\\"+"startdata"+"\n")
for i in range(0,len(limitP)-1):
    for j in range(0,len(limitR)-1):
        f_out.write(text_array[i][j]+"\\"+"\\"+"\n")
    if i != len(limitP)-2:
        f_out.write("\\"+"hline"+"\n")
    else:
        f_out.write("\\"+"enddata")

f_out.close()

import os
from pathlib import Path
import csv

import numpy as np
import ROOT

from raw_viewer import ddas_interface
from e23035_analysis import fitting_tools, root_vis_tools, e23035_runs, degai
from e23035_analysis.spectrum_fitter import spectrum_fitter

runs = e23035_runs.get_ddas_60_Ga_runs()
def get_timing_hist(crystal1, crystal2):
    return ddas_interface.get_histogram(runs, (30000,-15e-6, 15e-6), f't{crystal1}_minus_t{crystal2}', f"crystal {crystal1} time - crystal {crystal2} time", 
                                        f"clover_{crystal1}_t-clover_{crystal2}_t", f"(clover_{crystal1}_m==1)&&(clover_{crystal2}_m==1)", num_workers=len(runs)) 

d = {}
for i in degai.clover_str_list:
    i = i.split('_')[1]
    #for j in degai.clover_str_list:
    j = '5a'#j.split('_')[1]
    if i != j:
        d[(i,j)] = get_timing_hist(i,j)

if False:
    import matplotlib.pyplot as plt
    image = np.zeros((len(degai.clover_str_list), len(degai.clover_str_list)))
    for i in range(len(degai.clover_str_list)):
        for j in range(len(degai.clover_str_list)):
            if i!= j:
                image[i,j] = d[(degai.clover_str_list[i].split('_')[1], degai.clover_str_list[j].split('_')[1])].GetMean()
    plt.imshow(image)
    plt.colorbar()
    plt.show()
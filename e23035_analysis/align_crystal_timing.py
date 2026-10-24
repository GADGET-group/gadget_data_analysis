import os
from pathlib import Path
import csv

import numpy as np
import ROOT

from tpc_and_decay_station import ddas_interface
from analysis_tools import fitting_tools, root_vis_tools
from e23035_analysis import e23035_runs
from tpc_and_decay_station import degai
from analysis_tools.spectrum_fitter import spectrum_fitter

experiment = 'e23035'
runs = e23035_runs.get_ddas_60_Ga_runs()
def get_timing_hist(crystal1, crystal2):
    return ddas_interface.get_histogram(experiment, runs, (30000,-15e-6, 15e-6), f't{crystal1}_minus_t{crystal2}', f"crystal {crystal1} time - crystal {crystal2} time", 
                                        f"clover_{crystal1}_t-clover_{crystal2}_t", f"(clover_{crystal1}_m==1)&&(clover_{crystal2}_m==1)", num_workers=len(runs))

d = {}
for i in degai.get_clover_str_list(experiment):
    i = i.split('_')[1]
    #for j in degai.get_clover_str_list(experiment):
    j = '5a'#j.split('_')[1]
    if i != j:
        d[(i,j)] = get_timing_hist(i,j)

if False:
    import matplotlib.pyplot as plt
    image = np.zeros((len(degai.get_clover_str_list(experiment)), len(degai.get_clover_str_list(experiment))))
    for i in range(len(degai.get_clover_str_list(experiment))):
        for j in range(len(degai.get_clover_str_list(experiment))):
            if i!= j:
                image[i,j] = d[(degai.get_clover_str_list(experiment)[i].split('_')[1], degai.get_clover_str_list(experiment)[j].split('_')[1])].GetMean()
    plt.imshow(image)
    plt.colorbar()
    plt.show()
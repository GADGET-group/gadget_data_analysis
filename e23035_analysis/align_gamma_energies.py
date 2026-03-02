import os

import numpy as np
import ROOT

from raw_viewer import ddas_interface
from e23035_analysis import fitting_tools, root_vis_tools, e23035_runs

'''
Notes on peak finding an automatic fitting
https://root.cern/root/htmldoc/guides/spectrum/Spectrum.html
'''
gamma_binning = (12000-1,1,12000)
run_candidates = e23035_runs.run_df['DDAS'][(e23035_runs.run_df['Run Type']=='60Ga')]
runs = []
for run in run_candidates:
    if not np.isnan(run) and run not in [162,163,203,204,209, 213,217, 218, 238]:
        if os.path.exists(ddas_interface.get_merged_root_file_path(run)):
            runs.append(run)
n_workers=min(200, len(runs))

def make_energy_calibration(ddas_run, branch_name, binning_for_fit, peak1, peak2=[], selection_string=''):
    '''
    Fit peaks to get energy calibraiton
    Each peak should be a list containing [true energy, guess for location, +/- bounds to use when fitting the peak]
    If peak2 is an empty list, offset will be assumed to be 0.
    '''
    hist_to_fit = ddas_interface.get_histogram(ddas_run, binning_for_fit, branch_name, branch_name, branch_name, selection_string)
    
def save_energy_calibration(run, calibration):
    pass

gammas = ddas_interface.get_histogram(runs, gamma_binning, 'gammas', 'summed gamma spectrum', ddas_interface.get_summed_gamma_e_str(), num_workers=n_workers)
crystal_hists = ddas_interface.get_crystal_histograms(runs, gamma_binning)
c,l = root_vis_tools.draw_overlaid_histograms(crystal_hists)
ROOT.gPad.SetLogy(1)

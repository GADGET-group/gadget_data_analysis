import os

import numpy as np
import ROOT

from raw_viewer import ddas_interface
from e23035_analysis import fitting_tools, root_vis_tools, e23035_runs, clarion

'''
Notes on peak finding an automatic fitting
https://root.cern/root/htmldoc/guides/spectrum/Spectrum.html
'''
gamma_bin_size = 1 #keV
gamma_binning = (int((7000-0)/gamma_bin_size),0,7000) #was 1-12000 w/ 1 keV bins
run_candidates = e23035_runs.run_df['DDAS'][(e23035_runs.run_df['Run Type']=='60Ga')]
runs = []
for run in run_candidates:
    if not np.isnan(run) and run not in [162,163,203,204,209, 213,217, 218, 238] and run not in [161, 173, 174, 237]: #second set of runs lack gm
        if os.path.exists(ddas_interface.get_merged_root_file_path(run)):
            runs.append(run)

adj_dict = clarion.get_adjacency_dict(30)

coinc_canvas = ROOT.TCanvas()
gg_hist = clarion.get_addback_coincidence_spectrum(runs, adj_dict, 'gm', (6000,0,6000))
gg_hist.Draw('COLZ')
ROOT.gPad.SetLogz(1)

ab_hist = clarion.get_addback_spectrum(runs, adj_dict, 'gm', (6000, 0, 6000))
ab_canvas = ROOT.TCanvas()
ab_hist.Draw()

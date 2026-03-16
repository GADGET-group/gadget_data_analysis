import os

import numpy as np
import ROOT
#ROOT.EnableImplicitMT()

from raw_viewer import ddas_interface
from e23035_analysis import fitting_tools, root_vis_tools, e23035_runs, degai

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

adj_dict = degai.clover_adj_dict #degai.get_adjacency_dict(30)#clarion.get_adjacency_dict(30) #I've saved 1 deg (eg no add back) and 30 deg (adjacent crystals) before

event_build_window = 500 #ns

#coinc_canvas = ROOT.TCanvas()
gg_hist = degai.get_addback_coincidence_spectrum(runs, adj_dict, 'gm', gamma_binning, event_build_window)
# gg_hist.Draw('COLZ')
# ROOT.gPad.SetLogz(1)
print('getting add back histogram')
ab_hist = degai.get_addback_spectrum(runs, adj_dict, 'gm', gamma_binning, event_build_window)

print('getting sum histogram')
sum_hist = degai.get_summed_gamma_spectrum(runs, gamma_binning, 'gm')
sum_hist.SetLineColor(ROOT.kRed)


spec_canvas = ROOT.TCanvas()
gspec = degai.get_bg_subtracted_projection(gg_hist, 1003.7, 2, 1045, 2)
gspec.SetLineColor(ROOT.kGreen)


ab_hist.Draw()
sum_hist.Draw('SAME')
gspec.Draw('SAME')
spec_canvas.SetLogy(1)

timing_hist = degai.get_adjacent_timing_spectrum(126, adj_dict, (1500, 0, 15000))

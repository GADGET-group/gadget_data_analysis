import os

import numpy as np
import ROOT
#ROOT.EnableImplicitMT()

from raw_viewer import ddas_interface
from e23035_analysis import fitting_tools, root_vis_tools, e23035_runs, degai
from e23035_analysis.spectrum_fitter import spectrum_fitter

'''
Notes on peak finding an automatic fitting
https://root.cern/root/htmldoc/guides/spectrum/Spectrum.html
'''
gamma_bin_size = 1 #keV
addback_ethresh = 150
gamma_binning = (int((7000-0)/gamma_bin_size),addback_ethresh,7000) #was 1-12000 w/ 1 keV bins
#run_candidates = e23035_runs.run_df['DDAS'][(e23035_runs.run_df['Run Type']=='60Ga')]
runs = e23035_runs.get_ddas_60_Ga_runs()

event_build_window = 500 #ns

 ##clarion.get_adjacency_dict(30) #I've saved 1 deg (eg no add back) and 30 deg (adjacent crystals) before

crystal_canvas, crystal_th2 = None, None
def show_crystal_spectra():
    global crystal_canvas, crystal_th2
    crystal_hists = degai.get_crystal_histograms(runs, gamma_binning, 'cal', 'gm')
    crystal_canvas, crystal_th2 = root_vis_tools.create_2d_hist_from_dict(crystal_hists, '60Ga energy spectra')
    crystal_canvas.SetLogz(1)

cvrun_canvas, cvrun_hist = None, None
def show_crystal_vs_run(crystal_name):
    global cvrun_canvas, cvrun_hist
    hists_to_plot = {}
    for run in runs:
        hists_to_plot['%d'%run] = degai.get_crystal_histograms(run, gamma_binning, 'cal', 'gm')[crystal_name+'_gm']
    cvrun_canvas, cvrun_hist = root_vis_tools.create_2d_hist_from_dict(hists_to_plot, f'{crystal_name} spectra', 'run')
    cvrun_canvas.SetLogz(1)



# plot_hist_dic = {'adjacent addback':adj_ab_hist, 'clover addback':clover_ab_hist, 'summed spectrum':sum_hist}
# canvas, legend, stack = root_vis_tools.draw_overlaid_histograms(plot_hist_dic, 'gamma spectrum from 60Ga runs', "keV", "counts/1 keV")


clover_ab_hist = degai.get_addback_spectrum(runs, degai.clover_adj_dict, 'gm', gamma_binning, event_build_window, addback_ethresh)

adj_dict = degai.get_adjacency_dict(30)
print('getting add back histogram')
adj_ab_hist = degai.get_addback_spectrum(runs, adj_dict, 'gm', gamma_binning, event_build_window, addback_ethresh)
print('getting sliding scale with addback')
adj_ab_ss_hist = degai.get_addback_spectrum(runs, adj_dict, 'gm', gamma_binning, event_build_window, addback_ethresh, True)

print('getting sum histogram')
# sum_hist = degai.get_summed_gamma_spectrum(runs, gamma_binning, 'gm')
# sum_hist.SetLineColor(ROOT.kRed)

#this should be equivalent to sum spectrum
addback_1deg_hist = degai.get_addback_spectrum(runs, degai.get_adjacency_dict(1), 'gm', gamma_binning, event_build_window, addback_ethresh)

# if True:
gg_hist = degai.get_addback_coincidence_spectrum(runs, adj_dict, 'gm', gamma_binning, event_build_window, addback_ethresh, event_build_window)#adj_dict
cspec = ROOT.TCanvas()
gspec = degai.get_bg_subtracted_projection(gg_hist, 1003.7, 2, 1024, 2)
gspec.SetLineColor(ROOT.kRed)
gspec.Draw()
gspec2 = degai.get_bg_subtracted_projection(gg_hist, 1003.7, 2, 1141, 2)
gspec2.SetLineColor(ROOT.kBlue)
gspec2.Draw('SAME')
gspec3 = degai.get_bg_subtracted_projection(gg_hist, 1003.7, 2, 1265, 2)
gspec3.SetLineColor(ROOT.kGreen)
gspec3.Draw('SAME')

plot_hist_dic = {'adjacent addback':adj_ab_hist, 'clover addback':clover_ab_hist,  '1deg addback':addback_1deg_hist, 'ss':adj_ab_ss_hist} #\'summed spectrum':sum_hist,
#, , \
#                    'adj addback w/ 1ns event window':adj_addback_1ns_build_window}
canvas, legend, stack = root_vis_tools.draw_overlaid_histograms(plot_hist_dic, 'gamma spectrum from 60Ga runs', "keV", "counts/1 keV")
canvas.SetLogy(1)

f = spectrum_fitter(adj_ab_hist, 'gaus')
f.peaks_to_fit = [511, 1003.5, 1555, 2293,2559,3848.3]
for i in range(len(f.peaks_to_fit)):
    f.peaks_to_fit[i] = (f.peaks_to_fit[i],f.peaks_to_fit[i]*0.99, f.peaks_to_fit[i]*1.01)
f.fit_peaks()


#timing_hist = degai.get_adjacent_timing_spectrum(126, degai.get_adjacency_dict(180), (1500, 0, 15000))
if False:
    indiv_adj_dict = degai.get_adjacency_dict(1)
    gg_hist = degai.get_addback_coincidence_spectrum(runs, indiv_adj_dict, 'gm', gamma_binning, event_build_window, 0, event_build_window)
    gspec = degai.get_bg_subtracted_projection(gg_hist, 2614, 1, 2700, 2)

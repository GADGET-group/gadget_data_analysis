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
    t0, tf = ddas_interface.get_first_and_last_ddas_time(run)
    if not np.isnan(run) and run not in [162,163,203,204,209, 213,217, 218, 238] and run>=150 and run not in[169, 170,171, 172,173,174, 180, 181] and not (run>=182 and run<=191):# and (tf-t0)>600:
        #only looking at runs later than 150 since these definitely use final beam settings
        #169-173: beam disruptions, and following short runs
        #174: attenuated beam
        #180, 181: grow in after PID
        #Runs 182-191 also have poor beharior. Run 187 was LN2 fill, but reason for other runs is unknown.
        if os.path.exists(ddas_interface.get_merged_root_file_path(run)):
            runs.append(run)
print(runs)

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


clover_ab_hist = degai.get_addback_spectrum(runs, degai.clover_adj_dict, 'gm', gamma_binning, event_build_window)

adj_dict = degai.get_adjacency_dict(30)
print('getting add back histogram')
adj_ab_hist = degai.get_addback_spectrum(runs, adj_dict, 'gm', gamma_binning, event_build_window)

print('getting sum histogram')
sum_hist = degai.get_summed_gamma_spectrum(runs, gamma_binning, 'gm')
sum_hist.SetLineColor(ROOT.kRed)

#this should be equivalent to sum spectrum
addback_1deg_hist = degai.get_addback_spectrum(runs, degai.get_adjacency_dict(1), 'gm', gamma_binning, event_build_window)

# if True:
gg_hist = degai.get_addback_coincidence_spectrum(runs,  degai.get_adjacency_dict(1), 'gm', gamma_binning, event_build_window)#adj_dict
cspec = ROOT.TCanvas()
gspec = degai.get_bg_subtracted_projection(gg_hist, 1003.7, 2, 1045, 2)
gspec2 = degai.get_bg_subtracted_projection(gg_hist, 1003.7, 2, 1141, 2)
gspec2.SetLineColor(ROOT.kBlue)
gspec2.Draw('SAME')
gspec3 = degai.get_bg_subtracted_projection(gg_hist, 1003.7, 2, 1265, 2)
gspec3.SetLineColor(ROOT.kGreen)
gspec3.Draw('SAME')


adj_addback_1ns_build_window = degai.get_addback_spectrum(runs, degai.get_adjacency_dict(30), 'gm', gamma_binning, 1)
plot_hist_dic = {'adjacent addback':adj_ab_hist, 'clover addback':clover_ab_hist, 'summed spectrum':sum_hist, '1deg addback':addback_1deg_hist} #\
#, , \
#                    'adj addback w/ 1ns event window':adj_addback_1ns_build_window}
canvas, legend, stack = root_vis_tools.draw_overlaid_histograms(plot_hist_dic, 'gamma spectrum from 60Ga runs', "keV", "counts/1 keV")
canvas.SetLogy(1)


#timing_hist = degai.get_adjacent_timing_spectrum(126, degai.get_adjacency_dict(180), (1500, 0, 15000))
if False:
    indiv_adj_dict = degai.get_adjacency_dict(1)
    gg_hist = degai.get_addback_coincidence_spectrum(runs, indiv_adj_dict, 'gm', gamma_binning, event_build_window)
    gspec = degai.get_bg_subtracted_projection(gg_hist, 2614, 1, 2700, 2)

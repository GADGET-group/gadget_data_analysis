import os
import multiprocessing

import numpy as np
import ROOT

from raw_viewer import ddas_interface
from e23035_analysis import fitting_tools, root_vis_tools, e23035_runs, energy_calibration_tools

'''
Notes on peak finding an automatic fitting
https://root.cern/root/htmldoc/guides/spectrum/Spectrum.html
'''
gamma_bin_size = 1 #keV
gamma_binning = (int((7000-0)/gamma_bin_size),0,7000) #was 1-12000 w/ 1 keV bins
run_candidates = e23035_runs.run_df['DDAS'][(e23035_runs.run_df['Run Type']=='60Ga')]
runs = []
for run in run_candidates:
    if not np.isnan(run) and run not in [162,163,203,204,209, 213,217, 218, 238]:
        if os.path.exists(ddas_interface.get_merged_root_file_path(run)):
            runs.append(run)
runs=[126] #126 production run, 91 is bg
n_workers=min(200, len(runs))

#get pre-experiment energy calibration to make finding the 511 and 2614 keV peaks easier
ddas_ch_map_path = 'e23035_analysis/channel_map.csv'
chmap = np.genfromtxt(ddas_ch_map_path,delimiter=', ', dtype=str, skip_header=1)
ch_indexes = np.array(chmap[:,0], dtype=int)
ch_names = chmap[:,1]
init_slopes, init_offsets = np.array(chmap[:,2], dtype=float), np.array(chmap[:,3], dtype=float)
init_cal_dict = {} #maps channel name to slope, offset
for i in range(len(ch_names)):
    init_cal_dict[ch_names[i]] = (init_slopes[i], init_offsets[i])


clover_list = []
for num in range(1, 12):
    if num == 4 or num == 8:
        continue
    for letter in ['a', 'b', 'c', 'd']:
        clover_list.append(f'{num}{letter}')

def do_gain_match(ddas_run):
    '''
    ddas_run
    clover: 1a, 1b, etc
    '''
    original_batch_state = ROOT.gROOT.IsBatch()
    ROOT.gROOT.SetBatch(True)
    true_locations = [510.99895069, 2614.511]
    true_location_uncertainties =  [16e-7, 1e-2]
    
    for clover in clover_list:
        peaks = []
        clover_str = f'clover_{clover}'
        init_slope, init_offset = init_cal_dict[clover_str]
        adc_str = f'clover_{clover}_c'
        for i in range(len(true_locations)):
            loc_guess = (true_locations[i] - init_offset)/init_slope
            fit_window_width = 0.01*loc_guess*np.sqrt(511/true_locations[i])
            search_window_width = 50/init_slope
            #(true energy, true energy_uncertainty, location guess, location_wiggle, fit range start, fit range stop)
            peaks.append((true_locations[i], true_location_uncertainties[i],  (loc_guess-search_window_width, loc_guess+search_window_width), (-fit_window_width, fit_window_width)))
        energy_calibration_tools.make_energy_calibration(ddas_run, 'gm', adc_str, (2**16, 0, 2**16), peaks)

     #save histogram showing energy alignment 
    crystal_e_hists = ddas_interface.get_crystal_histograms(ddas_run, (6000, 0, 6000), 'e')
    canvas, th2 = root_vis_tools.create_2d_hist_from_dict(crystal_e_hists, "pre-experiment energy calibration")
    ROOT.gPad.SetLogz(1)
    canvas.Update()
    fname = f"e23035_analysis/calibrations/{ddas_run}/gm/initial_alignment.pdf"
    canvas.Print(fname+'(')
    dE_to_plot = 100
    for loc in true_locations:
        th2.GetXaxis().SetRangeUser(loc-dE_to_plot, loc+dE_to_plot)
        canvas.Update()
        if loc == true_locations[-1]:
            canvas.Print(fname+')')
        else:
            canvas.Print(fname)

    #save histogram showing enrgy alignment after gain matching
    gm_e_hists = ddas_interface.get_crystal_histograms(ddas_run, (6000, 0, 6000), 'cal', 'gm')
    canvas, th2 = root_vis_tools.create_2d_hist_from_dict(gm_e_hists, "with gain match applied")
    ROOT.gPad.SetLogz(1)
    
    canvas.Update()
    fname = f"e23035_analysis/calibrations/{ddas_run}/gm/gain_match.pdf"
    canvas.Print(fname+'(')
    dE_to_plot = 100
    for loc in true_locations:
        th2.GetXaxis().SetRangeUser(loc-dE_to_plot, loc+dE_to_plot)
        canvas.Update()
        if loc == true_locations[-1]:
            canvas.Print(fname+')')
        else:
            canvas.Print(fname)

    ROOT.gROOT.SetBatch(original_batch_state)
    



def process_all():
    runs = np.array(e23035_runs.run_df['DDAS'][np.isfinite(e23035_runs.run_df['DDAS'])], dtype=int)
    with multiprocessing.Pool(50) as pool:
        pool.map(do_gain_match, runs)
#h = ddas_interface.get_histogram(runs[0], (2**16, 0, 2**16), '3a_counts', '3a_counts', 'clover_3a_c')
#guess = 3295.8#(511-2.609910)/0.154992 #9405
# fit_res, background, peaks, rp, canvas, spectrum_to_plot, f_to_fit, h_fit = fitting_tools.fit_emg_peak(h, 'gamma_adc',guess, 100, (-25,25))
# fit_res2, background2, peaks2, rp2, canvas2, spectrum_to_plot2, f_to_fit2, h_fit2=fitting_tools.fit_gaussian_peaks(h, [guess],20,(guess-25, guess+25) ,True, background_type='constant')

# crystal_m_hists = ddas_interface.get_crystal_histograms(runs, (10, -0.5, 9.5), 'm')
# root_vis_tools.create_2d_hist_from_dict(crystal_m_hists, "crystal multiplicities")
ROOT.EnableImplicitMT()
run = 126
counts_branch_list = [f'clover_{clover}_c' for clover in clover_list]
gm_branch_list = [s + '_gm' for s in counts_branch_list]
df = energy_calibration_tools.get_run_dataframe(run)
df = energy_calibration_tools.apply_calibration(df, run, counts_branch_list, 'gm')
add_back_logic = ''
crystal_strings_for_max = ''
for i in range(len(gm_branch_list)):
    if i != 0:
        add_back_logic += '+'
        crystal_strings_for_max += ','
    add_back_logic += gm_branch_list[i]
    crystal_strings_for_max += gm_branch_list[i]
df = df.Define('add_back', add_back_logic)
add_back_hist = df.Histo1D(("add_back", "add back", 6000, 0., 6000.), "add_back")
df = df.Define("summed_gamma", "std::max({%s})"%crystal_strings_for_max)
summed_hist = df.Histo1D(("summed_gamma", "summed gamma", 6000, 0., 6000.), "add_back")
#ddas_interface.get_summed_gamma_spectrum(run, (6000,0, 6000), 'gm')
#root_vis_tools.draw_overlaid_histograms({'add back':add_back_hist, 'summed':summed_hist}, f'run {run}', 'keV')

# add_back_hist.Draw()

summed_hist.SetLineColor(ROOT.kRed)
summed_hist.Draw("SAME")

from e23035_analysis import clarion
adj_30_add_back_tree = clarion.get_addback_tree(run, clarion.get_adjacency_dict(30), 'gm')
df_adj = ROOT.RDataFrame(adj_30_add_back_tree)
adj_hist = df_adj.Histo1D(("h_energy", "Addback Energy;Energy (keV);Counts", 6000, 0, 6000), "energy")
adj_hist.SetLineColor(ROOT.kGreen)
adj_hist.Draw("SAME")

ROOT.gPad.SetLogy(1)
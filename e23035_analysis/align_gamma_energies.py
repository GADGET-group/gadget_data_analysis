import os
import multiprocessing

import numpy as np
import ROOT

from raw_viewer import ddas_interface
from e23035_analysis import fitting_tools, root_vis_tools, e23035_runs, energy_calibration_tools, degai

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

# #true_locations = [510.99895069, 2614.511]
# true_location_uncertainties =  [16e-7, 1e-2]
# norm_dict = {true_locations[0]:'slice', true_locations[1]: 'total'}
# pvalue_threshold_dict = {
#         true_locations[0]:{'1d':1e-7, 't_indep':0.01},
#         true_locations[1]:{'1d':0.001, 't_indep':0.01}
#     }

true_locations = [510.99895069, 1003.72, 2614.511, 3848.3] #511, 60Ga, 208Tl, 60Ga
true_location_uncertainties =  [16e-7, 0.2, 0.01, 0.7]
norm_dict = {true_locations[0]:'slice', true_locations[1]: 'slice', true_locations[2]: 'slice'}
pvalue_threshold_dict = {
        'cal':0.01,
        true_locations[0]:{'1d':1e-7, 't_indep':0.01},
        true_locations[1]:{'1d':0.001, 't_indep':0.01},
        true_locations[2]:{'1d':0.001, 't_indep':0.01},
        true_locations[3]:{'1d':0.001, 't_indep':0.01}
    }

def do_gain_match(ddas_run):
    '''
    ddas_run
    clover: 1a, 1b, etc
    '''
    original_batch_state = ROOT.gROOT.IsBatch()
    ROOT.gROOT.SetBatch(True)
    
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
        energy_calibration_tools.make_energy_calibration(ddas_run, 'gm', adc_str, (2**16, 0, 2**16), peaks, time_bin_size=1800, normalization_dict=norm_dict)

     #save histogram showing energy alignment 
    crystal_e_hists = degai.get_crystal_histograms(ddas_run, (6000, 0, 6000), 'e')
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

    #save histogram showing energy alignment after gain matching
    gm_e_hists = degai.get_crystal_histograms(ddas_run, (6000, 0, 6000), 'cal', 'gm')
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

    print('gain match of run %d is complete'%ddas_run)
    



import multiprocessing
import os
import sys
from tqdm import tqdm

def _mute_worker():
    """
    Runs once per worker process when the pool starts. 
    Silences tqdm and standard print statements so they don't corrupt the main progress bar.
    """
    # 1. Disable tqdm globally for this worker process
    os.environ['TQDM_DISABLE'] = '1'
    
    # 2. (Optional) Redirect standard prints to the void so they don't mess up the terminal
    # Comment this out if you still want to see normal print() statements from the workers!
    sys.stdout = open(os.devnull, 'w')

def process_all():
    # runs = np.array(e23035_runs.run_df['DDAS'][np.isfinite(e23035_runs.run_df['DDAS'])], dtype=int)
    
    # Pass our mute function to the pool via the initializer argument
    with multiprocessing.Pool(n_workers, initializer=_mute_worker) as pool:
        
        # imap_unordered yields results as they finish, allowing tqdm to track them!
        results_generator = pool.imap_unordered(do_gain_match, runs)
        
        # Wrap the generator in tqdm and exhaust it using list()
        list(tqdm(
            results_generator, 
            total=len(runs), 
            desc="Gain Matching Runs", 
            dynamic_ncols=True,
            smoothing=0.1 # Averages the speed estimate so it doesn't jump around wildly
        ))
        
    make_summary_pdf()
    
def make_summary_pdf():
    energy_calibration_tools.create_calibration_summary('gm', pvalue_threshold_dict, runs)


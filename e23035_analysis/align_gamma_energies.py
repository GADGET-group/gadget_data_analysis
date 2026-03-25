import os
import multiprocessing

import sys
from tqdm import tqdm

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
    t0, tf = ddas_interface.get_first_and_last_ddas_time(run)
    if not np.isnan(run) and run not in [162,163,203,204,209, 213,217, 218, 238] and run>=150 and run not in[159, 169, 170,171, 172,173,174, 180, 181] and not (run>=182 and run<=191):# and (tf-t0)>600:
        #only looking at runs later than 150 since these definitely use final beam settings
        #169-173: beam disruptions, and following short runs
        #174: attenuated beam
        #180, 181: grow in after PID
        #Runs 182-191 also have poor beharior. Run 187 was LN2 fill, but reason for other runs is unknown.
        if os.path.exists(ddas_interface.get_merged_root_file_path(run)):
            runs.append(run)
print(runs)
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
        'cal':1e-6,
        true_locations[0]:{'1d':1e-12, 't_indep':0.01},
        true_locations[1]:{'1d':0.0001, 't_indep':0.01},
        true_locations[2]:{'1d':0.0001, 't_indep':0.01},
        true_locations[3]:{'1d':0.0001, 't_indep':0.01}
    }
rebin_factors = [1,1,5,5]

def do_gain_match(ddas_run):
    '''
    ddas_run
    clover: 1a, 1b, etc
    '''
    ddas_run = int(ddas_run)
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
            peaks.append((true_locations[i], true_location_uncertainties[i],  (loc_guess-search_window_width, loc_guess+search_window_width), (-fit_window_width, fit_window_width), rebin_factors[i]))
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
    



# Global variable to hold the shared dictionary inside each worker process
worker_status_dict = None

def _tracking_initializer(shared_dict):
    """
    Initializes the worker, hands it the shared dictionary, and mutes terminal spam.
    """
    global worker_status_dict
    worker_status_dict = shared_dict
    
    # Silence tqdm and standard print statements for the workers
    os.environ['TQDM_DISABLE'] = '1'
    sys.stdout = open(os.devnull, 'w')

def _tracked_gain_match(run):
    """
    Wrapper function that updates the shared dictionary before and after processing.
    """
    global worker_status_dict
    
    # 1. Update status to show this run is actively chewing up CPU
    worker_status_dict[run] = "RUNNING"
    
    try:
        # 2. Call your actual heavy function
        do_gain_match(run)
        
        # 3. Mark as finished and return the run number back to the main thread
        worker_status_dict[run] = "FINISHED"
        return run, True
        
    except Exception as e:
        # If it crashes, mark it failed so we know!
        worker_status_dict[run] = "FAILED"
        return run, False

def process_all():
    # runs = np.array(e23035_runs.run_df['DDAS'][np.isfinite(e23035_runs.run_df['DDAS'])], dtype=int)
    
    # 1. Create the shared memory Manager
    with multiprocessing.Manager() as manager:
        status_dict = manager.dict()
        
        # Pre-fill the status board so everything starts as "QUEUED"
        for r in runs:
            status_dict[r] = "QUEUED"
            
        # 2. Start the pool, passing the shared dictionary via initargs
        with multiprocessing.Pool(n_workers, initializer=_tracking_initializer, initargs=(status_dict,)) as pool:
            
            # Point the pool at our new wrapper function
            results_generator = pool.imap_unordered(_tracked_gain_match, runs)
            
            # 3. Process the results manually with a tqdm loop
            pbar = tqdm(total=len(runs), desc="Gain Matching", dynamic_ncols=True, smoothing=0.1)
            
            for run_id, success in results_generator:
                pbar.update(1)
                
                # Scan the shared dictionary for runs that are currently pinned to a worker
                active_runs = [r for r, status in status_dict.items() if status == "RUNNING"]
                
                # tqdm.write prints safely ABOVE the progress bar
                tqdm.write(f"Run {run_id} finished. Actively running now: {sorted(active_runs)}")
                
            pbar.close()
            
    make_summary_pdf()
    
def make_summary_pdf():
    energy_calibration_tools.create_calibration_summary('gm', pvalue_threshold_dict, runs)

t_tot = 0
for run in runs:
    t1, t2 =  ddas_interface.get_first_and_last_ddas_time(run)
    t_tot += t2 - t1
print(f'total run time: {t_tot/3600} hours')

stability_binning = (int(7000/5), 0, 7000)
energy_calibration_tools.create_stability_summary('gm', stability_binning, 0.01, 200, runs)
import os
from pathlib import Path
import csv

import numpy as np
import ROOT

from raw_viewer import ddas_interface
from e23035_analysis import fitting_tools, root_vis_tools, e23035_runs, degai
from e23035_analysis.spectrum_fitter import spectrum_fitter, load_spectrum_fitter_from_file

gamma_bin_size = 0.25 #keV
addback_ethresh = 150
upper_energy = 7000
gamma_binning = (int((upper_energy-addback_ethresh)/gamma_bin_size),addback_ethresh,upper_energy) #was 1-12000 w/ 1 keV bins
#run_candidates = e23035_runs.run_df['DDAS'][(e23035_runs.run_df['Run Type']=='60Ga')]
if True:
    runs = e23035_runs.get_ddas_60_Ga_runs(good_gamma=True, good_low_energy_tpc=False, good_long_tracks_tpc=False, final_beam_settings=True)
    fit_prefix = '60Ga'
if False:
    bg_run = 281
    runs = [bg_run]
    fit_prefix = 'bg_run_%d'%bg_run
if False:
    runs = [280, 278, 277, 274, 271, 270, 269, 268]
    fit_prefix = '59Zn'


run_duration = 0
for run in runs:
    t1, t2 = ddas_interface.get_first_and_last_ddas_time(run)
    run_duration += (t2 - t1)
print('duration: %s hr'%(run_duration/3600))

event_build_window = 500 #ns


adj_dict =  degai.crystal_adj_dict#degai.clover_adj_dict#
cal_name = 'gm_511and2614_1'
nlc_name = 'c1'
gamma_hist = degai.get_histogram(runs, adj_dict, cal_name, gamma_binning, 'gamma_hist', 'gamma spectrum', 'addback_energy', '', event_build_window, addback_ethresh, True,
                                  nonlinearity_correction_name=nlc_name)
gamma_beam_off_hist = degai.get_histogram(runs, adj_dict, cal_name, gamma_binning, 'beam_off_gammas', 'beam off gamma spectrum', 'addback_energy', 
                                          "time_since_beam_off<0.094", event_build_window, addback_ethresh, True,
                                        nonlinearity_correction_name=nlc_name)
gamma_beam_on_hist = degai.get_histogram(runs, adj_dict, cal_name, gamma_binning, 'beam_on_gammas', 'beam on gamma spectrum', 'addback_energy', 
                                          "(time_since_beam_off>0.1) && (time_since_beam_off<0.195)", event_build_window, addback_ethresh, True,
                                        nonlinearity_correction_name=nlc_name)
beam_on_off_drawing= root_vis_tools.draw_overlaid_histograms({'beam off':gamma_beam_off_hist, 'beam on':gamma_beam_on_hist, 'all':gamma_hist})


fit_model = 'bg_shift_nemg'#'bg_shift_ngaus'#
sigma1_func = lambda E: 0.02078*np.sqrt(E + 742.9)
tau1_func = lambda E: 0.0001117*E + 0.157
sigma2_func = lambda E: 0.03037*np.sqrt(E + 498.5)
tau2_func = lambda E: max(0.01, 0.0003378*E - 0.2317)
bg_shift_func = lambda E:(0, 0.004 if E >500 else 0.01)

#def fit_nemg_w_bg_shift(spectrum:ROOT.TH1D, e_guess:float|list, fit_window:tuple, num_emgs:int, data_source=None, param_bounds=None, fit_options='LS0QEI'): 
fitters = []
def fit_peak(spectrum, location, window_start, window_stop):
    f = spectrum_fitter(spectrum, fit_model) # This will now use N=2 by default
    f.nemg = f.ngaus = 2
    f.peaks_to_fit = [(location, window_start, window_stop)]
    f.param_bound_functions['sigma1'] = lambda E: (sigma1_func(E), sigma1_func(E))
    f.param_bound_functions['tau1'] = lambda E: (tau1_func(E), tau1_func(E))
    f.param_bound_functions['sigma2'] = lambda E: (sigma2_func(E), sigma2_func(E))
    f.param_bound_functions['tau2'] = lambda E: (tau2_func(E), tau2_func(E))
    f.param_bound_functions['bg_shift'] = bg_shift_func
    f.fit_peaks()
    f.show_fit_results(0)
    fitters.append(f)

def fit_exists(save_name):
    save_path = os.path.join('e23035_analysis/peak_fitting/',fit_prefix+'_'+save_name + '.root')
    return Path(save_path).exists()

def get_fitter(save_name):
    save_path = os.path.join('e23035_analysis/peak_fitting/',fit_prefix+'_'+save_name)
    return load_spectrum_fitter_from_file(save_path+'.root')

def fit_peaks(spectrum, peaks, save_name, zero_bg_shift, likelihood, manual_bounds=False,force_refit=False):
    '''
    manual_bounds: if False, use add peaks function to cluster peaks and set fitting bounds.
    '''
    if not force_refit and fit_exists(save_name):
        return get_fitter(save_name)
    f = spectrum_fitter(spectrum, fit_model) # This will now use N=2 by default
    f.nemg = f.ngaus = 2
    f.param_bound_functions['sigma1'] = lambda E: (sigma1_func(E), sigma1_func(E))
    f.param_bound_functions['tau1'] = lambda E: (tau1_func(E), tau1_func(E))
    f.param_bound_functions['sigma2'] = lambda E: (sigma2_func(E), sigma2_func(E))
    f.param_bound_functions['tau2'] = lambda E: (tau2_func(E), tau2_func(E))
    f.param_bound_functions['bg_shift'] = bg_shift_func
    f.spectrum.GetXaxis().UnZoom()
    bg_const_bounds = (f.spectrum.GetMinimum(), f.spectrum.GetMaximum())
    f.param_bound_functions['bg_const'] = lambda E: bg_const_bounds
    if manual_bounds:
        f.peaks_to_fit = peaks
    else:
        f.add_peaks(peaks, lambda E: sigma2_func(E)*10, sep_factor=1.5)
    if zero_bg_shift:
        f.param_bound_functions['bg_shift'] = lambda E: (0, 0) #coincidence peaks should be really weak
    if not likelihood:
        f.fit_options = f.fit_options.replace('L','')
    f.fit_peaks()
    f.save(os.path.join('e23035_analysis/peak_fitting/',fit_prefix+'_'+save_name))
    return f

def compton_edge(E):
     return E - E/(1 + 2*E/511)

# Use a coarser binning for the 2D coincidence matrix to prevent ROOT's 1GB serialization limit
coincidence_bin_size = 1.0 # keV
coincidence_binning = (int((upper_energy-addback_ethresh)/coincidence_bin_size), addback_ethresh, upper_energy)
coincidence_hist = degai.get_addback_coincidence_spectrum(runs, adj_dict, cal_name, coincidence_binning, event_build_window, addback_ethresh, event_build_window, True, 
                                                nonlinearity_correction_name=nlc_name)
# h6134 = degai.get_bg_subtracted_projection(gg_hist, 6134, 4, 6180, 20)

#make 2D histogram of time vs energy
E_v_tsbo = degai.get_histogram(runs, adj_dict, cal_name, (200, 0, 0.200, *coincidence_binning), "E_vs_tsbo", "energy (keV) vs time (s)", "addback_energy:time_since_beam_off", 
                    dt_window_ns=event_build_window, e_thresh=addback_ethresh, nonlinearity_correction_name=nlc_name)
run_start_time, run_stop_time = ddas_interface.get_first_and_last_ddas_time(runs[0]) #currently only works for a single run
E_v_t = degai.get_histogram(runs, adj_dict, cal_name, (2000, run_start_time, run_stop_time, *coincidence_binning), "E_vs_t", "energy (keV) vs time (s)", "addback_energy:time", 
                    dt_window_ns=event_build_window, e_thresh=addback_ethresh, nonlinearity_correction_name=nlc_name)

# E_v_tsco = degai.get_histogram(runs, adj_dict, cal_name, (2000, 0, 0.200, *coincidence_binning), 
#                     "E_vs_t_c", "energy (keV) vs time since chopper off (s)", "addback_energy:time_since_chopper_off", 
#                     dt_window_ns=event_build_window, e_thresh=addback_ethresh, nonlinearity_correction_name=nlc_name)

fit_results = []
def fit_decay_curve(Egate, tgate, Egate_bg=None, source=E_v_tsbo):
    if Egate_bg is None:
        h_to_fit = degai.get_gated_projection(E_v_tsbo, Egate)
    else:
        h_to_fit = degai.get_bg_subtracted_projection(source, Egate, Egate_bg)
    fit_string = '[0]*exp(-0.693147*x/[1]) + [2]'
    res = fitting_tools.fit_hist(h_to_fit, fit_string, [1,1, 0],
                        ((0, 1e9), (0., 3600), (0, h_to_fit.GetMaximum())), tgate,
                        fit_options='S0QEI', names=['A', 'half life (s)', 'bg'])
    fit_results.append(res)
    
    fit_ptr = res[0]
    hl = fit_ptr.Parameter(1)
    lower_err = fit_ptr.LowerError(1)
    upper_err = fit_ptr.UpperError(1)
    print(f"Half-life for Egate {Egate}: {hl:.4g} ({lower_err:.4g}/+{upper_err:.4g}) s. Bounds: [{hl + lower_err:.4g}, {hl + upper_err:.4g}] s")


all_peaks = []
with open('e23035_analysis/peak_fitting/gamma_peaks.csv', 'r') as f:
    reader = csv.reader(f)
    current_group = []
    fit_window=(0,0)
    for row in reader:
        if row:
            if row[0]=='STOP':
                break
            try:
                if len(row[0]) > 0:
                    if len(current_group) > 0:
                        all_peaks.append((current_group, *fit_window))
                        current_group = []
                    start, stop = row[0].split('-')
                    fit_window = (float(start), float(stop))
                current_group.append(float(row[1]))
            except Exception as e:
                print(f'failed to load peak from row {row}: {e}')
    
    if len(current_group) > 0:
        all_peaks.append((current_group, *fit_window))

force_refit=False
f_all = fit_peaks(gamma_hist, all_peaks, 'all_gamma', False, True, manual_bounds=True, force_refit=force_refit)
if True:
    f_beam_off = fit_peaks(gamma_beam_off_hist, all_peaks, 'beam_off_gamma', False, True, manual_bounds=True, force_refit=force_refit)
    f_beam_on = fit_peaks(gamma_beam_on_hist, all_peaks, 'beam_on_gamma', False, True, manual_bounds=True, force_refit=force_refit)

    possible_coincidence_peaks = [511, 546, 1003, 1028, 1188, 1202,  1333, 1341, 1413, 1482, 1554, 2007,
                2293, 2334, 2390, 2435, 2484, 2507, 2826, 2996, 
            3337, 3378, 3588, 3848, 3888, 4177, 4208, 4719, 4806]
    #force_refit=True
    h1003 = degai.get_bg_subtracted_projection(coincidence_hist, (1002.0, 1005.0), (1009, 1011))
    f1003=fit_peaks(h1003, possible_coincidence_peaks,
            '1003keV_coincidence', True, False, force_refit=force_refit)

    h1028 = degai.get_bg_subtracted_projection(coincidence_hist, (1027, 1029), (1038, 1042))
    f1028=fit_peaks(h1028, possible_coincidence_peaks,
            '1028keV_coincidence', True, False,force_refit=force_refit)

    h1189 = degai.get_bg_subtracted_projection(coincidence_hist, (1188, 1190),(1194,1198))
    f1189=fit_peaks(h1189, possible_coincidence_peaks,
            '1189keV_coincidence', True, False,force_refit=force_refit)

    h1202 = degai.get_bg_subtracted_projection(coincidence_hist, (1200,1204),(1210,1213))
    f1202 = fit_peaks(h1202, possible_coincidence_peaks,
            '1202keV_coincidence', True, False,force_refit=force_refit)



    #fit_decay_curve((2006,2009), (0,0.095), (2018, 2038))
    h2007 = degai.get_bg_subtracted_projection(coincidence_hist, (2006, 2009), (2018, 2038))
    f2007=fit_peaks(h2007, possible_coincidence_peaks,
            '2007keV_coincidence', True, False,force_refit=force_refit)
    
    h4179 = degai.get_bg_subtracted_projection(coincidence_hist, (4175, 4180), (4213, 4255))
    f4179 = fit_peaks(h4179, possible_coincidence_peaks,
            '4179keV_coincidence', True, False,force_refit=force_refit)
    fit_decay_curve((4175, 4180), (0.005,0.095), (4213, 4255))

    h4189 = degai.get_bg_subtracted_projection(coincidence_hist, (4187, 4193),(4213, 4255))
    f4189 = fit_peaks(h4189, possible_coincidence_peaks,
            '4189keV_coincidence', True, False,force_refit=force_refit)
    fit_decay_curve((4187, 4193),(0.005,0.095), (4213, 4255))

    h4208 = degai.get_bg_subtracted_projection(coincidence_hist, (4205, 4211), (4213, 4255))
    f4208 = fit_peaks(h4208, possible_coincidence_peaks,
            '4208keV_coincidence', True, False,force_refit=force_refit)
    #fit_decay_curve((4205, 4211), (0.005,0.095), (4213, 4255))

    h4259 = degai.get_bg_subtracted_projection(coincidence_hist, (4257, 4265), (4213, 4255))
    f4259 = fit_peaks(h4259, possible_coincidence_peaks,
            '4259keV_coincidence', True, False,force_refit=force_refit)
    fit_decay_curve((4257, 4265), (0.005,0.095), (4213, 4255))


    h4341 = degai.get_bg_subtracted_projection(coincidence_hist, (4338,4344), (4347, 4376))
    f4341 = fit_peaks(h4341, possible_coincidence_peaks,
            '4341keV_coincidence', True, False,force_refit=force_refit)
    #fit_decay_curve((4338,4344),(0.005,0.095), (4347, 4376))

    h4293 = degai.get_bg_subtracted_projection(coincidence_hist, (4291, 4296), (4269, 4285))
    f4293 = fit_peaks(h4293, possible_coincidence_peaks,
            '4293keV_coincidence', True, False,force_refit=force_refit)
    #fit_decay_curve((4291, 4296),(0.005,0.095), (4269, 4285))


    h4382 = degai.get_bg_subtracted_projection(coincidence_hist, (4379, 4384), (4387, 4440))
    f4382 = fit_peaks(h4382, possible_coincidence_peaks,
            '4382keV_coincidence', True, False,force_refit=force_refit)
    #fit_decay_curve((4379, 4384),(0,0.095), (4387, 4440))

    h4538 = degai.get_bg_subtracted_projection(coincidence_hist, (4535,4542), (4544,4727))
    f4538 = fit_peaks(h4538, possible_coincidence_peaks,
            '4538keV_coincidence', True, False,force_refit=force_refit)
    #fit_decay_curve((4535,4542),(0.005,0.095), (4544,4727))


    h4719 = degai.get_bg_subtracted_projection(coincidence_hist, (4716, 4722), (4727, 4777))
    f4719 = fit_peaks(h4719, possible_coincidence_peaks,
            '4719keV_coincidence', True, False,force_refit=force_refit)
    #fit_decay_curve((4716, 4722),(0.005,0.095), (4727, 4777))


    h4786 = degai.get_bg_subtracted_projection(coincidence_hist, (4785, 4791), (4727, 4751))
    f4786 = fit_peaks(h4786, possible_coincidence_peaks,
            '4786keV_coincidence', True, False,force_refit=force_refit)
    #fit_decay_curve((4785, 4791),(0,0.095), (4727, 4751))


    h4852 = degai.get_bg_subtracted_projection(coincidence_hist, (4847, 4855), (4858, 4884))
    f4852 = fit_peaks(h4852, possible_coincidence_peaks,
            '4858keV_coincidence', True, False,force_refit=force_refit)
    
    h4804 = degai.get_bg_subtracted_projection(coincidence_hist, (4800, 4808), (4816, 4844))
    f4804 = fit_peaks(h4804, possible_coincidence_peaks,
            '4804keV_coincidence', True, False,force_refit=force_refit)
    

    h4892 = degai.get_bg_subtracted_projection(coincidence_hist, (4888, 4896), (4902, 4996))
    f4892 = fit_peaks(h4892, possible_coincidence_peaks,
            '4892keV_coincidence', True, False,force_refit=force_refit)
    
    h5050 = degai.get_bg_subtracted_projection(coincidence_hist, (5045, 5054), (5060, 5094))
    f5050 = fit_peaks(h5050, possible_coincidence_peaks,
            '5050keV_coincidence', True, False,force_refit=force_refit)
    
    h5299=degai.get_bg_subtracted_projection(coincidence_hist,(5294,5303),(5331,5400))
    f5299=fit_peaks(h5299, possible_coincidence_peaks,
            '5299keV_coincidence', True, False,force_refit=force_refit)
    
    h5266 = degai.get_bg_subtracted_projection(coincidence_hist, (5260,5275),(5340,5380))
    f5266 = fit_peaks(h5266, possible_coincidence_peaks,
            '5266keV_coincidence', True, False,force_refit=force_refit)
    

    h5809 = degai.get_bg_subtracted_projection(coincidence_hist, (5803,5812), (5820,5825))
    f5809 = fit_peaks(h5809, possible_coincidence_peaks,
            '5809keV_coincidence', True, False,force_refit=force_refit)

    
    
    
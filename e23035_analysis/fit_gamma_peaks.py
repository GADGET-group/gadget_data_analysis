import os
from pathlib import Path
import csv

import numpy as np
import ROOT

from raw_viewer import ddas_interface
from e23035_analysis import fitting_tools, root_vis_tools, e23035_runs
from raw_viewer import degai
from e23035_analysis.spectrum_fitter import spectrum_fitter, load_spectrum_fitter_from_file

experiment = 'e23035'

gamma_bin_size = 0.25 #keV
addback_ethresh = 150
upper_energy = 7000
gamma_binning = (int((upper_energy-addback_ethresh)/gamma_bin_size),addback_ethresh,upper_energy) #was 1-12000 w/ 1 keV bins
#run_candidates = e23035_runs.run_df['DDAS'][(e23035_runs.run_df['Run Type']=='60Ga')]
fit_prefix = '60Ga'
force_refit=True

beam_on_selection = None
beam_off_selection = None
if fit_prefix == '60Ga':
    runs = e23035_runs.get_ddas_60_Ga_runs(good_gamma=True, good_low_energy_tpc=False, good_long_tracks_tpc=False, final_beam_settings=True)
    cal_name = 'gm_511and2614_1'
    nlc_name = 'c1'
    beam_on_selection = "(time_since_beam_off>0.1) && (time_since_beam_off<0.195)"
    beam_off_selection = "time_since_beam_off<0.094"
elif fit_prefix == '59Zn':
    runs = e23035_runs.get_ddas_59_Zn_runs(good_gamma=True, good_low_energy_tpc=False, good_long_tracks_tpc=False, final_beam_settings=False)
    cal_name = 'gm_511and1301_1'
    nlc_name = 'c1'
    beam_on_selection = "(time_since_beam_off>0.5) && (time_since_beam_off<0.995)"
    beam_off_selection = "time_since_beam_off<0.494"
elif fit_prefix == 'source':
    source_type = '60Co'
    source_position = 'all'
    cal_name = 'gm_511and2614_1'
    nlc_name = 'c1'
    df = e23035_runs.run_df
    if source_position == 'all':
        mask = df['source type'] == source_type
    else:
        mask = (df['source type'] == source_type) & (df['source location'] == source_position)
        
    runs = df.loc[mask, 'DDAS'].dropna().astype(int).tolist()
    fit_prefix = f'source_{source_type}_{source_position}'
    
    beam_on_selection = beam_off_selection = None
if False:
    bg_run = 281
    runs = [bg_run]
    fit_prefix = 'bg_run_%d'%bg_run
if False:
    runs = [280, 278, 277, 274, 271, 270, 269, 268]
    fit_prefix = '59Zn'

#define which peaks shoul be fit
if fit_prefix == 'source':
    if source_type == '60Co':
        all_peaks = [(1173, 1163, 1183), (1332, 1322, 1342)]
    elif source_type == '152Eu':
        all_peaks = []
else:
    all_peaks = []
    with open('e23035_analysis/peak_fitting/gamma_peaks.csv', 'r') as f:
        reader = csv.reader(f)
        current_group = []
        fit_window=(0,0)
        for row in reader:
            if row:
                if row[0]=='STOP':
                    break
                if row[0] == 'fit_window': #skip header
                    continue
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


run_duration = 0
for run in runs:
    t1, t2 = ddas_interface.get_first_and_last_ddas_time(experiment, run)
    run_duration += (t2 - t1)
print('duration: %s hr'%(run_duration/3600))

event_build_window = 500 #ns


adj_dict =  degai.crystal_adj_dict#degai.clover_adj_dict#



gamma_hist = degai.get_histogram(experiment, runs, adj_dict, cal_name, gamma_binning, 'gamma_hist', 'gamma spectrum', 'addback_energy', '', event_build_window, addback_ethresh, True,
                                  nonlinearity_correction_name=nlc_name)
adj_ab_hist = degai.get_histogram(experiment, runs, degai.get_adjacency_dict(30), cal_name, gamma_binning, 'ab_hist', 'addback spectrum', 'addback_energy', '', event_build_window, addback_ethresh, True,
                                  nonlinearity_correction_name=nlc_name)
ab_to_crystal_comparison = root_vis_tools.draw_overlaid_histograms({'addback':adj_ab_hist, 'sum':gamma_hist}, title='addback to sum comparison',
                                                                    x_label='energy (keV)', y_label='counts/0.25 keV')

gamma_beam_off_hist = None
if beam_off_selection is not None:
    gamma_beam_off_hist = degai.get_histogram(experiment, runs, adj_dict, cal_name, gamma_binning, 'beam_off_gammas', 'beam off gamma spectrum', 'addback_energy', 
                                              beam_off_selection, event_build_window, addback_ethresh, True,
                                            nonlinearity_correction_name=nlc_name)

gamma_beam_on_hist = None
if beam_on_selection is not None:
    gamma_beam_on_hist = degai.get_histogram(experiment, runs, adj_dict, cal_name, gamma_binning, 'beam_on_gammas', 'beam on gamma spectrum', 'addback_energy', 
                                              beam_on_selection, event_build_window, addback_ethresh, True,
                                            nonlinearity_correction_name=nlc_name)

if beam_off_selection is not None and beam_on_selection is not None:
    beam_on_off_drawing= root_vis_tools.draw_overlaid_histograms({'beam off':gamma_beam_off_hist, 'beam on':gamma_beam_on_hist, 'all':gamma_hist},
                                                                 x_label='energy (keV)', y_label='counts/0.25 keV')

gamma_beam_off_hist_ab = None
if beam_off_selection is not None:
    gamma_beam_off_hist_ab = degai.get_histogram(experiment, runs, degai.get_adjacency_dict(30), cal_name, gamma_binning, 'beam_off_gammas_ab', 'beam off gamma spectrum_ab', 'addback_energy', 
                                              beam_off_selection, event_build_window, addback_ethresh, True,
                                            nonlinearity_correction_name=nlc_name)

gamma_beam_on_hist_ab = None
if beam_on_selection is not None:
    gamma_beam_on_hist_ab = degai.get_histogram(experiment, runs, degai.get_adjacency_dict(30), cal_name, gamma_binning, 'beam_on_gammas_ab', 'beam on gamma spectrum_ab', 'addback_energy', 
                                              beam_on_selection, event_build_window, addback_ethresh, True,
                                            nonlinearity_correction_name=nlc_name)

if beam_off_selection is not None and beam_on_selection is not None:
    beam_on_off_drawing_ab= root_vis_tools.draw_overlaid_histograms({'beam off':gamma_beam_off_hist_ab, 'beam on':gamma_beam_on_hist_ab, 'all':gamma_hist},
                                                                 x_label='addback energy (keV)', y_label='counts/0.25 keV')

if True: #make plot comparing 60Ga to 59Zn runs
        ga_runs = e23035_runs.get_ddas_60_Ga_runs(good_gamma=True, good_low_energy_tpc=False, good_long_tracks_tpc=False, final_beam_settings=True)
        zn_runs = e23035_runs.get_ddas_59_Zn_runs(good_gamma=True, good_low_energy_tpc=False, good_long_tracks_tpc=False, final_beam_settings=True)
        ga_gamma_hist = degai.get_histogram(experiment, ga_runs, adj_dict, 'gm_511and2614_1', gamma_binning, 'ga_gamma_hist', '60Ga gamma spectrum', 'addback_energy',
                                 '', event_build_window, addback_ethresh, True,
                                  nonlinearity_correction_name=nlc_name)
        zn_gamma_hist = degai.get_histogram(experiment, zn_runs, adj_dict, 'gm_511and1301_1', gamma_binning, 'zn_gamma_hist', '59Zn gamma spectrum', 'addback_energy',
                                 '', event_build_window, addback_ethresh, True,
                                  nonlinearity_correction_name=nlc_name)
        unscaled_zn_ga_comparison = root_vis_tools.draw_overlaid_histograms({'60Ga':ga_gamma_hist, '59Zn':zn_gamma_hist},
                                                             x_label='gamma energy (keV)', y_label='counts/0.25 keV')
        

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

def fit_peaks(spectrum, peaks, save_name, zero_bg_shift, likelihood, manual_bounds=False, force_refit=False, constrain_mu=None):
    if constrain_mu is None:
        constrain_mu = 'coincidence' in save_name
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

    f_all_obj = globals().get('f_all')
    if constrain_mu and f_all_obj is not None:
        def get_singles_mu_bound(E):
            # Use the new method to find the exact mu for this guess
            fitted_mu, _ = f_all_obj.get_param_for_guess('mu', E)
            
            if fitted_mu is not None:
                # Found a match, constrain it tightly
                return (fitted_mu, fitted_mu)
            else:
                # No exact match found, use a wider window as a fallback.
                return (E - f.location_wiggle, E + f.location_wiggle)
        f.param_bound_functions['mu'] = get_singles_mu_bound

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
coincidence_hist = degai.get_addback_coincidence_spectrum(experiment, runs, adj_dict, cal_name, coincidence_binning, event_build_window, addback_ethresh, event_build_window, True, 
                                                nonlinearity_correction_name=nlc_name)
ab_coincidence_hist = degai.get_addback_coincidence_spectrum(experiment, runs, degai.get_adjacency_dict(30), cal_name, coincidence_binning, event_build_window, addback_ethresh, event_build_window, True, 
                                                nonlinearity_correction_name=nlc_name)
# h6134 = degai.get_bg_subtracted_projection(gg_hist, 6134, 4, 6180, 20)

#make 2D histogram of time vs energy
E_v_tsbo = degai.get_histogram(experiment, runs, adj_dict, cal_name, (200, 0, 0.200, *coincidence_binning), "E_vs_tsbo", "energy (keV) vs time (s)", "addback_energy:time_since_beam_off", 
                    dt_window_ns=event_build_window, e_thresh=addback_ethresh, nonlinearity_correction_name=nlc_name)
run_start_time, run_stop_time = ddas_interface.get_first_and_last_ddas_time(experiment, runs[0]) #currently only works for a single run
E_v_t = degai.get_histogram(experiment, runs, adj_dict, cal_name, (2000, run_start_time, run_stop_time, *coincidence_binning), "E_vs_t", "energy (keV) vs time (s)", "addback_energy:time", 
                    dt_window_ns=event_build_window, e_thresh=addback_ethresh, nonlinearity_correction_name=nlc_name)

Eproton_v_tsbo = degai.get_histogram(experiment, runs, adj_dict, cal_name, (200, 0, 0.200, 4000, 0, 4000), "Ep_vs_tsbo", "proton energy (keV) vs time (s)", "tpc_energy:time_since_beam_off", 
                    selection='tpc_particle_id==1',
                    dt_window_ns=event_build_window, e_thresh=addback_ethresh, nonlinearity_correction_name=nlc_name)
Ealpha_v_tsbo = degai.get_histogram(experiment, runs, adj_dict, cal_name, (200, 0, 0.200, 700, 2000, 9000), "Ep_vs_tsbo", "alpha energy (keV) vs time (s)", "tpc_energy:time_since_beam_off", 
                    selection='tpc_particle_id==2',
                    dt_window_ns=event_build_window, e_thresh=addback_ethresh, nonlinearity_correction_name=nlc_name)

# E_v_tsco = degai.get_histogram(experiment, runs, adj_dict, cal_name, (2000, 0, 0.200, *coincidence_binning), 
#                     "E_vs_t_c", "energy (keV) vs time since chopper off (s)", "addback_energy:time_since_chopper_off", 
#                     dt_window_ns=event_build_window, e_thresh=addback_ethresh, nonlinearity_correction_name=nlc_name)

fit_results = []
def fit_decay_curve(Egate, tgate, Egate_bg=None, source=E_v_tsbo, nexp=1):
    if Egate_bg is None:
        h_to_fit = degai.get_gated_projection(source, Egate)
    else:
        h_to_fit = degai.get_bg_subtracted_projection(source, Egate, Egate_bg)
    
    fit_terms = []
    init_vals = []
    lims = []
    names = []
    for i in range(nexp):
        fit_terms.append(f'[{2*i}]*exp(-0.693147*x/[{2*i+1}])')
        init_vals.extend([np.random.rand(), np.random.rand()])
        lims.extend([(0, 1e9), (0., 3600)])
        if nexp == 1:
            names.extend(['A', 'half life (s)'])
        else:
            names.extend([f'A_{i}', f'half life {i} (s)'])
            
    fit_terms.append(f'[{2*nexp}]')
    fit_string = ' + '.join(fit_terms)
    
    init_vals.append(0)
    lims.append((0, h_to_fit.GetMaximum()))
    names.append('bg')

    if Egate_bg is None:
        fit_options = 'S0QEIL'
    else:
        fit_options = 'S0QEI'
    res = fitting_tools.fit_hist(h_to_fit, fit_string, init_vals,
                        lims, tgate,
                        fit_options=fit_options, names=names)
    fit_results.append(res)
    
    fit_ptr = res[0]
    for i in range(nexp):
        hl = fit_ptr.Parameter(2*i+1)
        lower_err = fit_ptr.LowerError(2*i+1)
        upper_err = fit_ptr.UpperError(2*i+1)
        if nexp == 1:
            print(f"Half-life for Egate {Egate}: {hl:.4g} ({lower_err:.4g}/+{upper_err:.4g}) s. Bounds: [{hl + lower_err:.4g}, {hl + upper_err:.4g}] s")
        else:
            print(f"Half-life {i} for Egate {Egate}: {hl:.4g} ({lower_err:.4g}/+{upper_err:.4g}) s. Bounds: [{hl + lower_err:.4g}, {hl + upper_err:.4g}] s")


f_all = fit_peaks(gamma_hist, all_peaks, 'all_gamma', False, True, manual_bounds=True, force_refit=force_refit)
if gamma_beam_off_hist is not None:
    f_beam_off = fit_peaks(gamma_beam_off_hist, all_peaks, 'beam_off_gamma', False, True, manual_bounds=True, force_refit=force_refit)
if gamma_beam_on_hist is not None:
    f_beam_on = fit_peaks(gamma_beam_on_hist, all_peaks, 'beam_on_gamma', False, True, manual_bounds=True, force_refit=force_refit)
if fit_prefix == '60Ga':
#     possible_coincidence_peaks = [511, 546, 1003, 1028, 1188, 1202,  1333, 1341, 1413, 1482, 1554, 2007,
#                 2293, 2334, 2390, 2435, 2484, 2507, 2826, 2996, 
#             3337, 3378, 3588, 3848, 3888, 4177, 4208, 4719, 4806]
    possible_coincidence_peaks = [1004, 1021, 1028, 1188, 1202, 1340, 1398, 1413, 1441,
        1481, 1554, 1780, 2007, 2047, 2092, 2293, 2333, 2390, 2433,
        2507, 2557, 2623, 2633, 2825, 2882, 2996, 3335, 3393, 3781,
        3847, 3888, 4000, 4179, 4208, 4293, 4538, 4719, 4786, 4804,
        4852, 4892, 5299, 5560, 5809
    ]
    force_refit=False
    h338 = degai.get_bg_subtracted_projection(coincidence_hist, (337, 340), (315,322))
    f338=fit_peaks(h338, possible_coincidence_peaks,
            '338keV_coincidence', True, False,force_refit=force_refit)



    h1003 = degai.get_bg_subtracted_projection(coincidence_hist, (1002.0, 1005.0), (1009, 1011))
    f1003=fit_peaks(h1003, possible_coincidence_peaks,
            '1003keV_coincidence', True, False, force_refit=force_refit)
        
    h1003_diff_bg = degai.get_bg_subtracted_projection(coincidence_hist, (1002.0, 1005.0), (1056,1068))
    f1003_diff_bg=fit_peaks(h1003_diff_bg, possible_coincidence_peaks,
                '1003keV_coincidence_diff_bg', True, False, force_refit=force_refit)
    #fit_decay_curve((1002, 1005), (0.005, 0.095), (1056,1068))
    

    h1028 = degai.get_bg_subtracted_projection(coincidence_hist, (1027, 1029), (1038, 1042))
    f1028=fit_peaks(h1028, possible_coincidence_peaks,
            '1028keV_coincidence', True, False,force_refit=force_refit)
    
    h1028_diff_bg = degai.get_bg_subtracted_projection(coincidence_hist, (1027, 1029), (1056,1068))
    f1028_diff_bg=fit_peaks(h1028_diff_bg, possible_coincidence_peaks,
            '1028keV_coincidence_diff_bg', True, False,force_refit=force_refit)

    h1189 = degai.get_bg_subtracted_projection(coincidence_hist, (1188, 1190),(1194,1198))
    f1189=fit_peaks(h1189, possible_coincidence_peaks,
            '1189keV_coincidence', True, False,force_refit=force_refit)

    h1202 = degai.get_bg_subtracted_projection(coincidence_hist, (1200,1204),(1210,1213))
    f1202 = fit_peaks(h1202, possible_coincidence_peaks,
            '1202keV_coincidence', True, False,force_refit=force_refit)


    #1340 is actually a 59Cu peak that is not interesting to us. But 2507 has a coincidence with 1004 and 1340, and 1340 has coincidence with 1004.
    # h1340 = degai.get_bg_subtracted_projection(coincidence_hist, (1339, 1342), (1345, 1365))
    # f1340 = fit_peaks(h1340, possible_coincidence_peaks,
    #         '1340keV_coincidence', True, False,force_refit=force_refit)

    h1413 = degai.get_bg_subtracted_projection(coincidence_hist, (1413, 1416), (1417, 1425))
    f1413 = fit_peaks(h1413, possible_coincidence_peaks,
            '1413keV_coincidence', True, False,force_refit=force_refit) 
    h1413_ab = degai.get_bg_subtracted_projection(ab_coincidence_hist, (1413, 1416), (1417, 1425))
    f1413_ab = fit_peaks(h1413_ab, possible_coincidence_peaks,
            '1413keV_coincidence_ab', True, False,force_refit=force_refit)

    h1441 = degai.get_bg_subtracted_projection(coincidence_hist, (1440,1444), (1420, 1430))
    f1441 = fit_peaks(h1441, possible_coincidence_peaks,
            '1441keV_coincidence', True, False,force_refit=force_refit)

    h1481 = degai.get_bg_subtracted_projection(coincidence_hist, (1480, 1483), (1468, 1478))
    f1481 = fit_peaks(h1481, possible_coincidence_peaks,
            '1481keV_coincidence', True, False,force_refit=force_refit)

    #fit_decay_curve((2006,2009), (0,0.095), (2018, 2038))
    h2007 = degai.get_bg_subtracted_projection(coincidence_hist, (2006, 2009), (2018, 2038))
    f2007=fit_peaks(h2007, possible_coincidence_peaks,
            '2007keV_coincidence', True, False,force_refit=force_refit)

    h2293 = degai.get_bg_subtracted_projection(coincidence_hist, (2291, 2295), (2297,2314))
    f2293 = fit_peaks(h2293, possible_coincidence_peaks,
            '2293keV_coincidence', True, False,force_refit=force_refit)

    h2390 = degai.get_bg_subtracted_projection(coincidence_hist, (2389, 2393), (2321, 2329))
    f2390 = fit_peaks(h2390, possible_coincidence_peaks,
            '2390keV_coincidence', True, False,force_refit=force_refit)
    #fit_decay_curve((2389, 2393),(0.005,0.095), (2321, 2329))
    

    h2433 = degai.get_bg_subtracted_projection(coincidence_hist, (2431, 2437), (2421, 2429))
    f2433 = fit_peaks(h2433, possible_coincidence_peaks,
            '2433keV_coincidence', True, False,force_refit=force_refit)


    h2507 = degai.get_bg_subtracted_projection(coincidence_hist, (2504, 2509), (2535, 2551))
    f2507 = fit_peaks(h2507, possible_coincidence_peaks,
            '2507keV_coincidence', True, False,force_refit=force_refit)
    fit_decay_curve((2504, 2509), (0.005, 0.095), (2535, 2551))

    

    h2825 = degai.get_bg_subtracted_projection(coincidence_hist, (2822,2829), (2831,2860))
    f2825 = fit_peaks(h2825, possible_coincidence_peaks,
            '2825keV_coincidence', True, False,force_refit=force_refit)
    #fit_decay_curve((2822,2829),(0.005,0.095), (2831,2860))

    h2996 = degai.get_bg_subtracted_projection(coincidence_hist, (2993, 3000), (3044,3076))
    f2996 = fit_peaks(h2996, possible_coincidence_peaks,
            '2996keV_coincidence', True, False,force_refit=force_refit)


    h3337 = degai.get_bg_subtracted_projection(coincidence_hist, (3334, 3339), (3346, 3370))
    f3337 = fit_peaks(h3337, possible_coincidence_peaks,
            '3337keV_coincidence', True, False,force_refit=force_refit)


    h3781 = degai.get_bg_subtracted_projection(coincidence_hist, (3776, 3785), (3788, 3818))
    f3781 = fit_peaks(h3781, possible_coincidence_peaks,
            '3781keV_coincidence', True, False,force_refit=force_refit)
    #fit_decay_curve((3776, 3785),(0.005,0.095), (3788, 3818))


    h3847 = degai.get_bg_subtracted_projection(coincidence_hist, (3844, 3852), (3855, 3880))
    f3847 = fit_peaks(h3847, possible_coincidence_peaks,
            '3847keV_coincidence', True, False,force_refit=force_refit)
    #fit_decay_curve((3844, 3852),(0.005,0.095), (3855, 3880))


    h3888 = degai.get_bg_subtracted_projection(coincidence_hist, (3883, 3891), (3900, 3980))
    f3888 = fit_peaks(h3888, possible_coincidence_peaks,
            '3888keV_coincidence', True, False,force_refit=force_refit)
    #fit_decay_curve((3883, 3891),(0.005,0.095), (3900, 3980))

    h4000 = degai.get_bg_subtracted_projection(coincidence_hist, (3996, 4004), (4030, 4098))
    f4000 = fit_peaks(h4000, possible_coincidence_peaks,
            '4000keV_coincidence', True, False,force_refit=force_refit)
    #fit_decay_curve((3996, 4004),(0.005,0.095), (4030, 3098))

    h4179 = degai.get_bg_subtracted_projection(coincidence_hist, (4175, 4180), (4213, 4255))
    f4179 = fit_peaks(h4179, possible_coincidence_peaks,
            '4179keV_coincidence', True, False,force_refit=force_refit)
    #fit_decay_curve((4175, 4180), (0.005,0.095), (4213, 4255))

    h4189 = degai.get_bg_subtracted_projection(coincidence_hist, (4187, 4193),(4213, 4255))
    f4189 = fit_peaks(h4189, possible_coincidence_peaks,
            '4189keV_coincidence', True, False,force_refit=force_refit)
    #fit_decay_curve((4187, 4193),(0.005,0.095), (4213, 4255))

    h4208 = degai.get_bg_subtracted_projection(coincidence_hist, (4205, 4211), (4213, 4255))
    f4208 = fit_peaks(h4208, possible_coincidence_peaks,
            '4208keV_coincidence', True, False,force_refit=force_refit)
    #fit_decay_curve((4205, 4211), (0.005,0.095), (4213, 4255))

    h4259 = degai.get_bg_subtracted_projection(coincidence_hist, (4257, 4265), (4213, 4255))
    f4259 = fit_peaks(h4259, possible_coincidence_peaks,
            '4259keV_coincidence', True, False,force_refit=force_refit)
    #fit_decay_curve((4257, 4265), (0.005,0.095), (4213, 4255))


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


    
    
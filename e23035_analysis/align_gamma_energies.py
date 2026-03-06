import os

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


def do_gain_match(ddas_run, clover):
    '''
    ddas_run
    clover: 1a, 1b, etc
    '''
    true_locations = [510.99895069, 2614.511]
    true_location_uncertainties =  [16e-7, 1e-2]
    peaks = []
    clover_str = f'clover_{clover}'
    init_slope, init_offset = init_cal_dict[clover_str]
    adc_str = f'clover_{clover}_c'
    for i in range(len(true_locations)):
        loc_guess = (true_locations[i] - init_offset)/init_slope
        loc_wiggle = 30/init_slope
        window_width = 50/init_slope
        #(true energy, true energy_uncertainty, location guess, location_wiggle, fit range start, fit range stop)
        peaks.append((true_locations[i], true_location_uncertainties[i],  loc_guess-window_width, loc_guess+window_width))
    return energy_calibration_tools.make_energy_calibration(ddas_run, f'ddas{ddas_run}_clover{clover}_gm', adc_str, (2**16, 0, 2**16), peaks)

do_gain_match(126, '3a')    
#h = ddas_interface.get_histogram(runs[0], (2**16, 0, 2**16), '3a_counts', '3a_counts', 'clover_3a_c')
#guess = 3295.8#(511-2.609910)/0.154992 #9405
# fit_res, background, peaks, rp, canvas, spectrum_to_plot, f_to_fit, h_fit = fitting_tools.fit_emg_peak(h, 'gamma_adc',guess, 100, (-25,25))
# fit_res2, background2, peaks2, rp2, canvas2, spectrum_to_plot2, f_to_fit2, h_fit2=fitting_tools.fit_gaussian_peaks(h, [guess],20,(guess-25, guess+25) ,True, background_type='constant')

# crystal_m_hists = ddas_interface.get_crystal_histograms(runs, (10, -0.5, 9.5), 'm')
# root_vis_tools.create_2d_hist_from_dict(crystal_m_hists, "crystal multiplicities")

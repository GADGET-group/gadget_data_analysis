import os

import numpy as np
import ROOT

from raw_viewer import ddas_interface
from e23035_analysis import fitting_tools, root_vis_tools, e23035_runs

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
runs=[126]

background_run = 91

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



def make_energy_calibration(ddas_run, calibration_name:str, branch_name:str, binning_for_fit:tuple, peaks:list, selection_string='', data_source='gamma_adc'):
    '''
    Fit peaks to get energy calibraiton

    ddas_run: single run or iterable
    calibration_name: name by which the generated slope/offset will be saved and may be retrieved
    branch_name: name of the branch to retrieve data from (eg clover_1a_m, tpc_energy, etc)
    binning_for_fit: tuple specifying TH1D binning
    peaks: List of peaks to use to specify the calibration. Each list entry should contain tuples of 
        (true energy, true energy, true energy_uncertainty, location guess, location_wiggle, fit range start, fit range stop). Will assume
        offset = 0 if length is 1.
    '''
    try:
        num_workers = min(200, len(ddas_runs))
    except ValueError:
        num_workers = 1
    hist_to_fit = ddas_interface.get_histogram(ddas_run, binning_for_fit, branch_name, branch_name, branch_name, selection_string, num_workers)


    #make folder in e23035_analysis/calibrations/calibration name
    #TODO

    #fit peaks
    true_energies, true_energy_uncertainties, peak_locations , peak_location_uncertainty = [], [], [], []
    for true_energy, location_guess, location_wiggle, fit_range_start, fit_range_stop in peaks:
        true_energies.append(true_energy)
        true_energy_uncertainties.append(true_energy_uncertainty)
        fit_res, background, peaks, rp, canvas, spectrum_to_plot, f_to_fit, h_fit = fitting_tools.fit_emg_peak(hist_to_fit, data_source, location_guess, location_wiggle, (fit_range_start, fit_range_stop))
    
    #TODO:save the fit plot to a pdf or image file, also including parameter fit results and p-value
    
    #TODO: add the fit peak location (mu) to peak_locations

    #TODO: fit line to peak locations. Save this to a pkl file



    
def get_energy_calibraiton(ddas_run, calibration_name):
    pass


add_back_gammas = ddas_interface.get_histogram(runs, gamma_binning, 'add_back_gammas', 'add back spectrum', ddas_interface.get_add_back_gamma_str(), num_workers=n_workers)
crystal_e_hists = ddas_interface.get_crystal_histograms(runs, gamma_binning, 'e')
summed_gamma_spectrum = ROOT.TH1D('summed_gamma', 'summed spectrum', *gamma_binning)
for crystal in crystal_e_hists:
    summed_gamma_spectrum.Add(crystal_e_hists[crystal])
summed_gamma_spectrum.SetLineColor(ROOT.kRed)
# summed_gamma_spectrum.Draw()
# add_back_gammas.Draw('SAME')
# ROOT.gPad.SetLogy(1)

h = ddas_interface.get_histogram(runs[0], (24000, 0, 24000), '3a_counts', '3a_counts', 'clover_3a_c')
guess = 3295.8#(511-2.609910)/0.154992 #9405
fit_res, background, peaks, rp, canvas, spectrum_to_plot, f_to_fit, h_fit = fitting_tools.fit_emg_peak(h, 'gamma_adc',guess, 100, (-25,25))
fit_res2, background2, peaks2, rp2, canvas2, spectrum_to_plot2, f_to_fit2, h_fit2=fitting_tools.fit_gaussian_peaks(h, [guess],20,(guess-25, guess+25) ,True, background_type='constant')

crystal_m_hists = ddas_interface.get_crystal_histograms(runs, (10, -0.5, 9.5), 'm')
root_vis_tools.create_2d_hist_from_dict(crystal_m_hists, "crystal multiplicities")

summed_multiplicities = ROOT.TH1D('summed_gamma_mult', 'crystal multiplicities', 10, -0.5, 9.5)
for crystal in crystal_m_hists:
    summed_multiplicities.Add(crystal_m_hists[crystal])

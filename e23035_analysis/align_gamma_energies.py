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

def make_energy_calibration(ddas_run, branch_name, binning_for_fit, peak1, peak2=[], selection_string=''):
    '''
    Fit peaks to get energy calibraiton
    Each peak should be a list containing [true energy, guess for location, +/- bounds to use when fitting the peak]
    If peak2 is an empty list, offset will be assumed to be 0.
    '''
    hist_to_fit = ddas_interface.get_histogram(ddas_run, binning_for_fit, branch_name, branch_name, branch_name, selection_string)
    
def save_energy_calibration(run, calibration):
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
guess = 9405
fit_range = (guess-15, guess+25)
fit_res, background, peaks, rp, canvas, spectrum_to_plot, f_to_fit, h_fit = fitting_tools.fit_emg_peak(h, 'gamma_adc',guess, 100, (-15,25))
fit_res2, background2, peaks2, rp2, canvas2, spectrum_to_plot2, f_to_fit2, h_fit2=fitting_tools.fit_gaussian_peaks(h, [guess],20,fit_range ,True, background_type='constant')


#background_gammas = ddas_interface.get_histogram(background_run, gamma_binning, 'gammas', 'summed gamma spectrum', ddas_interface.get_summed_gamma_e_str(), num_workers=n_workers)

#background_crystal_hists = ddas_interface.get_crystal_histograms(background_run, gamma_binning)

# c1, h1 = root_vis_tools.plot_crystal_vs_time(background_run, '3c')
# c2, h2 = root_vis_tools.plot_crystal_vs_time(runs[0], '3c')
# c3, h3 = root_vis_tools.plot_crystal_vs_time(runs[0], '5a')
# c4, h4 = root_vis_tools.plot_crystal_vs_time(runs[0], '7b')

#canvas, legend, stack = root_vis_tools.draw_overlaid_histograms(crystal_hists)
#canvas, legend, stack = root_vis_tools.draw_overlaid_histograms(background_crystal_hists)
#ROOT.gPad.SetLogy(1)

# canvas, legend, stack = root_vis_tools.draw_overlaid_histograms({'background run 91':background_gammas, '60Ga run 235':gammas},x_label='keV')
#fit_res, background, peaks, rp, canvas, spectrum_to_plot, f_to_fit, h_fit=fitting_tools.fit_gaussian_peaks(crystal_hists['clover_3c_e'], [511],30,(490,530) ,True, background_type='constant')
#


# canvas, th2 = root_vis_tools.create_2d_hist_from_dict(crystal_hists)
# ROOT.gPad.SetLogz(1)

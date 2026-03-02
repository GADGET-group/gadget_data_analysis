import ROOT

from raw_viewer import ddas_interface
from e23035_analysis import fitting_tools, root_vis_tools

'''
Notes on peak finding an automatic fitting
https://root.cern/root/htmldoc/guides/spectrum/Spectrum.html
'''
n_workers=10
gamma_binning = (12000-1,1,12000)
runs=235

def make_energy_calibration(ddas_run, branch_name, binning_for_fit, peak1, peak2=[], selection_string=''):
    '''
    Fit peaks to get energy calibraiton
    Each peak should be a list containing [true energy, guess for location, +/- bounds to use when fitting the peak]
    If peak2 is an empty list, offset will be assumed to be 0.
    '''
    hist_to_fit = ddas_interface.get_histogram(ddas_run, binning_for_fit, branch_name, branch_name, branch_name, selection_string)
    
def save_energy_calibration(run, calibration):
    pass

gammas = ddas_interface.get_histogram(runs, gamma_binning, 'gammas', 'summed gamma spectrum', ddas_interface.get_summed_gamma_e_str(), num_workers=n_workers)
crystal_hists = ddas_interface.get_crystal_histograms(runs, gamma_binning)
c,l = root_vis_tools.draw_overlaid_histograms(crystal_hists)
ROOT.gPad.SetLogy(1)

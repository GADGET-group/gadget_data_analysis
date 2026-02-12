import os

import ROOT
import numpy as np

from e23035_analysis import e23035_runs

import uuid # Import this to generate unique IDs

import ROOT
import numpy as np
import uuid

import ROOT
import numpy as np
import uuid

import ROOT
import numpy as np
import uuid

import ROOT
import numpy as np
import uuid

def fit_peaks(spectrum, energy_guesses, energy_wiggle, energy_window): 
    # 1. Unique ID
    unique_id = uuid.uuid4().hex[:8]
    
    # 2. Canvas
    canvas_name = f"c_fit_{unique_id}"
    canvas = ROOT.TCanvas(canvas_name, f"Fit {energy_guesses[0]:.1f} keV", 800, 600)

    # 3. Create Subset Histogram (The "Cut")
    #    We calculate the exact range and binning to match the original
    e_low = energy_guesses[0] - energy_window
    e_high = energy_guesses[-1] + energy_window
    
    bin_width = spectrum.GetBinWidth(1)
    # Calculate number of bins in this window (rounding safe)
    n_bins_new = int((e_high - e_low) / bin_width + 0.5)
    
    # Create the fresh histogram
    spectrum_to_plot = ROOT.TH1D(f"sub_{unique_id}", "Data vs Fit", n_bins_new, e_low, e_high)
    
    # Copy data from the original spectrum to the subset
    for i in range(1, n_bins_new + 1):
        center = spectrum_to_plot.GetBinCenter(i)
        source_bin = spectrum.FindBin(center)
        
        content = spectrum.GetBinContent(source_bin)
        error = spectrum.GetBinError(source_bin)
        
        spectrum_to_plot.SetBinContent(i, content)
        spectrum_to_plot.SetBinError(i, error)

    # 4. String Construction
    peaks_string = ''
    n_peaks = len(energy_guesses)
    for i in range(n_peaks):
        if i > 0: peaks_string += ' + '
        sigma_string = '(0.011107*[%d] + 0.008813049)' % (2*i+1) 
        peaks_string += '[%d]*exp(-0.5*((x-[%d])/%s)^2)/(%s *sqrt(2*pi))*%f' % (
            2*i, 2*i+1, sigma_string, sigma_string, bin_width
        )
    
    bg_idx_1 = 2 * n_peaks
    bg_idx_2 = 2 * n_peaks + 1
    background_string = '[%d] + [%d]*x' % (bg_idx_1, bg_idx_2)
    function_string = background_string + ' + ' + peaks_string
    
    # 5. Fit Function Setup
    func_name = f'to_fit_{unique_id}'
    f_to_fit = ROOT.TF1(func_name, function_string, e_low, e_high)
    f_to_fit.SetParLimits(0, 0, np.inf)
    
    for i in range(n_peaks):
        f_to_fit.SetParameter(2*i, 100) 
        f_to_fit.SetParLimits(2*i, 0, np.inf)
        f_to_fit.SetParName(2*i, f'A_{i}')
        f_to_fit.SetParameter(2*i+1, energy_guesses[i])
        f_to_fit.SetParLimits(2*i+1, energy_guesses[i] - energy_wiggle, energy_guesses[i] + energy_wiggle)
        f_to_fit.SetParName(2*i+1, f'mu_{i}')

    f_to_fit.SetNpx(1000) 

    # 6. Perform Fit
    # "S": Return Result, "0": Do not draw, "Q": Quiet
    fit_res = spectrum_to_plot.Fit(f_to_fit, "S0") 
    if not fit_res.IsValid():
        fit_res = spectrum_to_plot.Fit(f_to_fit, "S0")

    # --- Convert Function to Histogram for Residual Plot ---
    h_fit = spectrum_to_plot.Clone(f"h_fit_{unique_id}")
    h_fit.Reset() 
    for i in range(1, h_fit.GetNbinsX() + 1):
        val = f_to_fit.Eval(h_fit.GetBinCenter(i))
        h_fit.SetBinContent(i, val)
    h_fit.SetLineColor(ROOT.kRed)
    h_fit.SetLineWidth(2)

    # 7. Draw TRatioPlot
    canvas.cd()
    # (Data, Fit) -> Residuals = Data - Fit
    rp = ROOT.TRatioPlot(spectrum_to_plot, h_fit, "diff")
    
    rp.SetH1DrawOpt("E")      # Data: Points w/ Error
    rp.SetH2DrawOpt("L")      # Fit: Line
    rp.SetGraphDrawOpt("P")   # Residuals: Points
    
    rp.Draw()

    # Style Residuals
    rp.GetLowerRefYaxis().SetTitle("Resid. (Data-Fit)")
    rp.GetLowerRefGraph().SetMarkerStyle(20)
    rp.GetLowerRefGraph().SetMarkerSize(0.6)
    rp.GetLowerPad().SetGridy()
    
    canvas.Update()

    # 8. Return
    fit_params = np.array(fit_res.Parameters())
    bg_name = f'background_{unique_id}'
    background = ROOT.TF1(bg_name, background_string, e_low, e_high)
    background.SetParameters(fit_params[-2:])
    
    pk_name = f'peaks_{unique_id}'
    peaks = ROOT.TF1(pk_name, peaks_string, e_low, e_high)
    peaks.SetParameters(fit_params[:-2])
    
    return fit_res, background, peaks, rp, canvas, spectrum_to_plot, f_to_fit, h_fit

def fit_multiple_peaks_sigma_free(spectrum, energy_guesses, energy_wiggle, energy_window):
    function_string = '[0] + [1]*x'
    for i in range(len(energy_guesses)):
        function_string += ' + [%d]*exp(-0.5*((x-[%d])/[2])^2)/([2] *sqrt(2*pi))*%f'%(2*i+3, 2*i+4, spectrum.GetBinWidth(0))
    #use [2] for sigma
    f_to_fit = ROOT.TF1('to_fit', function_string, energy_guesses[0] - energy_window, energy_guesses[-1]+energy_window)
    f_to_fit.SetParLimits(0, 0, np.inf)
    for i in range(len(energy_guesses)):
        f_to_fit.SetParameter(2*i+3, 100) #magnitude
        f_to_fit.SetParLimits(2*i+3,0,np.inf)
        f_to_fit.SetParName(2*i+3, 'A_%d'%i)

        f_to_fit.SetParameter(2*i+4, energy_guesses[i])
        f_to_fit.SetParLimits(2*i+4,energy_guesses[i] - energy_wiggle,energy_guesses[i] + energy_wiggle)
        f_to_fit.SetParName(2*i+4, 'mu_%d'%i)

    f_to_fit.SetParameter(2, 0.05)
    f_to_fit.SetParLimits(2,0.005, 0.5)
    f_to_fit.SetParName(2, 'sigma')
    f_to_fit.SetNpx(3000)

    f_to_fit.SetNpx(3000)
    done = False
    while not done:
        fit_res = spectrum.Fit(f_to_fit, "LRS")
        done = fit_res.IsValid()
    #spectrum.Draw()
    rp = ROOT.TRatioPlot(spectrum)
    rp.Draw()
    fit_res.Print()
    return fit_res, rp

def get_residuals(histogram, function:ROOT.TF1, residual_hist_name=None, plot=True):
    '''
    Subtract function from histogram and return residuals histogram.
    If plot is True, the original histogram will be plotted with the function
    above the residuals.
    Residuals histogram will use the bin spacing of the original histogram but only cover the 
    portion of the histogram that is on top of the function.
    '''
    if type(residual_hist_name) == type(None):
        residual_hist_name = histogram.GetName() + '_residuals'
    xmin, xmax = function.GetXmin(), function.GetXmax()
    first_bin, last_bin = histogram.FindBin(xmin), histogram.FindBin(xmax)
    residuals = []
    for i in range(first_bin, last_bin+1):
        bin_center  = histogram.GetBinCenter(i)
        residuals.append()
  
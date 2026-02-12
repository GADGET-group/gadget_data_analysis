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

def fit_peaks(spectrum, energy_guesses, energy_wiggle, energy_window): 
    # 1. Unique ID
    unique_id = uuid.uuid4().hex[:8]
    
    # 2. Canvas
    canvas_name = f"c_fit_{unique_id}"
    canvas = ROOT.TCanvas(canvas_name, f"Fit {energy_guesses[0]:.1f} keV", 800, 600)

    # 3. Clone and Zoom (Data Histogram)
    spectrum_to_plot = spectrum.Clone(f"spectrum_zoom_{unique_id}")
    e_low = energy_guesses[0] - energy_window
    e_high = energy_guesses[-1] + energy_window
    spectrum_to_plot.GetXaxis().SetRangeUser(e_low, e_high)
    spectrum_to_plot.SetTitle("Data vs Fit")

    # 4. String Construction
    peaks_string = ''
    n_peaks = len(energy_guesses)
    for i in range(n_peaks):
        if i > 0: peaks_string += ' + '
        sigma_string = '(0.011107*[%d] + 0.008813049)' % (2*i+1) 
        peaks_string += '[%d]*exp(-0.5*((x-[%d])/%s)^2)/(%s *sqrt(2*pi))*%f' % (
            2*i, 2*i+1, sigma_string, sigma_string, spectrum.GetBinWidth(1)
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
    fit_res = spectrum_to_plot.Fit(f_to_fit, "S0Q") 
    if not fit_res.IsValid():
        fit_res = spectrum_to_plot.Fit(f_to_fit, "S0Q")

    # --- NEW: Convert Function to Histogram for Residual Plot ---
    # We clone the data histogram structure to ensure bins match exactly
    h_fit = spectrum_to_plot.Clone(f"h_fit_{unique_id}")
    h_fit.Reset() # Clear it
    
    # Fill h_fit with the function values
    for i in range(1, h_fit.GetNbinsX() + 1):
        bin_center = h_fit.GetBinCenter(i)
        val = f_to_fit.Eval(bin_center)
        h_fit.SetBinContent(i, val)
        h_fit.SetBinError(i, 0) # The fit curve itself has no statistical error in this view
    
    # Style the "Fit" histogram to look like a red line
    h_fit.SetLineColor(ROOT.kRed)
    h_fit.SetLineWidth(2)

    # 7. Draw TRatioPlot with "diff" option
    canvas.cd()
    
    # "diff" calculates (h1 - h2). 
    # We pass (spectrum, h_fit) so we get (Data - Fit).
    rp = ROOT.TRatioPlot(spectrum_to_plot, h_fit, "diff")
    
    rp.SetH1DrawOpt("E")      # Draw Data as Points w/ Errors
    rp.SetH2DrawOpt("L")      # Draw Fit as Line
    rp.SetGraphDrawOpt("P")   # Draw Residuals as Points
    
    rp.Draw()

    # --- Style the Residual Axis ---
    # Centering around 0 usually looks best for residuals
    rp.GetLowerRefYaxis().SetTitle("Residuals (Data - Fit)")
    rp.GetLowerRefGraph().SetMarkerStyle(20)
    rp.GetLowerRefGraph().SetMarkerSize(0.6)
    
    # Optional: Draw a dashed line at 0
    ref_line = ROOT.TLine()
    ref_line.SetLineStyle(2)
    # We need to draw the line after updating the canvas coordinates,
    # but TRatioPlot manages the pads tricky-ly. 
    # Often simpler to just grid the lower pad:
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
    
    # Return h_fit as well!
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
  
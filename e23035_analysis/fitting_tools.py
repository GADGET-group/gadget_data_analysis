import os
import uuid

import ROOT
import numpy as np

from e23035_analysis import e23035_runs

import ROOT
import numpy as np
import uuid

def fit_func(histogram, function_string, initial_values, bounds, fit_range, names=None): 
    """
    Fits a user-defined function to a histogram using Poisson statistics (Log-Likelihood) 
    and plots the residuals.
    
    Args:
        histogram: The ROOT TH1 object to fit.
        function_string (str): The ROOT TF1 string (e.g., '[0] + [1]*x + [2]*exp(-0.5*((x-[3])/[4])^2)').
        initial_values (list): Initial guesses for parameters [val0, val1, ...].
        bounds (list of tuples): Limits for each parameter [(min0, max0), (min1, max1), ...].
        fit_range (tuple): The (x_min, x_max) range to perform the fit.
        names (list of str, optional): Names for each parameter. Defaults to None (p0, p1, ...).
    """
    # 1. Unique ID & Setup
    unique_id = uuid.uuid4().hex[:8]
    canvas_name = f"c_fit_{unique_id}"
    canvas = ROOT.TCanvas(canvas_name, f"Fit Result: {function_string}", 800, 600)

    e_low, e_high = fit_range

    # 2. Create Subset Histogram (The "Cut" for a clean plot)
    bin_width = histogram.GetBinWidth(1)
    n_bins_new = int((e_high - e_low) / bin_width + 0.5)
    
    sub_hist = ROOT.TH1D(f"sub_{unique_id}", "Data vs Fit", n_bins_new, e_low, e_high)
    
    for i in range(1, n_bins_new + 1):
        center = sub_hist.GetBinCenter(i)
        source_bin = histogram.FindBin(center)
        sub_hist.SetBinContent(i, histogram.GetBinContent(source_bin))
        sub_hist.SetBinError(i, histogram.GetBinError(source_bin))

    # 3. Fit Function Setup
    func_name = f'to_fit_{unique_id}'
    f_to_fit = ROOT.TF1(func_name, function_string, e_low, e_high)
    
    # Ensure inputs match
    n_params = len(initial_values)
    if len(bounds) != n_params:
        raise ValueError("Length of initial_values must match length of bounds.")
    if names is not None and len(names) != n_params:
        raise ValueError("Length of names must match length of initial_values.")
    
    # Apply initial values, limits, and names
    for i in range(n_params):
        f_to_fit.SetParameter(i, initial_values[i])
        f_to_fit.SetParLimits(i, bounds[i][0], bounds[i][1])
        
        if names:
            f_to_fit.SetParName(i, names[i])
        else:
            f_to_fit.SetParName(i, f'p{i}')

    f_to_fit.SetNpx(1000)

    # 4. Perform Fit ("L" for Log-Likelihood / Poisson statistics)
    fit_options = 'LS0QEG'
    fit_res = sub_hist.Fit(f_to_fit, fit_options)
    attempts = 0
    while not fit_res.IsValid() and attempts < 20:
        fit_res = sub_hist.Fit(f_to_fit, fit_options)
        attempts += 1

    # 5. Convert Function to Histogram for Residual Plot
    h_fit = sub_hist.Clone(f"h_fit_{unique_id}")
    h_fit.Reset() 
    for i in range(1, h_fit.GetNbinsX() + 1):
        val = f_to_fit.Eval(h_fit.GetBinCenter(i))
        h_fit.SetBinContent(i, val)
        h_fit.SetBinError(i, 0)
        
    h_fit.SetLineColor(ROOT.kRed)
    h_fit.SetLineWidth(2)

    # 6. Draw TRatioPlot (Residuals)
    canvas.cd()
    rp = ROOT.TRatioPlot(sub_hist, h_fit, "diff")
    
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

    return fit_res, rp, canvas, sub_hist, f_to_fit, h_fit

def extract_fit_params(fit_res):
    """
    Extracts parameter names, values, uncertainties, and fit quality metrics 
    from a ROOT TFitResultPtr into a Python dictionary.
    """
    if not fit_res.IsValid():
        print("Warning: The fit result is marked as invalid. Check your initial guesses/bounds.")

    fit_info = {
        'metrics': {
            'is_valid': fit_res.IsValid(),
            'status': fit_res.Status(), # 0 usually means success
            'chi2': fit_res.Chi2(),
            'ndf': fit_res.Ndf(),
            'chi2_over_ndf': fit_res.Chi2() / fit_res.Ndf() if fit_res.Ndf() > 0 else float('inf'),
            'prob': fit_res.Prob()
        },
        'parameters': {}
    }

    n_params = fit_res.NPar()
    for i in range(n_params):
        name = fit_res.GetParName(i)
        val = fit_res.Parameter(i)
        err = fit_res.ParError(i)
        
        fit_info['parameters'][name] = {
            'value': val,
            'error': err
        }

    return fit_info

def get_sigma(data_source, energy):
    if data_source == 'tpc':
        sigma = 0.011107 * energy + 0.008813049 #formula for TPC energy resolution in MeV as a guess
    elif data_source == 'gamma_adc':
        sigma = 10
    elif data_source == 'gamma_keV':
        sigma = 2
    else:
        raise ValueError('unknown data source string%s'%data_source)
    return sigma


def fit_gaussian_peaks(spectrum, energy_guesses, energy_wiggle, energy_window, free_sigma=False, data_source='tpc', background_type='linear'): 
    '''
    
    background_type = none, constant, or linear
    '''
    # Extract the fit range
    e_low, e_high = energy_window[0], energy_window[1]
    bin_width = spectrum.GetBinWidth(1)

    # Prepare lists for the generalized fit_func
    initial_values = []
    bounds = []
    names = []
    
    # 1. String Construction & Parameter Setup for Peaks
    peaks_string = ''
    n_peaks = len(energy_guesses)
    
    # Track how many parameters each peak consumes to keep ROOT indices aligned
    params_per_peak = 3 if free_sigma else 2
    
    for i in range(n_peaks):
        if i > 0: peaks_string += ' + '
        
        # Calculate ROOT parameter indices for this specific peak
        amp_idx = params_per_peak * i
        mean_idx = params_per_peak * i + 1
        
        # Setup Amplitude (A_i)
        initial_values.append(100.0)
        bounds.append((0.0, np.inf))
        names.append(f'A_{i}')
        
        # Setup Mean (mu_i)
        initial_values.append(energy_guesses[i])
        bounds.append((energy_guesses[i] - energy_wiggle, energy_guesses[i] + energy_wiggle))
        names.append(f'mu_{i}')

        # Setup Sigma
        if free_sigma:
            sigma_idx = params_per_peak * i + 2
            sigma_string = f'[{sigma_idx}]'
            sigma_guess = get_sigma(data_source,energy_guesses[i])
            initial_values.append(sigma_guess)
            bounds.append((0.001, np.inf)) # Prevent sigma from hitting exactly 0 (divide by zero error)
            names.append(f'sigma_{i}')
        else:
            # Lock sigma to the formula using the mean parameter's index
            #TODO: fit peaks to get energy dependence for gamma ray detectors
            if data_source == 'tpc':
                sigma_string = f'(0.011107*[{mean_idx}] + 0.008813049)'
            else:
                raise ValueError('invaclid data source for fixed sigma')
            
        # Build the math string using f-strings for readability
        peaks_string += f'[{amp_idx}]*exp(-0.5*((x-[{mean_idx}])/{sigma_string})^2)/({sigma_string} *sqrt(2*pi))*{bin_width}'

    # 2. String Construction & Parameter Setup for Background
    bg_idx_1 = params_per_peak * n_peaks
    bg_idx_2 = params_per_peak * n_peaks + 1
    if background_type == 'linear':
        background_string = f'[{bg_idx_1}] + [{bg_idx_2}]*x'
        initial_values.extend([0.0, 0.0])
        bounds.extend([(-np.inf, np.inf), (-np.inf, np.inf)])
        names.extend(["bg_offset", "bg_slope"])
    elif background_type == 'constant':
        background_string = f'[{bg_idx_1}]'
        initial_values.extend([0])
        bounds.extend([(0, np.inf)])
        names.extend(['bg_offset'])
    elif background_type == 'none':
        background_string = '0'
    else:
        raise ValueError(f'invalid background type {background_type}')

    # Combine strings
    function_string = f'{background_string} + {peaks_string}'

    # 3. Call our generalized fit engine
    fit_res, rp, canvas, spectrum_to_plot, f_to_fit, h_fit = fit_func(
        histogram=spectrum, 
        function_string=function_string, 
        initial_values=initial_values, 
        bounds=bounds, 
        fit_range=energy_window, 
        names=names
    )

    # 4. Reconstruct individual TF1 components for return
    comp_id = uuid.uuid4().hex[:6]
    fit_params = np.array(fit_res.Parameters())
    
    # Background is always the last 2 parameters, regardless of sigma mode
    background = ROOT.TF1(f'bg_{comp_id}', background_string, e_low, e_high)
    background.SetParameters(fit_params[-2:])
    
    # Peaks take all parameters EXCEPT the last 2
    peaks = ROOT.TF1(f'peaks_{comp_id}', peaks_string, e_low, e_high)
    peaks.SetParameters(fit_params[:-2])
    
    return fit_res, background, peaks, rp, canvas, spectrum_to_plot, f_to_fit, h_fit

def fit_emg_peak(spectrum:ROOT.TH1D, data_source:str, e_guess:float, e_wiggle:float, fit_window): 
    """
    Fits an Exponentially Modified Gaussian (low-energy tail) + constant background 
    using the fit_func engine.
    """
    # try:
    #     e_low, e_high = e_guess + min(fit_window), e_guess+max(fit_window)
    # except TypeError: #single number passed in
    #     e_low, e_high = e_guess - fit_window, e_guess + fit_window
    e_low, e_high = fit_window

    # 1. Construct the Mathematical Model
    # [0]: Constant Background
    # [1]: Amplitude (A)
    # [2]: Mean (mu)
    # [3]: Sigma (sigma)
    # [4]: Tail decay parameter (tau)
    
    bg_string = "[0]"
    
    # ROOT's TMath::Erfc and TMath::Exp are used for stability. 
    # 1.41421356 is used in place of sqrt(2) for parsing efficiency.
    bin_width = spectrum.GetBinWidth(0)
    norm_factor = f"([1] * {bin_width} / (2.0 * [4]))"
    exp_term = "TMath::Exp((x-[2])/[4] + ([3]*[3])/(2.0*[4]*[4]))"
    erfc_term = "TMath::Erfc((x-[2])/(1.41421356*[3]) + [3]/(1.41421356*[4]))"
    
    emg_string = f"{norm_factor} * {exp_term} * {erfc_term}"
    function_string = f"{bg_string} + {emg_string}"
    
    function_string = f"{bg_string} + {emg_string}"

    # 2. Setup Parameters and Initial Guesses
    # Calculate a good starting sigma using your empirical formula
    sigma_guess = get_sigma(data_source, e_guess)
    tau_guess = 3
    
    spectrum.GetXaxis().SetRangeUser(*fit_window)

    bg_guess = spectrum.GetBinContent(spectrum.GetXaxis().GetFirst())
    A_guess = (spectrum.GetBinContent(spectrum.GetMaximumBin()) - bg_guess)*sigma_guess/bin_width#35413.4#
    if data_source == 'gamma_adc':
        tau_bounds = (0.01, 10)
        sigma_bounds = (1,20)
        A_bounds = (1,np.inf)#((spectrum.GetBinContent(spectrum.GetMaximumBin()) - bg_guess), np.inf)

    spectrum.GetXaxis().UnZoom()

    initial_values = [
        bg_guess,          # p0: bg_const
        A_guess,        # p1: amplitude
        e_guess,      # p2: mu
        sigma_guess,  # p3: sigma
        tau_guess     # p4: tau (tail length)
    ]
    #print('initial values: ',initial_values)

    bounds = [
        (0, np.inf),                                 # p0: bg_const
        A_bounds,                             # p1: amplitude
        (e_guess - e_wiggle, e_guess + e_wiggle),          # p2: mu bounds
        sigma_bounds,                                   # p3: sigma 
        tau_bounds                                   # p4: tau 
    ]

    #print('bounds: ',bounds)

    names = ["bg_const", "amplitude", "mu", "sigma", "tau"]

    # 3. Call our generalized fit engine
    fit_res, rp, canvas, spectrum_to_plot, f_to_fit, h_fit = fit_func(
        histogram=spectrum, 
        function_string=function_string, 
        initial_values=initial_values, 
        bounds=bounds, 
        fit_range=(e_low, e_high), 
        names=names
    )

    # 4. Reconstruct individual TF1 components for visualization/return
    comp_id = uuid.uuid4().hex[:6]
    fit_params = np.array(fit_res.Parameters())
    
    # Background component (just parameter 0)
    background = ROOT.TF1(f'bg_{comp_id}', bg_string, e_low, e_high)
    background.SetParameter(0, fit_params[0])
    
    # EMG Peak component (parameters 1 through 4)
    peaks = ROOT.TF1(f'emg_{comp_id}', emg_string, e_low, e_high)
    for i in range(1, 5):
        peaks.SetParameter(i, fit_params[i])
        
    return fit_res, background, peaks, rp, canvas, spectrum_to_plot, f_to_fit, h_fit


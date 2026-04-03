import os
import uuid

import ROOT
import numpy as np

from e23035_analysis import e23035_runs

import ROOT
import numpy as np
import uuid

def fit_func(histogram, function_string, initial_values, bounds, fit_range, names=None): 
    # 1. Unique ID & Setup
    unique_id = uuid.uuid4().hex[:8]
    canvas_name = f"c_fit_{unique_id}"
    canvas = ROOT.TCanvas(canvas_name, f"Fit Result: {function_string}", 800, 600)

    e_low, e_high = fit_range

    # Snap e_low and e_high to the exact bin edges of the original histogram
    x_axis = histogram.GetXaxis()
    bin_low = x_axis.FindBin(e_low)
    bin_high = x_axis.FindBin(e_high)
    
    e_low = x_axis.GetBinLowEdge(bin_low)
    e_high = x_axis.GetBinUpEdge(bin_high)
    n_bins_new = bin_high - bin_low + 1

    # 2. Create Subset Histogram (The "Cut" for a clean plot)
    sub_hist = ROOT.TH1D(f"sub_{unique_id}", "Data vs Fit", n_bins_new, e_low, e_high)
    
    for i in range(1, n_bins_new + 1):
        source_bin = bin_low + i - 1
        sub_hist.SetBinContent(i, histogram.GetBinContent(source_bin))
        sub_hist.SetBinError(i, histogram.GetBinError(source_bin))

    # 3. Fit Function Setup
    func_name = f'to_fit_{unique_id}'
    n_params = len(initial_values)
    
    if callable(function_string):
        f_to_fit = ROOT.TF1(func_name, function_string, e_low, e_high, n_params)
        f_to_fit._pyfunc = function_string # Keep a reference to prevent garbage collection
    else:
        f_to_fit = ROOT.TF1(func_name, function_string, e_low, e_high)
        
    # Ensure inputs match
    if len(bounds) != n_params:
        raise ValueError("Length of initial_values must match length of bounds.")
    if names is not None and len(names) != n_params:
        raise ValueError("Length of names must match length of initial_values.")
    
    # Apply initial values, limits, and names
    for i in range(n_params):
        if bounds[i][0] == bounds[i][1]:
            f_to_fit.FixParameter(i, bounds[i][0])
        else:
            f_to_fit.SetParameter(i, initial_values[i])
            f_to_fit.SetParLimits(i, bounds[i][0], bounds[i][1])
        
        if names:
            f_to_fit.SetParName(i, names[i])
        else:
            f_to_fit.SetParName(i, f'p{i}')

    f_to_fit.SetNpx(1000)

    # 4. Perform Fit ("L" for Log-Likelihood / Poisson statistics)
    fit_options = 'LS0QEI'
    fit_res = sub_hist.Fit(f_to_fit, fit_options)
    attempts = 0
    
    # Cleaned up overlapping while loop syntax
    while (not fit_res.Get() or not fit_res.IsValid()) and attempts < 20:
        fit_res = sub_hist.Fit(f_to_fit, fit_options)
        attempts += 1

    # 5. Convert Function to Histogram for Residual Plot
    h_fit = sub_hist.Clone(f"h_fit_{unique_id}")
    h_fit.Reset() 
    for i in range(1, h_fit.GetNbinsX() + 1):
        bin_low = h_fit.GetXaxis().GetBinLowEdge(i)
        bin_high = h_fit.GetXaxis().GetBinUpEdge(i)
        bin_width = h_fit.GetXaxis().GetBinWidth(i)
        h_fit.SetBinContent(i, f_to_fit.Integral(bin_low, bin_high) / bin_width)
        h_fit.SetBinError(i, 0)
        
    h_fit.SetLineColor(ROOT.kRed)
    h_fit.SetLineWidth(2)

    # 6. Draw TRatioPlot (Residuals)
    canvas.cd()
    rp = ROOT.TRatioPlot(sub_hist, h_fit, "diff")
    
    rp.SetH1DrawOpt("E")      
    rp.SetH2DrawOpt("L")      
    rp.SetGraphDrawOpt("P")   
    
    rp.Draw()

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

def fit_emg_peak(spectrum:ROOT.TH1D, data_source:str, e_guess:float, fit_window, param_bounds=None): 
    from scipy.special import erfcx, erfc
    if param_bounds is None:
        param_bounds = {}
    e_low, e_high = fit_window

    bin_width = spectrum.GetBinWidth(1)
    
    # Python evaluation function utilizing erfcx
    def emg_eval(x, p):
        val_x = x[0]
        bg_const = p[0]
        amp = p[1]
        mu = p[2]
        sigma = p[3]
        tau = p[4]
        
        norm = amp * bin_width / (2.0 * tau)
        
        # Unified argument logic
        z_arg = ((val_x - mu) / sigma + sigma / tau) / 1.41421356
        
        if z_arg < -25: 
            # Fallback to stable exp*erfc for extreme negative arguments
            exp_arg = (sigma**2)/(2*tau**2) + (val_x - mu)/tau
            if exp_arg > 700: 
                peak_val = 0.0
            else: 
                peak_val = norm * np.exp(exp_arg) * erfc(z_arg)
        else: 
            # Stable erfcx form for the main peak and tail
            gaus_arg = (val_x - mu) / sigma
            gaus_part = np.exp(-0.5 * gaus_arg * gaus_arg)
            erfcx_part = erfcx(z_arg)
            peak_val = norm * gaus_part * erfcx_part
            
        return bg_const + peak_val

    # Setup Parameters and Initial Guesses
    sigma_guess = get_sigma(data_source, e_guess)
    tau_guess = 3
    
    spectrum.GetXaxis().SetRangeUser(*fit_window)
    bg_guess = spectrum.GetBinContent(spectrum.GetXaxis().GetFirst())
    A_guess = (spectrum.GetBinContent(spectrum.GetMaximumBin()) - bg_guess) * sigma_guess / bin_width
    
    if data_source == 'gamma_adc':
        tau_bounds = (0.01, 10)
        sigma_bounds = (1, 20)
        A_bounds = (1, np.inf)
    else:
        tau_bounds = (0.01, 100)
        sigma_bounds = (0.1, 100)
        A_bounds = (0, np.inf)

    spectrum.GetXaxis().UnZoom()

    initial_values = [bg_guess, A_guess, e_guess, sigma_guess, tau_guess]

    bounds = [
        param_bounds.get('bg_const', (0, np.inf)),
        param_bounds.get('amplitude', A_bounds),
        param_bounds.get('mu', (e_low, e_high)),
        param_bounds.get('sigma', sigma_bounds),
        param_bounds.get('tau', tau_bounds)
    ]

    names = ["bg_const", "amplitude", "mu", "sigma", "tau"]

    # Call fit engine (passing ONLY the python function)
    fit_res, rp, canvas, spectrum_to_plot, f_to_fit, h_fit = fit_func(
        histogram=spectrum, 
        function_string=emg_eval, 
        initial_values=initial_values, 
        bounds=bounds, 
        fit_range=(e_low, e_high), 
        names=names
    )

    comp_id = uuid.uuid4().hex[:6]
    fit_params = np.array([f_to_fit.GetParameter(i) for i in range(f_to_fit.GetNpar())])
    
    # Reconstruct individual TF1 components for visualization
    def bg_eval(x, p):
        return p[0]
        
    background = ROOT.TF1(f'bg_{comp_id}', bg_eval, e_low, e_high, 1)
    background._pyfunc = bg_eval
    background.SetParameter(0, fit_params[0])
    
    def peak_eval(x, p):
        return emg_eval(x, [0, p[1], p[2], p[3], p[4]])

    peaks = ROOT.TF1(f'emg_{comp_id}', peak_eval, e_low, e_high, 5)
    peaks._pyfunc = peak_eval
    for i in range(1, 5):
        peaks.SetParameter(i, fit_params[i])
        
    return fit_res, background, peaks, rp, canvas, spectrum_to_plot, f_to_fit, h_fit

def fit_gaussian_peak(spectrum:ROOT.TH1D, data_source:str, e_guess:float, fit_window, param_bounds=None): 
    """
    Fits a standard Gaussian + constant background using the fit_func engine.
    """
    if param_bounds is None:
        param_bounds = {}
    e_low, e_high = fit_window

    # 1. Construct the Mathematical Model
    # [0]: Constant Background
    # [1]: Amplitude (Area)
    # [2]: Mean (mu)
    # [3]: Sigma (sigma)
    
    bg_string = "[0]"
    
    # Gaussian normalized to area [1]
    # Using GetBinWidth(1) instead of 0, as 0 is technically the underflow bin in ROOT
    bin_width = spectrum.GetBinWidth(1) 
    
    # 2.50662827 is sqrt(2*pi)
    gaus_string = f"([1] * {bin_width} / ([3] * 2.50662827)) * TMath::Exp(-0.5 * ((x-[2])/[3]) * ((x-[2])/[3]))"
    function_string = f"{bg_string} + {gaus_string}"

    # 2. Setup Parameters and Initial Guesses
    sigma_guess = get_sigma(data_source, e_guess)
    
    spectrum.GetXaxis().SetRangeUser(*fit_window)

    bg_guess = spectrum.GetBinContent(spectrum.GetXaxis().GetFirst())
    
    # Estimate Area: (Max Height - Background) * Sigma * sqrt(2*pi) / bin_width
    max_height = spectrum.GetBinContent(spectrum.GetMaximumBin()) - bg_guess
    A_guess = max_height * sigma_guess * 2.50662827 / bin_width
    A_guess = max(A_guess, 1.0) # Prevent 0 or negative area guesses
    
    if data_source == 'gamma_adc':
        sigma_bounds = (1, 20)
        A_bounds = (1, np.inf)
    else:
        # Fallback bounds if other data sources are passed
        sigma_bounds = (0.1, 100)
        A_bounds = (0, np.inf)

    spectrum.GetXaxis().UnZoom()

    initial_values = [
        bg_guess,      # p0: bg_const
        A_guess,       # p1: amplitude
        e_guess,       # p2: mu
        sigma_guess    # p3: sigma
    ]

    bg_bounds = param_bounds.get('bg_const', (0, np.inf))
    A_bounds = param_bounds.get('amplitude', A_bounds)
    mu_bounds = param_bounds.get('mu', (e_low, e_high))
    sigma_bounds = param_bounds.get('sigma', sigma_bounds)

    bounds = [
        bg_bounds,       # p0: bg_const
        A_bounds,        # p1: amplitude
        mu_bounds,       # p2: mu bounds
        sigma_bounds     # p3: sigma 
    ]

    names = ["bg_const", "amplitude", "mu", "sigma"]

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
    
    # Gaussian Peak component (parameters 1 through 3)
    peaks = ROOT.TF1(f'gaus_{comp_id}', gaus_string, e_low, e_high)
    for i in range(1, 4):
        peaks.SetParameter(i, fit_params[i])
        
    return fit_res, background, peaks, rp, canvas, spectrum_to_plot, f_to_fit, h_fit

def fit_gaussian_w_bg_shift(spectrum:ROOT.TH1D, e_guess:float|list, fit_window:tuple, data_source=None, param_bounds=None): 
    """
    Fits a standard Gaussian + a step-like background shift using the fit_func engine.
    data_source can be specified to get default guesses and values for sigma and A
    """
    if param_bounds is None:
        param_bounds = {}
    e_low, e_high = fit_window
    
    if not isinstance(e_guess, list) and not isinstance(e_guess, tuple):
        e_guess_list = [e_guess]
    else:
        e_guess_list = e_guess
        
    n_peaks = len(e_guess_list)

    # 1. Construct the Mathematical Model
    # [0]: Constant Background
    # [1]: Amplitude (Area) for peak 0
    # [2]: Mean (mu) for peak 0
    # [3]: Sigma (shared across all peaks)
    # [4]: shift in background (left - right) as fraction of peak area, shared for all peaks
    # If n_peaks > 1:
    # [5]: Amplitude of peak 1
    # [6]: Mean of peak 1
    # ...
    
    # Gaussian normalized to area [1]
    # Using GetBinWidth(1) instead of 0, as 0 is technically the underflow bin in ROOT
    bin_width = spectrum.GetBinWidth(1) 
    
    bg_string = "[0]"
    gaus_strings = []
    
    for i in range(n_peaks):
        if i == 0:
            amp_idx = 1
            mu_idx = 2
        else:
            amp_idx = 3 + 2 * i
            mu_idx = 4 + 2 * i
            
        bg_string += f" + 0.5*[{amp_idx}]*[4]*TMath::Erfc((x-[{mu_idx}])/(1.41421356*[3]))"
        
        # 2.50662827 is sqrt(2*pi)
        gaus_string = f"([{amp_idx}] * {bin_width} / ([3] * 2.50662827)) * TMath::Exp(-0.5 * ((x-[{mu_idx}])/[3]) * ((x-[{mu_idx}])/[3]))"
        gaus_strings.append(gaus_string)

    function_string = f"{bg_string} + {' + '.join(gaus_strings)}"

    # 2. Setup Parameters and Initial Guesses
    if data_source is None:
        sigma_guess = 1
    else:
        sigma_guess = get_sigma(data_source, e_guess)
    
    spectrum.GetXaxis().SetRangeUser(*fit_window)

    bg_guess = spectrum.GetBinContent(spectrum.GetXaxis().GetFirst())
    
    # Calculate max and min within the zoomed fit window
    max_bin = spectrum.GetMaximumBin()
    min_bin = spectrum.GetMinimumBin()
    max_val = spectrum.GetBinContent(max_bin)
    min_val = spectrum.GetBinContent(min_bin)
    
    # Estimate Area: (Max Height - Background) * Sigma * sqrt(2*pi) / bin_width
    max_height = max_val - bg_guess
    A_guess = max_height * sigma_guess * 2.50662827 / bin_width
    A_guess = max(A_guess, 1.0) # Prevent 0 or negative area guesses
    
    # The absolute maximum physical shift is the difference between the max and min bins
    bg_shift_limit = 1#max_val - min_val
    
    if data_source is None:
        sigma_bounds = (0.1, 100)
        A_bounds_default = (0, np.inf)
    elif data_source == 'gamma_adc':
        sigma_bounds = (1, 20)
        A_bounds_default = (1, np.inf)
    else:
        # Fallback bounds if other data sources are passed
        sigma_bounds = (0.1, 100)
        A_bounds_default = (0, np.inf)

    spectrum.GetXaxis().UnZoom()

    initial_values = [
        bg_guess,      # p0: bg_const
        A_guess,       # p1: amplitude 0
        e_guess_list[0], # p2: mu 0
        sigma_guess,   # p3: sigma
        0.002            # p4: bg_shift
    ]

    bg_bounds = param_bounds.get('bg_const', (0, np.inf))
    A_bounds = param_bounds.get('amplitude_0', param_bounds.get('amplitude', A_bounds_default))
    mu_bounds = param_bounds.get('mu_0', param_bounds.get('mu', (e_low, e_high)))
    sigma_bounds = param_bounds.get('sigma', sigma_bounds)
    bg_shift_bounds = param_bounds.get('bg_shift', (0, bg_shift_limit))

    bounds = [
        bg_bounds,       # p0: bg_const
        A_bounds,        # p1: amplitude 0
        mu_bounds,       # p2: mu 0
        sigma_bounds,    # p3: sigma 
        bg_shift_bounds  # p4: bg_shift
    ]

    if n_peaks == 1:
        names = ["bg_const", "amplitude", "mu", "sigma", "bg_shift"]
    else:
        names = ["bg_const", "amplitude_0", "mu_0", "sigma", "bg_shift"]

    for i in range(1, n_peaks):
        initial_values.extend([A_guess, e_guess_list[i]])
        bounds.extend([
            param_bounds.get(f'amplitude_{i}', param_bounds.get('amplitude', A_bounds_default)),
            param_bounds.get(f'mu_{i}', param_bounds.get('mu', (e_low, e_high)))
        ])
        names.extend([f"amplitude_{i}", f"mu_{i}"])

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
    
    # Background component
    background = ROOT.TF1(f'bg_{comp_id}', bg_string, e_low, e_high)
    for i in range(len(fit_params)):
        background.SetParameter(i, fit_params[i])
    
    # Gaussian Peak component
    gaus_combined_string = " + ".join(gaus_strings)
    peaks = ROOT.TF1(f'gaus_{comp_id}', gaus_combined_string, e_low, e_high)
    for i in range(len(fit_params)):
        peaks.SetParameter(i, fit_params[i])
        
    return fit_res, background, peaks, rp, canvas, spectrum_to_plot, f_to_fit, h_fit

def fit_emg_w_bg_shift(spectrum:ROOT.TH1D, e_guess:float|list, fit_window:tuple, data_source=None, param_bounds=None): 
    from scipy.special import erfcx, erfc
    if param_bounds is None:
        param_bounds = {}
    e_low, e_high = fit_window
    
    if not isinstance(e_guess, list) and not isinstance(e_guess, tuple):
        e_guess_list = [e_guess]
    else:
        e_guess_list = e_guess
        
    n_peaks = len(e_guess_list)

    # 1. Call fit_gaussian_w_bg_shift to get initial guesses
    gaus_res = fit_gaussian_w_bg_shift(spectrum, e_guess, fit_window, data_source, param_bounds)
    gaus_params = np.array(gaus_res[0].Parameters())
    gaus_res[4].Close()

    bin_width = spectrum.GetBinWidth(1) 
    
    def emg_bg_shift_eval(x, p):
        val_x = x[0]
        bg_const = p[0]
        sigma = p[3]
        bg_shift = p[4]
        tau = p[5]
        
        total = bg_const
        for i in range(n_peaks):
            if i == 0:
                amp = p[1]
                mu = p[2]
            else:
                amp = p[4 + 2 * i]
                mu = p[5 + 2 * i]
                
            # Background Step Component
            total += 0.5 * amp * bg_shift * erfc((val_x - mu) / (1.41421356 * sigma))
            
            # EMG Component
            norm = amp * bin_width / (2.0 * tau)
            z_arg = ((val_x - mu) / sigma + sigma / tau) / 1.41421356

            if z_arg < -25:
                exp_arg = (sigma**2)/(2*tau**2) + (val_x - mu)/tau
                if exp_arg < 700:
                    total += norm * np.exp(exp_arg) * erfc(z_arg)
            else:
                gaus_arg = (val_x - mu) / sigma
                gaus_part = np.exp(-0.5 * gaus_arg * gaus_arg)
                erfcx_part = erfcx(z_arg)
                total += norm * gaus_part * erfcx_part
            
        return total

    # 3. Setup Parameters and Initial Guesses
    if data_source is None:
        A_bounds_default = (0, np.inf)
        sigma_bounds = (0.1, 100)
        tau_bounds = (0.01, 100)
    elif data_source == 'gamma_adc':
        A_bounds_default = (1, np.inf)
        sigma_bounds = (1, 20)
        tau_bounds = (0.01, 100)
    else:
        A_bounds_default = (0, np.inf)
        sigma_bounds = (0.1, 100)
        tau_bounds = (0.01, 100)

    bg_shift_limit = 1

    initial_values = [
        gaus_params[0], # p0: bg_const
        gaus_params[1], # p1: amplitude 0
        gaus_params[2], # p2: mu 0
        gaus_params[3], # p3: sigma
        gaus_params[4], # p4: bg_shift
        0.1             # p5: tau guess
    ]

    bounds = [
        param_bounds.get('bg_const', (0, np.inf)),
        param_bounds.get('amplitude_0', param_bounds.get('amplitude', A_bounds_default)),
        param_bounds.get('mu_0', param_bounds.get('mu', (e_low, e_high))),
        param_bounds.get('sigma', sigma_bounds),
        param_bounds.get('bg_shift', (0, bg_shift_limit)),
        param_bounds.get('tau', tau_bounds)
    ]

    if n_peaks == 1:
        names = ["bg_const", "amplitude", "mu", "sigma", "bg_shift", "tau"]
    else:
        names = ["bg_const", "amplitude_0", "mu_0", "sigma", "bg_shift", "tau"]

    for i in range(1, n_peaks):
        initial_values.extend([gaus_params[3 + 2 * i], gaus_params[4 + 2 * i]])
        bounds.extend([
            param_bounds.get(f'amplitude_{i}', param_bounds.get('amplitude', A_bounds_default)),
            param_bounds.get(f'mu_{i}', param_bounds.get('mu', (e_low, e_high)))
        ])
        names.extend([f"amplitude_{i}", f"mu_{i}"])

    # 4. Call generalized fit engine
    fit_res, rp, canvas, spectrum_to_plot, f_to_fit, h_fit = fit_func(
        histogram=spectrum, 
        function_string=emg_bg_shift_eval, 
        initial_values=initial_values, 
        bounds=bounds, 
        fit_range=(e_low, e_high), 
        names=names
    )

    comp_id = uuid.uuid4().hex[:6]
    fit_params = np.array([f_to_fit.GetParameter(i) for i in range(f_to_fit.GetNpar())])
    
    # 5. Background component
    def bg_eval(x, p):
        return emg_bg_shift_eval(x, [p[0], p[1], p[2], p[3], p[4], p[5]]) - peak_eval(x, p)
        
    # Peak component
    def peak_eval(x, p):
        val_x = x[0]
        sigma = p[3]
        tau = p[5]
        
        total = 0.0
        for i in range(n_peaks):
            if i == 0:
                amp = p[1]
                mu = p[2]
            else:
                amp = p[4 + 2 * i]
                mu = p[5 + 2 * i]
            
            norm = amp * bin_width / (2.0 * tau)
            z_arg = ((val_x - mu) / sigma + sigma / tau) / 1.41421356

            if z_arg < -25:
                exp_arg = (sigma**2)/(2*tau**2) + (val_x - mu)/tau
                if exp_arg < 700:
                    total += norm * np.exp(exp_arg) * erfc(z_arg)
            else:
                gaus_arg = (val_x - mu) / sigma
                gaus_part = np.exp(-0.5 * gaus_arg * gaus_arg)
                erfcx_part = erfcx(z_arg)
                total += norm * gaus_part * erfcx_part
                
        return total
        
    background = ROOT.TF1(f'bg_{comp_id}', bg_eval, e_low, e_high, len(initial_values))
    background._pyfunc = bg_eval
    
    peaks = ROOT.TF1(f'emg_{comp_id}', peak_eval, e_low, e_high, len(initial_values))
    peaks._pyfunc = peak_eval
    
    for i in range(len(fit_params)):
        background.SetParameter(i, fit_params[i])
        peaks.SetParameter(i, fit_params[i])
        
    return fit_res, background, peaks, rp, canvas, spectrum_to_plot, f_to_fit, h_fit

def fit_double_gaussian_w_bg_shift(spectrum:ROOT.TH1D, e_guess:float|list, fit_window:tuple, data_source=None, param_bounds=None):
    """
    Wrapper for fit_ngaussian_w_bg_shift to fit two Gaussians per peak.
    """
    return fit_ngaussian_w_bg_shift(spectrum, e_guess, fit_window, num_gaussians=2, data_source=data_source, param_bounds=param_bounds)

def fit_ngaussian_w_bg_shift(spectrum:ROOT.TH1D, e_guess:float|list, fit_window:tuple, num_gaussians:int, data_source=None, param_bounds=None): 
    """
    Fits N Gaussians + a step-like background shift to each peak location.
    All peaks share the same set of N sigmas and N-1 fractions.
    """
    if num_gaussians < 1:
        raise ValueError("num_gaussians must be at least 1.")
    if param_bounds is None:
        param_bounds = {}
    e_low, e_high = fit_window
    
    if not isinstance(e_guess, list) and not isinstance(e_guess, tuple):
        e_guess_list = [e_guess]
    else:
        e_guess_list = e_guess
        
    n_peaks = len(e_guess_list)

    # 1. Get initial guesses from a single Gaussian fit
    temp_spectrum = spectrum.Clone(f"{spectrum.GetName()}_temp_for_guess_{uuid.uuid4().hex[:6]}")
    temp_spectrum.SetDirectory(0) 
    
    gaus_res = fit_gaussian_w_bg_shift(temp_spectrum, e_guess, fit_window, data_source, param_bounds)
    if not gaus_res[0].IsValid():
        print(f"Warning: Initial single Gaussian fit failed. Guesses may be poor.")
        gaus_params = np.ones(5 + 2*(n_peaks-1)) * 0.1 
    else:
        gaus_params = np.array(gaus_res[0].Parameters())
        
    gaus_res[4].Close() 

    # 2. Construct the Mathematical Model
    bg_const_idx = 0
    bg_shift_idx = 1
    sigma_start_idx = 2
    frac_start_idx = 2 + num_gaussians
    peak_params_start_idx = frac_start_idx + (num_gaussians - 1) if num_gaussians > 1 else frac_start_idx
    
    bin_width = spectrum.GetBinWidth(1) 
    
    bg_string = f"[{bg_const_idx}]"
    all_gaus_strings = []
    
    # --- FIX 1: Generate stable recursive fraction weights ---
    weights = []
    if num_gaussians == 1:
        weights.append("1.0")
    else:
        for j in range(num_gaussians):
            if j == 0:
                weights.append(f"[{frac_start_idx}]")
            elif j < num_gaussians - 1:
                terms = [f"(1.0 - [{frac_start_idx + k}])" for k in range(j)]
                terms.append(f"[{frac_start_idx + j}]")
                weights.append(" * ".join(terms))
            else: # The very last weight
                terms = [f"(1.0 - [{frac_start_idx + k}])" for k in range(num_gaussians - 1)]
                weights.append(" * ".join(terms))

    for i in range(n_peaks):
        amp_idx = peak_params_start_idx + 2*i
        mu_idx = peak_params_start_idx + 2*i + 1
            
        bg_string += f" + 0.5*[{amp_idx}]*[{bg_shift_idx}]*TMath::Erfc((x-[{mu_idx}])/(1.41421356*[{sigma_start_idx}]))"
        
        for j in range(num_gaussians):
            sigma_idx = sigma_start_idx + j
            weight_str = weights[j]
            
            gaus_string = f"([{amp_idx}]*({weight_str}) * {bin_width} / ([{sigma_idx}] * 2.50662827)) * TMath::Exp(-0.5 * ((x-[{mu_idx}])/[{sigma_idx}]) * ((x-[{mu_idx}])/[{sigma_idx}]))"
            all_gaus_strings.append(gaus_string)

    function_string = f"{bg_string} + {' + '.join(all_gaus_strings)}"

    # 2. Setup Parameters and Initial Guesses
    bg_const_guess = gaus_params[0]
    sigma_from_gaus = gaus_params[3] 
    bg_shift_guess = gaus_params[4] if gaus_params.size > 4 else 0.0
    
    if data_source is None:
        sigma_bounds_default = (0.1, 100)
        A_bounds_default = (0, np.inf)
    elif data_source == 'gamma_adc':
        sigma_bounds_default = (1, 20)
        A_bounds_default = (1, np.inf)
    else:
        sigma_bounds_default = (0.1, 100)
        A_bounds_default = (0, np.inf)

    initial_values = [bg_const_guess, bg_shift_guess]
    bounds = [
        param_bounds.get('bg_const', (0, np.inf)),
        param_bounds.get('bg_shift', (0, 1.0))
    ]
    names = ["bg_const", "bg_shift"]

    # Sigmas
    for j in range(num_gaussians):
        sigma_guess = sigma_from_gaus * (0.8 + 0.4 * j / max(1, num_gaussians - 1)) if num_gaussians > 1 else sigma_from_gaus
        initial_values.append(sigma_guess)
        bounds.append(param_bounds.get(f'sigma{j+1}', sigma_bounds_default))
        names.append(f"sigma{j+1}")

    # Fractions
    if num_gaussians > 1:
        for j in range(num_gaussians - 1):
            initial_values.append(1.0 / num_gaussians)
            bounds.append(param_bounds.get(f'frac{j+1}', (0.0, 1.0)))
            names.append(f"frac{j+1}")

    # Peaks
    for i in range(n_peaks):
        if i == 0:
            amp_guess = gaus_params[1]
            mu_guess = gaus_params[2]
            amp_name = "amplitude" if n_peaks == 1 else "amplitude_0"
            mu_name = "mu" if n_peaks == 1 else "mu_0"
        else:
            amp_guess = gaus_params[3 + 2 * i]
            mu_guess = gaus_params[4 + 2 * i]
            amp_name = f"amplitude_{i}"
            mu_name = f"mu_{i}"
        
        initial_values.extend([amp_guess, mu_guess])
        bounds.extend([
            param_bounds.get(amp_name, param_bounds.get('amplitude', A_bounds_default)),
            param_bounds.get(mu_name, param_bounds.get('mu', (e_low, e_high)))
        ])
        names.extend([amp_name, mu_name])

    # 3. Call our generalized fit engine
    fit_res, rp, canvas, spectrum_to_plot, f_to_fit, h_fit = fit_func(
        histogram=spectrum, 
        function_string=function_string, 
        initial_values=initial_values, 
        bounds=bounds, 
        fit_range=(e_low, e_high), 
        names=names
    )
    
    # --- FIX 2: Fixed loop bounds so the first peak isn't excluded ---
    reconstructed_bg_string = '[0]'
    for i in range(n_peaks): 
        amp_idx = peak_params_start_idx + 2*i
        mu_idx = peak_params_start_idx + 2*i + 1
        reconstructed_bg_string += f" + 0.5*[{amp_idx}]*[{bg_shift_idx}]*TMath::Erfc((x-[{mu_idx}])/(1.41421356*[{sigma_start_idx}]))"

    comp_id = uuid.uuid4().hex[:6]
    fit_params = np.array([f_to_fit.GetParameter(i) for i in range(f_to_fit.GetNpar())])

    # --- FIX 3: Removed invalid length arguments from string TF1 constructors ---
    background = ROOT.TF1(f'bg_{comp_id}', reconstructed_bg_string, e_low, e_high)
    for i in range(len(fit_params)): 
        background.SetParameter(i, fit_params[i])
    
    gaus_combined_string = " + ".join(all_gaus_strings)
    peaks = ROOT.TF1(f'ngaussian_{comp_id}', gaus_combined_string, e_low, e_high)
    for i in range(len(fit_params)): 
        peaks.SetParameter(i, fit_params[i])
        
    return fit_res, background, peaks, rp, canvas, spectrum_to_plot, f_to_fit, h_fit

def fit_voigt_w_bg_shift(spectrum:ROOT.TH1D, e_guess:float|list, fit_window:tuple, data_source=None, param_bounds=None): 
    """
    Fits a Voigt profile (Gaussian + Lorentzian convolution) + a step-like background shift.
    data_source can be specified to get default guesses and values for sigma and A.
    """
    if param_bounds is None:
        param_bounds = {}
    e_low, e_high = fit_window
    
    if not isinstance(e_guess, list) and not isinstance(e_guess, tuple):
        e_guess_list = [e_guess]
    else:
        e_guess_list = e_guess
        
    n_peaks = len(e_guess_list)

    # 1. Construct the Mathematical Model
    # [0]: Constant Background
    # [1]: Amplitude (Area) for peak 0
    # [2]: Mean (mu) for peak 0
    # [3]: Sigma (Gaussian width, shared across all peaks)
    # [4]: shift in background (left - right) as fraction of peak area, shared for all peaks
    # [5]: Gamma (Lorentzian width, shared across all peaks)
    # If n_peaks > 1:
    # [6]: Amplitude of peak 1
    # [7]: Mean of peak 1
    # ...
    
    bin_width = spectrum.GetBinWidth(1) 
    
    bg_string = "[0]"
    voigt_strings = []
    
    for i in range(n_peaks):
        if i == 0:
            amp_idx = 1
            mu_idx = 2
        else:
            # Shifted by +1 compared to the Gaussian function because of the new Gamma parameter
            amp_idx = 4 + 2 * i 
            mu_idx = 5 + 2 * i
            
        # Background step function uses the Gaussian sigma [3] for the resolution smearing
        bg_string += f" + 0.5*[{amp_idx}]*[4]*TMath::Erfc((x-[{mu_idx}])/(1.41421356*[3]))"
        
        # TMath::Voigt is normalized to 1. Multiply by Amplitude and bin_width.
        voigt_string = f"[{amp_idx}] * {bin_width} * TMath::Voigt(x-[{mu_idx}], [3], [5])"
        voigt_strings.append(voigt_string)

    function_string = f"{bg_string} + {' + '.join(voigt_strings)}"

    # 2. Setup Parameters and Initial Guesses
    if data_source is None:
        sigma_guess = 1.0
    else:
        sigma_guess = get_sigma(data_source, e_guess_list[0])
        
    # Lorentzian width guess. Often smaller than or roughly equal to Gaussian resolution.
    gamma_guess = sigma_guess * 0.5 
    
    spectrum.GetXaxis().SetRangeUser(*fit_window)

    bg_guess = spectrum.GetBinContent(spectrum.GetXaxis().GetFirst())
    
    max_bin = spectrum.GetMaximumBin()
    min_bin = spectrum.GetMinimumBin()
    max_val = spectrum.GetBinContent(max_bin)
    min_val = spectrum.GetBinContent(min_bin)
    
    max_height = max_val - bg_guess
    A_guess = max_height * sigma_guess * 2.50662827 / bin_width
    A_guess = max(A_guess, 1.0) 
    
    bg_shift_limit = 1.0
    
    if data_source is None:
        sigma_bounds = (0.1, 100)
        A_bounds_default = (0, np.inf)
    elif data_source == 'gamma_adc':
        sigma_bounds = (1, 20)
        A_bounds_default = (1, np.inf)
    else:
        sigma_bounds = (0.1, 100)
        A_bounds_default = (0, np.inf)

    spectrum.GetXaxis().UnZoom()

    initial_values = [
        bg_guess,        # p0: bg_const
        A_guess,         # p1: amplitude 0
        e_guess_list[0], # p2: mu 0
        sigma_guess,     # p3: sigma
        0.002,           # p4: bg_shift
        gamma_guess      # p5: gamma (Lorentzian width)
    ]

    bg_bounds = param_bounds.get('bg_const', (0, np.inf))
    A_bounds = param_bounds.get('amplitude_0', param_bounds.get('amplitude', A_bounds_default))
    mu_bounds = param_bounds.get('mu_0', param_bounds.get('mu', (e_low, e_high)))
    sigma_bounds = param_bounds.get('sigma', sigma_bounds)
    bg_shift_bounds = param_bounds.get('bg_shift', (0, bg_shift_limit))
    gamma_bounds = param_bounds.get('gamma', (0.001, 100.0))

    bounds = [
        bg_bounds,       # p0: bg_const
        A_bounds,        # p1: amplitude 0
        mu_bounds,       # p2: mu 0
        sigma_bounds,    # p3: sigma 
        bg_shift_bounds, # p4: bg_shift
        gamma_bounds     # p5: gamma
    ]

    if n_peaks == 1:
        names = ["bg_const", "amplitude", "mu", "sigma", "bg_shift", "gamma"]
    else:
        names = ["bg_const", "amplitude_0", "mu_0", "sigma", "bg_shift", "gamma"]

    for i in range(1, n_peaks):
        initial_values.extend([A_guess, e_guess_list[i]])
        bounds.extend([
            param_bounds.get(f'amplitude_{i}', param_bounds.get('amplitude', A_bounds_default)),
            param_bounds.get(f'mu_{i}', param_bounds.get('mu', (e_low, e_high)))
        ])
        names.extend([f"amplitude_{i}", f"mu_{i}"])

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
    
    # Background component
    background = ROOT.TF1(f'bg_{comp_id}', bg_string, e_low, e_high)
    for i in range(len(fit_params)):
        background.SetParameter(i, fit_params[i])
    
    # Voigt Peak component
    voigt_combined_string = " + ".join(voigt_strings)
    peaks = ROOT.TF1(f'voigt_{comp_id}', voigt_combined_string, e_low, e_high)
    for i in range(len(fit_params)):
        peaks.SetParameter(i, fit_params[i])
        
    return fit_res, background, peaks, rp, canvas, spectrum_to_plot, f_to_fit, h_fit
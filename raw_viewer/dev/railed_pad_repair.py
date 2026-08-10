import numpy as np
import matplotlib.pylab as plt
from scipy.optimize import curve_fit
from raw_viewer import process_runs 

import numpy as np
import matplotlib.pylab as plt
from scipy.optimize import curve_fit
from scipy.special import erf
# from raw_viewer import process_runs 

def skew_normal(x, amplitude, center, sigma, alpha, baseline):
    """
    Skew normal distribution model for asymmetric pulses.
    center: The location parameter (not exactly the peak if alpha != 0)
    sigma: The scale/width parameter
    alpha: The skewness parameter (alpha > 0 means right-skewed, alpha < 0 means left-skewed)
    """
    # Standardize x
    t = (x - center) / sigma
    
    # Gaussian PDF component (unnormalized, peaks at 1.0)
    pdf = np.exp(-0.5 * t**2)
    
    # Skew CDF component
    cdf = 0.5 * (1.0 + erf(alpha * t / np.sqrt(2)))
    
    # The factor of 2 ensures that when alpha=0, the peak height equals 'amplitude'
    return amplitude * pdf * (2.0 * cdf) + baseline

def repair(railed_trace, repair_start_offset, repair_stop_offset, rail_val=4095):
    '''
    railed_trace: array containing trace
    repair_start_offset: int, number of bins before saturation to use for the fit
    repair_stop_offset: int, number of bins after saturation to use for the fit

    Fit a single skew normal distribution to the bins just before and just after 
    the pad rails. Exclude the railed bins from the fit. Fill the railed region 
    with the fitted skew normal curve.
    '''
    # Find indices where the trace is saturated
    railed_indecies = np.where(railed_trace >= rail_val)[0]
    
    # If the trace isn't actually railed, return it as-is
    if len(railed_indecies) == 0:
        return np.copy(railed_trace)
        
    railed_start = railed_indecies[0]
    railed_end = railed_indecies[-1]
    
    # Define indices for left and right regions (excluding the railed bins)
    left_start = max(railed_start - repair_start_offset, 0)
    left_end = railed_start
    
    right_start = railed_end + 1
    right_end = min(railed_end + 1 + repair_stop_offset, len(railed_trace))

    # Extract data for fitting
    x_left = np.arange(left_start, left_end)
    y_left = railed_trace[left_start:left_end]
    
    x_right = np.arange(right_start, right_end)
    y_right = railed_trace[right_start:right_end]
    
    # Combine left and right sides into a single dataset for the fit
    x_fit = np.concatenate((x_left, x_right))
    y_fit = np.concatenate((y_left, y_right))
    
    # Ensure we have enough data points (need at least 5 for a 5-parameter fit)
    if len(x_fit) < 5:
        print("Warning: Not enough data points to perform the Skew Normal fit.")
        return np.copy(railed_trace)

    # --- Initial Guesses ---
    baseline_guess = np.min(y_fit) 
    center_guess = (railed_start + railed_end) / 2.0 
    amplitude_guess = 5000.0 - baseline_guess 
    sigma_guess = len(railed_indecies) + (repair_start_offset + repair_stop_offset) / 4.0
    alpha_guess = 0.0 # Start by assuming no skew (symmetric pulse)
    
    # p0 matches the argument order of skew_normal
    p0 = [amplitude_guess, center_guess, sigma_guess, alpha_guess, baseline_guess]
    
    # --- Bounds ---
    # Alpha bounded roughly between -10 (highly left-skewed) and 10 (highly right-skewed)
    # to prevent mathematical overflow in the error function.
    lower_bounds = [0, left_start, 0.1, -10.0, -np.inf]
    upper_bounds = [np.inf, right_end, np.inf, 10.0, rail_val]
    
    repaired_trace = railed_trace.astype(float).copy()
    
    try:
        # Fit the single Skew Normal to the combined, non-railed data
        popt, _ = curve_fit(skew_normal, x_fit, y_fit, p0=p0, bounds=(lower_bounds, upper_bounds), maxfev=15000)
        
        # Replace ONLY the railed trace bins with the Skew Normal extrapolation
        repaired_trace[railed_indecies] = skew_normal(railed_indecies, *popt)
        
    except RuntimeError:
        print("Skew Normal fit failed to converge. Returning original trace.")
        
    return repaired_trace

# Test Execution
f = process_runs.get_h5_file('e23035', 131)
f.background_subtract_mode = 'none'
pads, traces = f.get_pad_traces(3292)#(2745)
traces = np.array(traces)
railed_trace = traces[np.where(traces==4095)[0][0]]
#repaired_trace = repair(railed_trace, 20, 3)
plt.plot(railed_trace)
plt.plot(repair(railed_trace, 20, 20))
plt.show()

#try "reparing" a non railed pads
def g(i):
    plt.plot(traces[i])
    threshold = (np.max(traces[i]) - np.min(traces[i]))*0.75 + np.min(traces[i])
    plt.plot(repair(traces[i], 8, 8, threshold))
    plt.plot((0, len(traces[i])), (threshold, threshold))
    plt.show()


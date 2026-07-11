import pandas as pd
import numpy as np
import ROOT
import array
import os
import uuid
import numpy as np
import matplotlib.pylab as plt

def extract_cal_data(fit_name, use_in_cal_only=True, fit_column='mu'):
    '''
    Extracts the data from gamma_peaks.csv and the fit results CSV.
    Returns a dictionary of decay groups with the true and fit energies and their uncertainties.
    '''
    script_dir = os.path.dirname(os.path.abspath(__file__))
    peaks_csv_path = os.path.join(script_dir, 'peak_fitting', 'gamma_peaks.csv')
    fit_csv_path = os.path.join(script_dir, 'peak_fitting', f'{fit_name}.csv')
    

    peaks_df = pd.read_csv(peaks_csv_path)
    fit_df = pd.read_csv(fit_csv_path)

    # Find the column for "use for calibration?"
    cal_col = None
    for col in peaks_df.columns:
        if 'use' in col.lower() and 'cal' in col.lower():
            cal_col = col
            break
            
    if use_in_cal_only:
        if cal_col is None:
            print("Could not find calibration column in gamma_peaks.csv")
            return None
        use_in_cal = peaks_df[cal_col].astype(str).str.lower().str.strip()
        cal_peaks = peaks_df[(use_in_cal == 'yes') | (use_in_cal == 'true') | (use_in_cal == '1') | (use_in_cal == 'y')]
    else:
        cal_peaks = peaks_df
    
    decay_groups = {}
    
    for idx, row in cal_peaks.iterrows():
        # true energy
        try:
            true_e = float(row.get('known peak energy (keV)', np.nan))
            peak_e = float(row.get('peak energy', np.nan))
            
            # error on true energy
            true_err_val = row.get('known peak energy uncertainty(keV)', 0.0)
            true_err = float(true_err_val) if not pd.isna(true_err_val) else 0.0
        except (ValueError, TypeError):
            continue
            
        if pd.isna(true_e) or pd.isna(peak_e):
            continue
            
        decay_label = str(row.get('from decay of', 'Unknown')).replace('?', '').strip()
        if decay_label.lower() == 'nan' or decay_label == '':
            decay_label = 'Unknown'
            
        # Find closest loc_guess in fit_df
        val_col = f'{fit_column}_val'
        err_col = f'{fit_column}_err'
        
        if 'loc_guess' in fit_df.columns and val_col in fit_df.columns:
            diffs = np.abs(fit_df['loc_guess'] - peak_e)
            if len(diffs) > 0:
                closest_idx = diffs.idxmin()
                if diffs[closest_idx] < 5.0: # Ensure it's a reasonable match
                    fit_val = fit_df.loc[closest_idx, val_col]
                    fit_err = fit_df.loc[closest_idx, err_col] if err_col in fit_df.columns else 0.0
                    if pd.isna(fit_err):
                        fit_err = 0.0
                        
                    if not pd.isna(fit_val):
                        if decay_label not in decay_groups:
                            decay_groups[decay_label] = {'true_energies': [], 'fit_energies': [], 'true_errs': [], 'fit_errs': []}
                            
                        decay_groups[decay_label]['true_energies'].append(float(true_e))
                        decay_groups[decay_label]['fit_energies'].append(float(fit_val))
                        decay_groups[decay_label]['true_errs'].append(float(true_err))
                        decay_groups[decay_label]['fit_errs'].append(float(fit_err))
        else:
            print(f"Required columns missing in {fit_name}.csv")
            return None
            
    return decay_groups

def show_cal_comparison(decay_groups, fit_name, fit_column='mu', cal_fit=None):
    '''
    Create and show a TGraph comparing the true energy to fit mu (or the specified parameter) for each peak in peak_fitting/gamma_peaks.csv
    If cal_fit is provided (from fit_polynomial_to_cal_comparison), the calibration is applied to the fit energies and errors are propagated.
    '''
    import copy
    decay_groups = copy.deepcopy(decay_groups)
    
    if decay_groups is None:
        return None
        
    n_points = sum(len(g['true_energies']) for g in decay_groups.values())
    if n_points == 0:
        print("No calibration peaks found or matched.")
        return None

    # Apply calibration if provided
    if cal_fit is not None:
        for label, data in decay_groups.items():
            cal_energies, cal_errs = apply_energy_calibration_to_point(data['fit_energies'], data['fit_errs'], cal_fit)
            data['fit_energies'] = list(cal_energies) if not isinstance(cal_energies, list) else cal_energies
            data['fit_errs'] = list(cal_errs) if not isinstance(cal_errs, list) else cal_errs

    mg_main = ROOT.TMultiGraph()
    y_axis_title = "Calibrated Energy (keV)" if cal_fit else f"Fit {fit_column} (keV)"
    
    title = f"Calibration Comparison;True Energy (keV);{y_axis_title}"
    if cal_fit is not None:
        chi2 = 0.0
        ndf = 0
        for data in decay_groups.values():
            for te, fe, te_err, fe_err in zip(data['true_energies'], data['fit_energies'], data['true_errs'], data['fit_errs']):
                err = np.sqrt(te_err**2 + fe_err**2)
                if err > 0:
                    chi2 += ((te - fe) / err)**2
                    ndf += 1
        p_val = ROOT.TMath.Prob(chi2, ndf) if ndf > 0 else 0.0
        title = f"Calibration Comparison (p={p_val:.2e}, #chi^{{2}}/NDF={chi2:.1f}/{ndf});True Energy (keV);{y_axis_title}"
        
    mg_main.SetTitle(title)
    
    mg_res = ROOT.TMultiGraph()
    mg_res.SetTitle(";True Energy (keV);True - Fit (keV)")
    
    legend = ROOT.TLegend(0.15, 0.65, 0.45, 0.85)
    legend.SetBorderSize(0)
    legend.SetFillStyle(0)
    
    colors = [ROOT.kBlack, ROOT.kRed, ROOT.kBlue, ROOT.kGreen+2, ROOT.kMagenta, ROOT.kCyan, ROOT.kOrange+7, ROOT.kViolet, ROOT.kAzure+1]
    
    graphs = [] # Keep references to prevent GC
    color_idx = 0
    all_true_energies = [] # For min/max lines
    
    for label, data in decay_groups.items():
        n = len(data['true_energies'])
        if n == 0:
            continue
            
        x_arr = array.array('d', data['true_energies'])
        y_arr = array.array('d', data['fit_energies'])
        x_err_arr = array.array('d', data['true_errs'])
        y_err_arr = array.array('d', data['fit_errs'])
        
        all_true_energies.extend(data['true_energies'])
        
        res_arr = array.array('d', [t - f for t, f in zip(data['true_energies'], data['fit_energies'])])
        res_err_arr = array.array('d', [np.sqrt(te**2 + fe**2) for te, fe in zip(data['true_errs'], data['fit_errs'])])
        
        color = colors[color_idx % len(colors)]
        color_idx += 1
        
        g_main = ROOT.TGraphErrors(n, x_arr, y_arr, x_err_arr, y_err_arr)
        g_main.SetMarkerStyle(20)
        g_main.SetMarkerColor(color)
        g_main.SetLineColor(color)
        
        g_res = ROOT.TGraphErrors(n, x_arr, res_arr, x_err_arr, res_err_arr)
        g_res.SetMarkerStyle(20)
        g_res.SetMarkerColor(color)
        g_res.SetLineColor(color)
        
        mg_main.Add(g_main)
        mg_res.Add(g_res)
        
        legend.AddEntry(g_main, label, "pe")
        
        graphs.extend([g_main, g_res])

    uid = uuid.uuid4().hex[:6]
    c1 = ROOT.TCanvas(f"c1_{fit_name}_{'calibrated' if cal_fit else 'init'}_{uid}", "Calibration Comparison", 800, 800)
    
    pad1 = ROOT.TPad(f"pad1_{uid}", "pad1", 0, 0.3, 1, 1.0)
    pad1.SetBottomMargin(0.02)
    pad1.Draw()
    c1.cd()
    pad2 = ROOT.TPad(f"pad2_{uid}", "pad2", 0, 0.0, 1, 0.3)
    pad2.SetTopMargin(0.02)
    pad2.SetBottomMargin(0.3)
    pad2.Draw()

    pad1.cd()
    mg_main.Draw("AP")
    
    # Hide x-axis labels on main plot since they share the same axis
    mg_main.GetXaxis().SetLabelSize(0)
    mg_main.GetXaxis().SetTitleSize(0)
    
    legend.Draw()
    
    min_e = min(all_true_energies) * 0.9
    max_e = max(all_true_energies) * 1.1
    line = ROOT.TLine(min_e, min_e, max_e, max_e)
    line.SetLineColor(ROOT.kRed)
    line.SetLineStyle(2)
    line.Draw("SAME")
    
    pad2.cd()
    mg_res.Draw("AP")
    
    # Adjust sizes for bottom pad
    mg_res.GetXaxis().SetTitleSize(0.12)
    mg_res.GetXaxis().SetLabelSize(0.12)
    mg_res.GetYaxis().SetTitleSize(0.12)
    mg_res.GetYaxis().SetLabelSize(0.12)
    mg_res.GetYaxis().SetTitleOffset(0.4)
    
    line_zero = ROOT.TLine(min_e, 0, max_e, 0)
    line_zero.SetLineColor(ROOT.kRed)
    line_zero.SetLineStyle(2)
    line_zero.Draw("SAME")
    
    c1.Update()
    
    return c1, pad1, pad2, mg_main, mg_res, legend, graphs, line, line_zero


def fit_polynomial_to_cal_comparison(cal_results, order=1):
    '''
    Extracts the points from the uncalibrated comparison plot and fits E_true = P(mu).
    Draws the parametric fit curve on the main pad and the expected residual curve on the bottom pad.
    Returns the fit function (E vs mu), the residual drawing graph, and the fit result.
    '''
    if not cal_results or len(cal_results) < 9:
        print("Invalid results from show_init_cal_comparison")
        return None, None, None
        
    c1 = cal_results[0]
    pad1 = cal_results[1]
    pad2 = cal_results[2]
    mg_main = cal_results[3]
    
    # Check if canvas is still alive in ROOT to avoid segfaults
    canvas_alive = False
    if c1:
        for c in ROOT.gROOT.GetListOfCanvases():
            if c.GetName() == c1.GetName():
                canvas_alive = True
                break

    # Extract X=mu and Y=E_true to fit E = P(mu)
    mu_all = []
    e_all = []
    mu_err_all = []
    e_err_all = []
    
    # Graphs list is cal_results[6]. 
    # mg_main.GetListOfGraphs() holds only the main pad's graphs (E_true vs mu).
    for g in mg_main.GetListOfGraphs():
        for i in range(g.GetN()):
            mu_all.append(g.GetPointY(i))
            e_all.append(g.GetPointX(i))
            mu_err_all.append(g.GetErrorY(i))
            e_err_all.append(g.GetErrorX(i))
            
    if len(mu_all) == 0:
        print("No points to fit.")
        return None, None, None
        
    g_fit = ROOT.TGraphErrors(len(mu_all), array.array('d', mu_all), array.array('d', e_all),
                              array.array('d', mu_err_all), array.array('d', e_err_all))
                              
    fit_name = f"pol{order}"
    
    # For order >= 2, ROOT's MINUIT minimizer can struggle to converge
    # because mu^n can be very large (e.g., 3000^3 = 2.7e10). The default step sizes
    # for the parameters are too large, causing the fit to immediately blow up.
    # We fix this by explicitly defining the TF1 and setting smart initial step sizes.
    temp_func = ROOT.TF1(f"temp_{fit_name}", fit_name, min(mu_all), max(mu_all))
    
    # Initialize assuming roughly E = 0 + 1*mu
    temp_func.SetParameter(0, 0.0)
    if order >= 1:
        temp_func.SetParameter(1, 1.0)
    for i in range(2, order + 1):
        temp_func.SetParameter(i, 0.0)
        
    # Set the initial step size (ParError) so MINUIT takes appropriately small steps.
    # For parameter i, the term is p_i * mu^i. If mu is ~1000, mu^i is ~1000^i.
    for i in range(order + 1):
        temp_func.SetParError(i, 1.0 / (6000.0 ** i))
        
    # "S0" means return TFitResultPtr, but do not draw automatically
    fit_result = g_fit.Fit(temp_func, "MS0")
    fit_func_orig = g_fit.GetFunction(f"temp_{fit_name}")
    
    if not fit_func_orig:
        print("Fit failed.")
        return None, None, None
        
    uid = uuid.uuid4().hex[:6]
    fit_func = fit_func_orig.Clone(f"{fit_name}_{uid}")
        
    # Prevent garbage collection of the temporary graph which owns the fit function
    fit_func.g_fit = g_fit
        
    # Generate parametric curves to draw on the original axes: X = E_true, Y = mu
    # We will vary mu, compute E_true = P(mu), and plot X=P(mu), Y=mu
    mu_min = min(mu_all)
    mu_max = max(mu_all)
    mu_range = np.linspace(mu_min * 0.9, mu_max * 1.1, 500)
    e_range = [fit_func.Eval(m) for m in mu_range]
    
    g_draw = ROOT.TGraph(len(mu_range), array.array('d', e_range), array.array('d', mu_range))
    g_draw.SetLineColor(ROOT.kBlue)
    g_draw.SetLineStyle(1)
    g_draw.SetLineWidth(2)
    
    # Residuals: original plot shows y = True - mu vs x = True.
    # Our model predicts True = P(mu). So expected residual is P(mu) - mu vs P(mu).
    res_range = [e - m for e, m in zip(e_range, mu_range)]
    g_res_draw = ROOT.TGraph(len(mu_range), array.array('d', e_range), array.array('d', res_range))
    g_res_draw.SetLineColor(ROOT.kBlue)
    g_res_draw.SetLineStyle(1)
    g_res_draw.SetLineWidth(2)
    
    if canvas_alive:
        pad1.cd()
        g_draw.Draw("L SAME")
        pad2.cd()
        g_res_draw.Draw("L SAME")
        c1.Update()
    else:
        print("Note: The canvas was closed, so the fits were calculated but could not be drawn.")
        
    # Prevent garbage collection of the drawing graphs
    if not hasattr(fit_func, "g_draw"):
        fit_func.g_draw = g_draw
        fit_func.g_res_draw = g_res_draw
        
    return fit_func, g_res_draw, fit_result

def apply_energy_calibration_to_point(mu, mu_err, calibration_results):
    '''
    Applies the energy calibration and propagates uncertainties.
    calibration_results is expected to be a tuple (fit_func, res_func, fit_result).
    Handles both scalar values and lists/arrays.
    Returns (E_cal, E_err).
    '''
    if not calibration_results or len(calibration_results) < 3:
        return mu, mu_err
        
    fit_func = calibration_results[0]
    fit_result = calibration_results[2]
    
    if not fit_func or not fit_result:
        return mu, mu_err
        
    # Handle array-like inputs recursively
    if isinstance(mu, (list, np.ndarray, pd.Series)):
        cal_e = []
        cal_err = []
        for m, merr in zip(mu, mu_err):
            e, err = apply_energy_calibration_to_point(m, merr, calibration_results)
            cal_e.append(e)
            cal_err.append(err)
            
        if isinstance(mu, np.ndarray):
            return np.array(cal_e), np.array(cal_err)
        elif isinstance(mu, pd.Series):
            return pd.Series(cal_e, index=mu.index), pd.Series(cal_err, index=mu_err.index)
        return cal_e, cal_err
        
    # Scalar computation
    E_cal = fit_func.Eval(mu)
    dE_dmu = fit_func.Derivative(mu)
    
    cov = fit_result.GetCovarianceMatrix()
    n_params = fit_func.GetNpar()
    
    err2_cov = 0.0
    for i in range(n_params):
        for j in range(n_params):
            err2_cov += (mu**i) * (mu**j) * cov(i, j)
            
    err2 = err2_cov + (dE_dmu * mu_err)**2
    E_err = np.sqrt(err2) if err2 > 0 else 0.0
    
    return E_cal, E_err

def apply_energy_calibration_to_cascade(mus, mu_errs, calibration_results):
    '''
    Applies the energy calibration and propagates uncertainties to a cascade of gammas
    to get the summed energy of the cascade.
    mus: array-like of peak energies
    mu_errs: array-like of peak uncertainties
    calibration_results: tuple (fit_func, res_func, fit_result) from show_cal_comparison
    Returns (E_cal, E_err)
    '''
    if not calibration_results or len(calibration_results) < 3:
        return np.sum(mus), np.sqrt(np.sum(np.array(mu_errs)**2))
        
    fit_func = calibration_results[0]
    fit_result = calibration_results[2]
    
    if not fit_func or not fit_result:
        return np.sum(mus), np.sqrt(np.sum(np.array(mu_errs)**2))
        
    E_cal = 0.0
    stat_err2 = 0.0
    
    for mu, mu_err in zip(mus, mu_errs):
        E_cal += fit_func.Eval(mu)
        dE_dmu = fit_func.Derivative(mu)
        stat_err2 += (dE_dmu * mu_err)**2
        
    cov = fit_result.GetCovarianceMatrix()
    n_params = fit_func.GetNpar()
    
    # S_j = sum(mu^j for mu in mus)
    S = [sum(mu**j for mu in mus) for j in range(n_params)]
    
    err2_cov = 0.0
    for i in range(n_params):
        for j in range(n_params):
            err2_cov += S[i] * S[j] * cov(i, j)
            
    err2 = err2_cov + stat_err2
    E_err = np.sqrt(err2) if err2 > 0 else 0.0
    
    return E_cal, E_err

def apply_energy_calibration_to_fit(fit_name, calibration_results):
    '''
    Read in the specified fit csv file, and write the (fit_index, pvalue, calibrated_energy, calibrated energy error, amplitude, amplitude error)
    to a new csv file in the same folder with file name {original_name}_calibrated.csv.
    '''
    script_dir = os.path.dirname(os.path.abspath(__file__))
    fit_csv_path = os.path.join(script_dir, 'peak_fitting', f'{fit_name}.csv')
    
    if not os.path.exists(fit_csv_path):
        print(f"Fit file {fit_csv_path} not found.")
        return
        
    df = pd.read_csv(fit_csv_path)
    
    if 'mu_val' not in df.columns or 'mu_err' not in df.columns:
        print(f"Required columns (mu_val, mu_err) missing in {fit_csv_path}")
        return
        
    cal_e, cal_err = apply_energy_calibration_to_point(df['mu_val'], df['mu_err'], calibration_results)
    
    out_df = pd.DataFrame()
    out_df['fit_index'] = df['fit_index'] if 'fit_index' in df.columns else np.arange(len(df))
    
    if 'p_value' in df.columns:
        out_df['pvalue'] = df['p_value']
    elif 'pvalue' in df.columns:
        out_df['pvalue'] = df['pvalue']
        
    out_df['calibrated_energy'] = cal_e
    out_df['calibrated_energy_err'] = cal_err
    
    if 'amplitude_val' in df.columns:
        out_df['amplitude'] = df['amplitude_val']
    if 'amplitude_err' in df.columns:
        out_df['amplitude_err'] = df['amplitude_err']
        
    out_csv_path = os.path.join(script_dir, 'peak_fitting', f'{fit_name}_calibrated.csv')
    out_df.to_csv(out_csv_path, index=False)
    print(f"Calibrated results written to {out_csv_path}")

def show_comparison_of_poly_corrections(fit_name = '60Ga_beam_off_gamma',max_order=6):
    decay_groups_init = extract_cal_data(fit_name, True)
    init_cal = show_cal_comparison(decay_groups_init, fit_name)

    corrections = {}
    comparisons = {}

    for order in range(1, max_order + 1):
        correction = fit_polynomial_to_cal_comparison(init_cal, order)
        comparison = show_cal_comparison(decay_groups_init, fit_name, cal_fit=correction)
        corrections[order] = correction
        comparisons[order] = comparison

    Es = np.linspace(0, 6000, 6001)

    # Plot 1: Calibration Error
    plt.figure()
    for order in range(1, max_order + 1):
        errs = [apply_energy_calibration_to_point(E, 0, corrections[order])[1] for E in Es]
        plt.plot(Es, errs, label=f'{order}{"st" if order==1 else "nd" if order==2 else "rd" if order==3 else "th"} order')
    plt.legend()
    plt.xlabel('Energy (keV)')
    plt.ylabel('Calibration Error (keV)')
    plt.yscale('log')

    # Plot 2: Calibration Offset
    plt.figure()
    for order in range(1, max_order + 1):
        diffs = [apply_energy_calibration_to_point(E, 0, corrections[order])[0] - E for E in Es]
        plt.plot(Es, diffs, label=f'{order}{"st" if order==1 else "nd" if order==2 else "rd" if order==3 else "th"} order')
    plt.legend()
    plt.xlabel('Energy (keV)')
    plt.ylabel('Calibrated Energy - Uncalibrated Energy (keV)')

    # Plot 3: Chi2/NDF vs Order
    orders = list(range(1, max_order + 1))
    chi2_vals = [corrections[o][2].Chi2() / corrections[o][2].Ndf() if corrections[o][2].Ndf() > 0 else 0 for o in orders]

    plt.figure()
    plt.plot(orders, chi2_vals, marker='o')
    plt.xlabel('Polynomial Order')
    plt.ylabel(r'$\chi^2 / NDF$')
    plt.title(r'Fit $\chi^2 / NDF$ vs. Polynomial Order')
    plt.grid(True)

    #calibration offset / calibration error
    plt.figure()
    for order in range(1, max_order + 1):
        diffs = [apply_energy_calibration_to_point(E, 0, corrections[order])[0] - E for E in Es]
        errs = [apply_energy_calibration_to_point(E, 0, corrections[order])[1] for E in Es]
        plt.plot(Es, np.array(diffs) / np.array(errs), label=f'{order}{"st" if order==1 else "nd" if order==2 else "rd" if order==3 else "th"} order')
    plt.legend()
    plt.xlabel('Energy (keV)')
    plt.ylabel('Calibration Offset / Calibration Error')
    plt.show()

cal_fit_name = '60Ga_all_gamma'
#init_cal_decay_groups = extract_cal_data(cal_fit_name, True)
all_decay_groups = extract_cal_data(cal_fit_name, False)
init_cal_decay_groups = {
    '76Ge(n,n\')':{
        'true_energies': [562.197],
        'fit_energies': [all_decay_groups['76Ge(n,n\')']['fit_energies'][0]],
        'true_errs': [0.023],
        'fit_errs': [all_decay_groups['76Ge(n,n\')']['fit_errs'][0]]
    },
    '27Al(n,n\')': {
        'true_energies': [1014.56],
        'fit_energies': [all_decay_groups['27Al*']['fit_energies'][0]],
        'true_errs': [0.09],
        'fit_errs': [all_decay_groups['27Al*']['fit_errs'][0]]
    },
    '208Tl': {
        'true_energies': [2614.511],
        'fit_energies': [all_decay_groups['208Tl']['fit_energies'][0]],
        'true_errs': [0.01],
        'fit_errs': [all_decay_groups['208Tl']['fit_errs'][0]]
    },
    # '60Zn': {
    #     'true_energies': [273.4],
    #     'fit_energies': [all_decay_groups['60Zn']['fit_energies'][0]],
    #     'true_errs': [0.4],
    #     'fit_errs': [all_decay_groups['60Zn']['fit_errs'][0]]
    # },
    '16O(n,n\')': {
        'true_energies': [6129.89],
        'fit_energies': [all_decay_groups['16O*']['fit_energies'][0]],
        'true_errs': [0.04],
        'fit_errs': [all_decay_groups['16O*']['fit_errs'][0]]
    }
}

init_cal = show_cal_comparison(init_cal_decay_groups, cal_fit_name)
calibration = fit_polynomial_to_cal_comparison(init_cal, 1)
disp_fit_name = '60Ga_beam_off_gamma'
disp_cal_decay_groups = extract_cal_data(disp_fit_name, False)
calibrated_data = show_cal_comparison(disp_cal_decay_groups, disp_fit_name, cal_fit=calibration)
apply_energy_calibration_to_fit(disp_fit_name, calibration)

Es = np.linspace(0, 6000, 6001)
err = [apply_energy_calibration_to_point(E, 0, calibration)[1] for E in Es]
plt.figure()
plt.plot(Es, err)
plt.xlabel('Energy (keV)')
plt.ylabel('Uncertainty in Energy Calibration (keV)')
#plt.yscale('log')
plt.show(block=False)

cascade1=[[4852.753],[0.108]]
cascade2=[[2434.722,1414.244, 1004.043],[0.036,0.099,0.003]]
print('cascade consistency check')
print('cascade 1 sum = ', apply_energy_calibration_to_cascade(cascade1[0], cascade1[1], calibration))
print('cascade 2 sum = ', apply_energy_calibration_to_cascade(cascade2[0], cascade2[1], calibration))
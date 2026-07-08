import pandas as pd
import numpy as np
import ROOT
import array
import os

def show_init_cal_comparison(fit_name, use_in_cal_only=True, fit_column='mu'):
    '''
    Create and show a TGraph comparing the true energy to fit mu (or the specified parameter) for each peak in peak_fitting/gamma_peaks.csv
    If use_in_cal_only is true, then only peaks with "use in cal" set in the csv file will be used.
    Fit values are pulled from peak_fitting/{fit_name}.csv. Plot a line with slope
    of one and intercept of 0. In smaller axis below the main scatter plot, show (true energy) - (fit energy).
    '''
    script_dir = os.path.dirname(os.path.abspath(__file__))
    peaks_csv_path = os.path.join(script_dir, 'peak_fitting', 'gamma_peaks.csv')
    fit_csv_path = os.path.join(script_dir, 'peak_fitting', f'{fit_name}.csv')
    
    try:
        peaks_df = pd.read_csv(peaks_csv_path)
        fit_df = pd.read_csv(fit_csv_path)
    except FileNotFoundError as e:
        print(f"Error loading CSV files: {e}")
        return None

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
                    fit_mu = fit_df.loc[closest_idx, val_col]
                    fit_mu_err = fit_df.loc[closest_idx, err_col] if err_col in fit_df.columns else 0.0
                    if pd.isna(fit_mu_err):
                        fit_mu_err = 0.0
                        
                    if not pd.isna(fit_mu):
                        if decay_label not in decay_groups:
                            decay_groups[decay_label] = {'true_energies': [], 'fit_energies': [], 'true_errs': [], 'fit_errs': []}
                            
                        decay_groups[decay_label]['true_energies'].append(float(true_e))
                        decay_groups[decay_label]['fit_energies'].append(float(fit_mu))
                        decay_groups[decay_label]['true_errs'].append(float(true_err))
                        decay_groups[decay_label]['fit_errs'].append(float(fit_mu_err))
        else:
            print(f"Required columns missing in {fit_name}.csv")
            return None

    n_points = sum(len(g['true_energies']) for g in decay_groups.values())
    if n_points == 0:
        print("No calibration peaks found or matched.")
        return None
        
    mg_main = ROOT.TMultiGraph()
    mg_main.SetTitle(f"Initial Calibration Comparison;True Energy (keV);Fit {fit_column} (keV)")
    
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

    c1 = ROOT.TCanvas(f"c1_{fit_name}", "Calibration Comparison", 800, 800)
    
    pad1 = ROOT.TPad("pad1", "pad1", 0, 0.3, 1, 1.0)
    pad1.SetBottomMargin(0.02)
    pad1.Draw()
    c1.cd()
    pad2 = ROOT.TPad("pad2", "pad2", 0, 0.0, 1, 0.3)
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
    
    # Return objects so they are not garbage collected by PyROOT
    return c1, pad1, pad2, mg_main, mg_res, legend, graphs, line, line_zero

def fit_polynomial_to_cal_comparison(cal_results, order=1):
    '''
    Fits an N-th order polynomial to the TMultiGraph produced by show_init_cal_comparison.
    Draws the fit on the main pad and the expected residual on the bottom pad.
    Returns the main fit function, the residual fit function, and the fit result.
    '''
    if not cal_results or len(cal_results) < 9:
        print("Invalid results from show_init_cal_comparison")
        return None
        
    c1 = cal_results[0]
    pad1 = cal_results[1]
    pad2 = cal_results[2]
    mg_main = cal_results[3]
    line_zero = cal_results[8]
    
    # Check if canvas is still alive in ROOT to avoid segfaults
    canvas_alive = False
    if c1:
        for c in ROOT.gROOT.GetListOfCanvases():
            if c.GetName() == c1.GetName():
                canvas_alive = True
                break

    if canvas_alive:
        pad1.cd()
    
    # "polN" is ROOT's built-in polynomial function
    fit_name = f"pol{order}"
    
    # Fit the multigraph. "S" returns the TFitResultPtr. "0" prevents drawing if canvas is closed.
    fit_opt = "S" if canvas_alive else "S0"
    fit_result = mg_main.Fit(fit_name, fit_opt)
    
    fit_func = mg_main.GetFunction(fit_name)
    res_func = None
    
    if fit_func:
        fit_func.SetLineColor(ROOT.kBlue)
        fit_func.SetLineStyle(1)
        fit_func.SetLineWidth(2)
        
        # Create residual function: x - polynomial(x)
        formula = "x - (" + " + ".join([f"[{i}]*TMath::Power(x,{i})" for i in range(order+1)]) + ")"
        
        xmin = line_zero.GetX1()
        xmax = line_zero.GetX2()
        res_func = ROOT.TF1(f"res_pol{order}", formula, xmin, xmax)
        
        for i in range(order+1):
            res_func.SetParameter(i, fit_func.GetParameter(i))
            
        res_func.SetLineColor(ROOT.kBlue)
        res_func.SetLineStyle(1)
        res_func.SetLineWidth(2)
        
        if canvas_alive:
            pad2.cd()
            res_func.Draw("SAME")
            c1.Update()
    else:
        print("Fit failed.")
        
    if not canvas_alive:
        print("Note: The canvas was closed, so the fits were calculated but could not be drawn.")
        
    return fit_func, res_func, fit_result
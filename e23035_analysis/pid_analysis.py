import numpy as np
import ROOT
import uuid
import os
import json

from raw_viewer import ddas_interface
from e23035_analysis import fitting_tools

experiment = 'e23035'
ddas_run = 142#262
num_workers = 10

tof_axis_length = np.abs(-6.217e-7 + 6.242e-7)
de_axis_length = 6730-6583

de_dict = {'59Zn':6346,'60Ga':6646}#holds delta E center for each isotope. Assumed to be same for all runs.
tof_dict = {262:{'59Zn':-6.207e-7,'60Ga':-6.229e-7},
            142:{'59Zn':-6.1854e-7,'60Ga':-6.207e-7}}#holds TOF center for each isotope and run

def get_pid_hist(ddas_run, de_detector='msx100_e'):
    uid = uuid.uuid4().hex[:8]
    if de_detector=='msx100_e':
        binning = (1000,-0.63e-6,-0.6e-6,1000,4000,8000)
    elif de_detector=='msx40_e':
        binning = (1000,-0.63e-6,-0.6e-6,1000,2000,12000)
        
    detector_m = de_detector.split('_')[0] + '_m'
    pid_hist = ddas_interface.get_histogram(experiment, ddas_run, binning, f'pid_{de_detector}_{uid}', f'pid {de_detector} Run {ddas_run}', 
                    f'{de_detector}:(cross_scint_b2_t - db_5_scint_t)', 
                    selection=f'cross_scint_b2_m==1 && db_5_scint_m==1 && {detector_m}==1', num_workers=num_workers)
    return pid_hist

def get_cached_fit_params(run, species, pid_hist):
    cache_dir = "hist_cache/pid"
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, f"fit_{run}_{species}.json")
    
    if os.path.exists(cache_file):
        with open(cache_file, 'r') as f:
            return json.load(f)
            
    # Need to fit
    fit_res, canvas, sub_hist, f_to_fit, h_fit, h_resid = fit_gauss_to_pid(run, species, 1.0, 1.0, rho=0.17, pid_hist=pid_hist)
    
    params = {
        'amp': f_to_fit.GetParameter(0),
        'mu_tof': f_to_fit.GetParameter(1),
        'sig_tof': f_to_fit.GetParameter(2),
        'mu_de': f_to_fit.GetParameter(3),
        'sig_de': f_to_fit.GetParameter(4),
        'rho': f_to_fit.GetParameter(5)
    }
    
    with open(cache_file, 'w') as f:
        json.dump(params, f)
        
    return params

def show_pid(ddas_run, de_detector='msx100_e', draw_labels=True, sigma=2.0):
    pid_hist = get_pid_hist(ddas_run, de_detector)
    uid = uuid.uuid4().hex[:8]
    c1 = ROOT.TCanvas(f"c_pid_{de_detector}_{uid}", f"PID {de_detector}", 800, 600)
    pid_hist.Draw("colz")
    cuts = {}
    labels = []
    
    if de_detector == 'msx100_e':
        for species in de_dict:
            try:
                params = get_cached_fit_params(ddas_run, species, pid_hist)
            except Exception as e:
                print(f"Failed to get fit params for {species}: {e}")
                continue
                
            mu_tof = params['mu_tof']
            sig_tof = params['sig_tof']
            mu_de = params['mu_de']
            sig_de = params['sig_de']
            fit_rho = params['rho']
            
            cut = ROOT.TCutG(f"{species}_{uid}", 100)
            cut.SetVarX("(cross_scint_b2_t - db_5_scint_t)")
            cut.SetVarY("msx100_e")
            
            for i in range(100):
                angle = i * 2 * np.pi / 99
                x = mu_tof + (sigma * sig_tof) * np.cos(angle)
                y = mu_de + (sigma * sig_de) * (fit_rho * np.cos(angle) + np.sqrt(1 - fit_rho**2) * np.sin(angle))
                cut.SetPoint(i, x, y)
                
            cut.SetLineColor(ROOT.kRed)
            cut.SetLineWidth(2)
            cut.Draw("same")
            cuts[species] = cut
            
            if draw_labels:
                label = ROOT.TLatex(mu_tof, mu_de + (sigma * sig_de) + 50, species)
                label.SetTextSize(0.04)
                label.SetTextColor(ROOT.kRed)
                label.Draw()
                labels.append(label)

    c1.Update()
    return c1, pid_hist, cuts, labels

def get_pid_counts(ddas_run, species, sigma, de_detector='msx100_e'):
    _, pid_hist, cuts, _ = show_pid(ddas_run, de_detector, draw_labels=False, sigma=sigma)
    
    cut = cuts.get(species)
    if not cut:
        return 0
        
    try:
        counts = cut.IntegralHist(pid_hist)
    except AttributeError:
        # Fallback in case IntegralHist is not available in this ROOT version
        counts = 0
        for bx in range(1, pid_hist.GetNbinsX() + 1):
            for by in range(1, pid_hist.GetNbinsY() + 1):
                x = pid_hist.GetXaxis().GetBinCenter(bx)
                y = pid_hist.GetYaxis().GetBinCenter(by)
                if cut.IsInside(x, y):
                    counts += pid_hist.GetBinContent(bx, by)
                    
    return counts

def show_de_de_comparison(ddas_run, dt_bounds=(-6.217e-7, -6.199e-7)):
    uid = uuid.uuid4().hex[:8]
    zn_cut = f'cross_scint_b2_m==1 && db_5_scint_m==1 && msx100_m==1 && (cross_scint_b2_t - db_5_scint_t) > {dt_bounds[0]} && (cross_scint_b2_t - db_5_scint_t)<{dt_bounds[1]}'
    de_de_hist = ddas_interface.get_histogram(experiment, ddas_run, (1000,0,24000, 1000,0,24000), f'de_de_{uid}', f'de_de Run {ddas_run}', 
                            'msx100_e:msx40_e', selection=zn_cut, num_workers=num_workers)
    c2 = ROOT.TCanvas(f"c2_{uid}", "De-De", 800, 600)
    de_de_hist.Draw("colz")
    c2.Update()
    return c2, de_de_hist

def fit_gauss_to_pid(run, species, de_bounds_mult, tof_bounds_mult, rho=0.0, pid_hist=None):
    #fit a gaussian to the PID histogram. Center the guess at the specified value in de_dict and tof_dict.
    #Only include the portion of the histram within X_bounds_mult*X_axis_length/2 of the center guess.
    
    if pid_hist is None:
        pid_hist = get_pid_hist(run, 'msx100_e')
    
    if species not in de_dict or species not in tof_dict[run]:
        raise ValueError(f"Species {species} not found in dictionaries for run {run}")
        
    de_center = de_dict[species]
    tof_center = tof_dict[run][species]
    
    tof_hw = tof_bounds_mult * tof_axis_length / 2.0
    de_hw = de_bounds_mult * de_axis_length / 2.0
    
    fit_range = ((tof_center - tof_hw, tof_center + tof_hw), 
                 (de_center - de_hw, de_center + de_hw))
                 
    # [0]: amplitude, [1]: mu_x, [2]: sigma_x, [3]: mu_y, [4]: sigma_y, [5]: rho
    func_str = "[0] * TMath::Exp(-0.5/(1-[5]*[5]) * ( ((x - [1])/[2])*((x - [1])/[2]) - 2*[5]*((x - [1])/[2])*((y - [3])/[4]) + ((y - [3])/[4])*((y - [3])/[4]) ))"
    
    bin_x = pid_hist.GetXaxis().FindBin(tof_center)
    bin_y = pid_hist.GetYaxis().FindBin(de_center)
    amp_guess = pid_hist.GetBinContent(bin_x, bin_y)
    if amp_guess <= 0:
        amp_guess = 1.0 # fallback
        
    initial_values = [
        amp_guess,
        tof_center,
        tof_axis_length / 4.0,
        de_center,
        de_axis_length / 4.0,
        rho
    ]
    
    bounds = [
        (0, np.inf),
        (tof_center - tof_hw, tof_center + tof_hw),
        (1e-12, np.inf),
        (de_center - de_hw, de_center + de_hw),
        (1e-12, np.inf),
        (-0.999, 0.999)
    ]
    
    names = ["Amplitude", "mu_tof", "sigma_tof", "mu_de", "sigma_de", "rho"]
    
    fit_res, canvas, sub_hist, f_to_fit, h_fit, h_resid = fitting_tools.fit_hist2d(
        pid_hist, func_str, initial_values, bounds, fit_range, names
    )
    
    amp = f_to_fit.GetParameter(0)
    sig_x = f_to_fit.GetParameter(2)
    sig_y = f_to_fit.GetParameter(4)
    fit_rho = f_to_fit.GetParameter(5)
    
    bin_area = pid_hist.GetXaxis().GetBinWidth(1) * pid_hist.GetYaxis().GetBinWidth(1)
    fit_counts = amp * 2 * np.pi * sig_x * sig_y * np.sqrt(1 - fit_rho**2) / bin_area
    
    print(f"--- Fit Results for {species} (Run {run}) ---")
    print(f"Estimate from 2D Gaussian fit: {fit_counts:.2f}")
    print("-" * 40)
    
    return fit_res, canvas, sub_hist, f_to_fit, h_fit, h_resid

if __name__ == '__main__':
    c1, pid_hist_100, cuts, labels = show_pid(ddas_run, 'msx100_e')
    for isotope in de_dict:
        count = get_pid_counts(ddas_run, isotope, 2.0, 'msx100_e')
        print(f"Counts for {isotope} within 2 sigma: {count}")

    c2, de_de_hist = show_de_de_comparison(ddas_run)
    c3, pid_hist_40, _, _ = show_pid(ddas_run, 'msx40_e')
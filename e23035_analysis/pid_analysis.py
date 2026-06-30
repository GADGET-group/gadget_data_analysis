import numpy as np
import ROOT
import uuid

from raw_viewer import ddas_interface



experiment = 'e23035'
ddas_run = 142#262
num_workers = 10

tof_axis_length = np.abs(-6.217e-7 + 6.242e-7)
de_axis_length = 6730-6583

de_dict = {'59Zn':6346,'60Ga':6646}#holds delta E center for each isotope. Assumed to be same for all runs.
tof_dict = {262:{'59Zn':-6.207e-7,'60Ga':-6.229e-7},
            142:{'59Zn':-6.1854e-7,'60Ga':-6.207e-7}}#holds TOF center for each isotope and run

def show_pid(ddas_run, de_detector='msx100_e', draw_labels=True):
    uid = uuid.uuid4().hex[:8]
    if de_detector=='msx100_e':
        binning = (1000,-0.63e-6,-0.6e-6,1000,4000,8000)
    elif de_detector=='msx40_e':
        binning = (1000,-0.63e-6,-0.6e-6,1000,2000,12000)
        
    detector_m = de_detector.split('_')[0] + '_m'
    pid_hist = ddas_interface.get_histogram(experiment, ddas_run, binning, f'pid_{de_detector}_{uid}', f'pid {de_detector} Run {ddas_run}', 
                    f'{de_detector}:(cross_scint_b2_t - db_5_scint_t)', 
                    selection=f'cross_scint_b2_m==1 && db_5_scint_m==1 && {detector_m}==1', num_workers=num_workers)

    c1 = ROOT.TCanvas(f"c_pid_{de_detector}_{uid}", f"PID {de_detector}", 800, 600)
    pid_hist.Draw("colz")

    cuts = {}
    labels = []
    
    if de_detector == 'msx100_e':
        for isotope in de_dict:
            tof_center = tof_dict[ddas_run][isotope]
            de_center = de_dict[isotope]
            
            cut = ROOT.TCutG(f"{isotope}_{uid}", 100)
            cut.SetVarX("(cross_scint_b2_t - db_5_scint_t)")
            cut.SetVarY("msx100_e")
            
            for i in range(100):
                angle = i * 2 * np.pi / 99
                x = tof_center + (tof_axis_length / 2.0) * np.cos(angle)
                y = de_center + (de_axis_length / 2.0) * np.sin(angle)
                cut.SetPoint(i, x, y)
                
            cut.SetLineColor(ROOT.kRed)
            cut.SetLineWidth(2)
            cut.Draw("same")
            cuts[isotope] = cut
            
            if draw_labels:
                label = ROOT.TLatex(tof_center, de_center + (de_axis_length / 2.0) + 50, isotope)
                label.SetTextSize(0.04)
                label.SetTextColor(ROOT.kRed)
                label.Draw()
                labels.append(label)

    c1.Update()
    return c1, pid_hist, cuts, labels

def get_pid_counts(ddas_run, de_detector='msx100_e'):
    c1, pid_hist, cuts, labels = show_pid(ddas_run, de_detector, draw_labels=False)
    counts_dict = {}
    for isotope, cut in cuts.items():
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
        counts_dict[isotope] = counts
    return counts_dict

def show_de_de_comparison(ddas_run, dt_bounds=(-6.217e-7, -6.199e-7)):
    uid = uuid.uuid4().hex[:8]
    zn_cut = f'cross_scint_b2_m==1 && db_5_scint_m==1 && msx100_m==1 && (cross_scint_b2_t - db_5_scint_t) > {dt_bounds[0]} && (cross_scint_b2_t - db_5_scint_t)<{dt_bounds[1]}'
    de_de_hist = ddas_interface.get_histogram(experiment, ddas_run, (1000,0,24000, 1000,0,24000), f'de_de_{uid}', f'de_de Run {ddas_run}', 
                            'msx100_e:msx40_e', selection=zn_cut, num_workers=num_workers)
    c2 = ROOT.TCanvas(f"c2_{uid}", "De-De", 800, 600)
    de_de_hist.Draw("colz")
    c2.Update()
    return c2, de_de_hist

if __name__ == '__main__':
    c1, pid_hist_100, cuts, labels = show_pid(ddas_run, 'msx100_e')
    counts = get_pid_counts(ddas_run, 'msx100_e')
    for isotope, count in counts.items():
        print(f"Counts for {isotope}: {count}")

    c2, de_de_hist = show_de_de_comparison(ddas_run)
    c3, pid_hist_40, _, _ = show_pid(ddas_run, 'msx40_e')
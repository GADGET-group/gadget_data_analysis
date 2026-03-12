import os

import ROOT
import numpy as np

from raw_viewer import ddas_interface, process_runs
from e23035_analysis import e23035_runs, fitting_tools


run_candidates = e23035_runs.run_df['DDAS'][(e23035_runs.run_df['Run Type']=='60Ga')]
runs = []
for run in run_candidates:
    if not np.isnan(run) and run not in [162,163,203,204,209, 213,217, 218, 238]:
        if os.path.exists(ddas_interface.get_merged_root_file_path(run)):
            runs.append(run)

n_workers = min(len(runs), 150) #cap at 150 so as not to murder the TPCGPU

ROOT.EnableImplicitMT()
counts_branch_list = [f'clover_{clover}_c' for clover in clover_list]
gm_branch_list = [s + '_gm' for s in counts_branch_list]
df = energy_calibration_tools.get_run_dataframe(runs)
df = energy_calibration_tools.apply_calibration(df, runs, counts_branch_list, 'gm')
add_back_logic = ''
crystal_strings_for_max = ''
for i in range(len(gm_branch_list)):
    if i != 0:
        add_back_logic += '+'
        crystal_strings_for_max += ','
    add_back_logic += gm_branch_list[i]
    crystal_strings_for_max += gm_branch_list[i]
df = df.Define('add_back', add_back_logic)
add_back_hist = df.Histo1D(("add_back", "add back", 6000, 0., 6000.), "add_back")
df = df.Define("summed_gamma", "std::max({%s})"%crystal_strings_for_max)
summed_hist = df.Histo1D(("summed_gamma", "summed gamma spectrum", 6000, 0., 6000.), "summed_gamma")
#ddas_interface.get_summed_gamma_spectrum(run, (6000,0, 6000), 'gm')
#root_vis_tools.draw_overlaid_histograms({'add back':add_back_hist, 'summed':summed_hist}, f'run {run}', 'keV')
add_back_hist.Draw()
ROOT.gPad.SetLogy(1)
summed_hist.SetLineColor(ROOT.kRed)
summed_hist.Draw("SAME")

gammas = ddas_interface.get_histogram(runs, (12000-1,1,12000), 'gammas', 'summed gamma spectrum', ddas_interface.get_add_back_gamma_str(), num_workers=n_workers)
s = ddas_interface.get_add_back_gamma_str()
gamma_gated_on_proton = ddas_interface.get_histogram(runs, (12000-1,1,12000), 'gamma_gated_on_protons', 'gamma rays gated on protons', 
                ddas_interface.get_add_back_gamma_str(), 'tpc_particle_id==1', num_workers=n_workers)

protons = ddas_interface.get_histogram(runs, (3000, 0, 3000), 'protons', 'proton energy (keV)', 'tpc_energy', 'tpc_particle_id==1', num_workers=n_workers)
protons_gated_on_gammas = ddas_interface.get_histogram(runs, (3000, 0, 3000), 'protons_gated_on_gammas', 'proton energy (keV) gated on gamma rays',
                                                        'tpc_energy', 'tpc_particle_id==1 && (%s>0)'%s, num_workers=n_workers)

protons_gated_on_914 = ddas_interface.get_histogram(runs, (3000, 0, 3000), 'protons_gated_on_914', 'protons gated on 914 keV gamma',
                                                        'tpc_energy', 'tpc_particle_id==1 && (%s>913) && (%s < 917)'%(s,s), num_workers=n_workers)

protons_gated_on_491 = ddas_interface.get_histogram(runs, (3000, 0, 3000), 'protons_gated_on_491', 'protons gated on 491 keV gamma',
                                                        'tpc_energy', 'tpc_particle_id==1 && (%s<494) && (%s > 489)'%(s,s), num_workers=n_workers)

alphas = ddas_interface.get_histogram(runs, (int((9000-1500/50)), 1500, 9000), 'alphas', 'alpha energy (keV)', 'tpc_energy', 'tpc_particle_id==2', num_workers=n_workers)
gamma_gated_on_alpha = ddas_interface.get_histogram(runs, (12000-1,1,12000), 'gamma_gated_on_alpha', 'gamma rays gated on alpha particle', 
                ddas_interface.get_add_back_gamma_str(), 'tpc_particle_id==2', num_workers=n_workers)

c1,c2 = ROOT.TCanvas(), ROOT.TCanvas()
c1.cd()
gammas.Draw()
#ROOT.gPad.SetLogy(1)
c2.cd()
gamma_gated_on_proton.Draw()
#ROOT.gPad.SetLogy(1)
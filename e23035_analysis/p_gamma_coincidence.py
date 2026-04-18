import os

import ROOT
import numpy as np

from raw_viewer import ddas_interface, process_runs
from e23035_analysis import e23035_runs, fitting_tools, degai


run_candidates = e23035_runs.run_df['DDAS'][(e23035_runs.run_df['Run Type']=='60Ga')]
runs = []
for run in run_candidates:
    if not np.isnan(run) and run not in [162,163,203,204,209, 213,217, 218, 238]:
        if os.path.exists(ddas_interface.get_merged_root_file_path(run)):
            runs.append(run)
runs = e23035_runs.get_ddas_60_Ga_runs()
print(runs)


n_workers = min(len(runs), 150) #cap at 150 so as not to murder the TPCGPU



adj_dict =  degai.crystal_adj_dict#degai.clover_adj_dict#
cal_name = 'gm_511and2614_1'
nlc_name = 'c1'
gamma_bin_size = 0.25 #keV
addback_ethresh = 150
upper_energy = 7000
gamma_binning = (int((upper_energy-addback_ethresh)/gamma_bin_size),addback_ethresh,upper_energy)
event_build_window = 500 #ns
gammas = degai.get_histogram(runs, adj_dict, cal_name, gamma_binning, 'gamma_hist', 'gamma spectrum', 'addback_energy', '', event_build_window, addback_ethresh, True,
                                  nonlinearity_correction_name=nlc_name)

gamma_gated_on_proton = degai.get_histogram(runs, adj_dict, cal_name, gamma_binning, 'gamma_gated_on_protons', 'gamma rays gated on protons', 
                'addback_energy', 'tpc_particle_id==1', event_build_window, addback_ethresh, True, nonlinearity_correction_name=nlc_name)

protons = ddas_interface.get_histogram(runs, (3000, 0, 3000), 'protons', 'proton energy (keV)', 'tpc_energy', 'tpc_particle_id==1', num_workers=n_workers)
protons_gated_on_gammas = degai.get_histogram(runs,  adj_dict, cal_name, (3000, 0, 3000), 'protons_gated_on_gammas', 'proton energy (keV) gated on gamma rays',
                                            'tpc_energy', 'tpc_particle_id==1 && (addback_energy>0)', event_build_window, addback_ethresh, True,
                                            nonlinearity_correction_name=nlc_name)
alphas_gated_on_gammas = degai.get_histogram(runs,  adj_dict, cal_name, (3000, 0, 9000),'alphas_gated_on_gammas', 'alpha energy (keV) gated on gamma rays',
                                             'tpc_energy', 'tpc_particle_id==2 && (addback_energy>0)', event_build_window, addback_ethresh, True,
                                            nonlinearity_correction_name=nlc_name)

gamma_gated_on_alpha = degai.get_histogram(runs, adj_dict, cal_name, gamma_binning, 'gamma_gated_on_alpha', 'gamma rays gated on alphas', 
                'addback_energy', 'tpc_particle_id==2', event_build_window, addback_ethresh, True,
                 nonlinearity_correction_name=nlc_name)

c1,c2 = ROOT.TCanvas(), ROOT.TCanvas()
c1.cd()
gammas.Draw()
#ROOT.gPad.SetLogy(1)
c2.cd()
gamma_gated_on_proton.Draw()
#ROOT.gPad.SetLogy(1)
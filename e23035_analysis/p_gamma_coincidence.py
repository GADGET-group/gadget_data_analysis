import os

import ROOT
import numpy as np

from raw_viewer import ddas_interface, process_runs
from e23035_analysis import e23035_runs, fitting_tools, root_vis_tools
from raw_viewer import degai

experiment = 'e23035'

# run_candidates = e23035_runs.run_df['DDAS'][(e23035_runs.run_df['Run Type']=='60Ga')]
# runs = []
# for run in run_candidates:
#     if not np.isnan(run) and run not in [162,163,203,204,209, 213,217, 218, 238]:
#         if os.path.exists(ddas_interface.get_merged_root_file_path(run)):
#             runs.append(run)
# runs = e23035_runs.get_ddas_60_Ga_runs()
# print(runs)

runs = e23035_runs.get_ddas_60_Ga_runs(good_gamma=True, good_long_tracks_tpc=False, good_low_energy_tpc=False, final_beam_settings=True)
n_workers = min(len(runs), 255) #cap at 150 so as not to murder the TPCGPU



adj_dict =  degai.get_adjacency_dict(30)#degai.crystal_adj_dict#degai.clover_adj_dict#
cal_name = 'gm_511and2614_1'
nlc_name = 'c1'
gamma_bin_size = 0.25 #keV
addback_ethresh = 150
upper_energy = 7000
gamma_binning = (int((upper_energy-addback_ethresh)/gamma_bin_size),addback_ethresh,upper_energy)
event_build_window = 500 #ns

tstart, tstop = 0, 7.6e-6
time_gate_str = f'(mesh_pre_amp_t - time)>{tstart} && (mesh_pre_amp_t - time)<{tstop}'
t_accidental_start, t_accidental_stop = -15e-6, -1e-6
accidental_time_gate_str = f'(mesh_pre_amp_t - time)>{t_accidental_start} && (mesh_pre_amp_t - time)<{t_accidental_stop}'

gammas = degai.get_histogram(experiment, runs, adj_dict, cal_name, gamma_binning, 'gamma_hist', 'gamma spectrum', 'addback_energy', '', event_build_window, addback_ethresh, True,
                                  nonlinearity_correction_name=nlc_name)

gamma_gated_on_proton = degai.get_histogram(experiment, runs, adj_dict, cal_name, gamma_binning, 'gamma_gated_on_protons', 'gamma rays gated on protons', 
                'addback_energy', 'tpc_particle_id==1 &&'+time_gate_str, event_build_window, addback_ethresh, True, nonlinearity_correction_name=nlc_name)

protons = ddas_interface.get_histogram(experiment, runs, (3000, 0, 3000), 'protons', 'proton energy (keV)', 'tpc_energy', 'tpc_particle_id==1', num_workers=n_workers)
protons_gated_on_gammas = degai.get_histogram(experiment, runs,  adj_dict, cal_name, (3000, 0, 3000), 'protons_gated_on_gammas', 'proton energy (keV) gated on gamma rays',
                                            'tpc_energy', 'tpc_particle_id==1 && (addback_energy>0)&&'+time_gate_str, event_build_window, addback_ethresh, True,
                                            nonlinearity_correction_name=nlc_name)
alphas_gated_on_gammas = degai.get_histogram(experiment, runs,  adj_dict, cal_name, (3000, 0, 9000),'alphas_gated_on_gammas', 'alpha energy (keV) gated on gamma rays',
                                             'tpc_energy', 'tpc_particle_id==2 && (addback_energy>0)&&'+time_gate_str, event_build_window, addback_ethresh, True,
                                            nonlinearity_correction_name=nlc_name)

gamma_gated_on_alpha = degai.get_histogram(experiment, runs, adj_dict, cal_name, gamma_binning, 'gamma_gated_on_alpha', 'gamma rays gated on alphas', 
                'addback_energy', 'tpc_particle_id==2 &&'+time_gate_str, event_build_window, addback_ethresh, True,
                 nonlinearity_correction_name=nlc_name)

c1,c2 = ROOT.TCanvas(), ROOT.TCanvas()
c1.cd()
gammas.Draw()
#ROOT.gPad.SetLogy(1)
c2.cd()
gamma_gated_on_proton.Draw()
#ROOT.gPad.SetLogy(1)

protons_gated_on_491 = degai.get_histogram(experiment, runs,  adj_dict, cal_name, (3000, 0, 3000), 'protons_gated_on_491', 'proton energy (keV) gated on 491 keV gamma rays',
                                            'tpc_energy', 'tpc_particle_id==1 && (addback_energy>490) && (addback_energy<493)&&'+time_gate_str, event_build_window, addback_ethresh, True,
                                            nonlinearity_correction_name=nlc_name)
protons_gated_on_914 = degai.get_histogram(experiment, runs,  adj_dict, cal_name, (3000, 0, 3000), 'protons_gated_on_914', 'proton energy (keV) gated on 914 keV gamma rays',
                                            'tpc_energy', 'tpc_particle_id==1 && (addback_energy>912) && (addback_energy<917)&&'+time_gate_str, event_build_window, addback_ethresh, True,
                                            nonlinearity_correction_name=nlc_name)
protons_gated_on_511 = degai.get_histogram(experiment, runs, adj_dict, cal_name, (3000, 0, 3000), 'protons_gated_on_511', 'proton energy (keV) gated on 511 keV gamma rays',
                                            'tpc_energy', 'tpc_particle_id==1 && (addback_energy>509) && (addback_energy<513)&&'+time_gate_str, event_build_window, addback_ethresh, True,
                                            nonlinearity_correction_name=nlc_name)


canvas, legend, stack = root_vis_tools.draw_overlaid_histograms({'protons':protons, 'proton gated on gammas':protons_gated_on_gammas, 
                                                                 'proton gated on 491 keV gammas':protons_gated_on_491, 'proton gated on 914 keV gammas':protons_gated_on_914})

protons_gated_on_491.Rebin(10)
protons_gated_on_914.Rebin(10)
protons_gated_on_511.Rebin(10)
canvas2, legend2, stack2 = root_vis_tools.draw_overlaid_histograms({'proton gated on 491 keV gammas':protons_gated_on_491, 'proton gated on 914 keV gammas':protons_gated_on_914})
canvas3, legend3, stack3 = root_vis_tools.draw_overlaid_histograms({'proton gated on 491 keV gammas':protons_gated_on_491, 
                                                                    'proton gated on 914 keV gammas':protons_gated_on_914, 
                                                                     'proton gated on 511 keV gammas':protons_gated_on_511})
particle_gamma_dt = degai.get_histogram(experiment, runs, adj_dict, cal_name, (200, -15e-6, 15e-6), "particle_gamma_dt", "mesh_time - gamma time (s)", 
                                             "mesh_pre_amp_t - time", "tpc_particle_id==1 || tpc_particle_id==2", event_build_window, addback_ethresh, True,
                                             nonlinearity_correction_name=nlc_name)
particle_gamma491_dt = degai.get_histogram(experiment, runs, adj_dict, cal_name, (200, -15e-6, 15e-6), "particle_gamma_dt", "mesh_time - gamma time (s)", 
                                             "mesh_pre_amp_t - time", "(tpc_particle_id==1 || tpc_particle_id==2)&& (addback_energy>490) && (addback_energy<493)", event_build_window, addback_ethresh, True,
                                             nonlinearity_correction_name=nlc_name)

gammaE_v_protonE = degai.get_histogram(experiment, runs, adj_dict, cal_name, (150, 0, 3000, 7000-150, 150, 7000), "gamma_v_proton_energy_time_gate", "gamma energy (keV) vs proton energy (keV) w/ expected (mesh time - gamma time)",
                                        "addback_energy:tpc_energy", 
                                       selection='tpc_particle_id==1 &&'+time_gate_str,
                                        dt_window_ns=event_build_window, e_thresh=addback_ethresh, nonlinearity_correction_name=nlc_name)
gammaE_v_protonE_accidental = degai.get_histogram(experiment, runs, adj_dict, cal_name, (150, 0, 3000, 7000-150, 150, 7000), "gamma_v_proton_energy_accidental_gate", 
                                                  "gamma energy (keV) vs proton energy (keV) for accidental coincidences",
                                        "addback_energy:tpc_energy", 
                                       selection='tpc_particle_id==1 &&'+accidental_time_gate_str,
                                        dt_window_ns=event_build_window, e_thresh=addback_ethresh, nonlinearity_correction_name=nlc_name)
gammaE_v_protonE.Sumw2()
gammaE_v_protonE_accidental.Sumw2()
gammaE_v_protonE_bg_subtracted = gammaE_v_protonE.Clone()
gammaE_v_protonE_bg_subtracted.SetName('gammaE_v_protonE_bg_subtracted')
gammaE_v_protonE_bg_subtracted.SetTitle('gamma energy (keV) vs proton energy (keV) with accidental coincidences subtracted')
gammaE_v_protonE_bg_subtracted.Add(gammaE_v_protonE_accidental, -(tstop-tstart)/(t_accidental_stop-t_accidental_start))

h491 = degai.get_bg_subtracted_projection(gammaE_v_protonE_bg_subtracted,(488, 494), (471,486))
h491_new = degai.get_bg_subtracted_projection(gammaE_v_protonE_bg_subtracted,(488, 494), (494, 501))
h914 = degai.get_bg_subtracted_projection(gammaE_v_protonE_bg_subtracted, (911, 917), (918,927))
h1398 = degai.get_bg_subtracted_projection(gammaE_v_protonE_bg_subtracted, (1395, 1401), (1420,1430))
h511 = degai.get_bg_subtracted_projection(gammaE_v_protonE_bg_subtracted, (508, 513), (518,529))

overlay3 = root_vis_tools.draw_overlaid_histograms({'491 keV':h491, '914 keV':h914,'1398 keV':h1398, '511 keV':h511})

pscaled = protons.Clone('pscaled')
pscaled.Scale(0.01)
overlay4 = root_vis_tools.draw_overlaid_histograms({'491 keV':h491, '914 keV':h914,'1398 keV':h1398, 'all protons (scaled)':pscaled})
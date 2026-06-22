import ROOT
from raw_viewer import ddas_interface
from e23035_analysis import root_vis_tools, e23035_runs
num_workers = 100

run = 246
run = e23035_runs.get_ddas_60_Ga_runs(False, True, True, True)
mesh = ddas_interface.get_histogram(run, (1500, 0, 6000), 'mesh', 'mesh', 'mesh_pre_amp_e', 'mesh_pre_amp_e>0', num_workers=num_workers)
mesh_energy = ddas_interface.get_histogram(run, (4500, 0, 9000), 'mesh', 'mesh', '(mesh_pre_amp_e/2.5)+52', 'mesh_pre_amp_e>0', num_workers=num_workers)
tpc_energy = ddas_interface.get_histogram(run, (4500, 0, 9000), 'mesh', 'mesh', 'tpc_energy', 'tpc_energy>0', num_workers=num_workers)
_1 = root_vis_tools.draw_overlaid_histograms({'mesh':mesh_energy, 'tpc':tpc_energy})

mesh_energy_protons = ddas_interface.get_histogram(run, (4500, 0, 9000), 'mesh', 'mesh', '(mesh_pre_amp_e/2.5)+52', 'tpc_particle_id==1', num_workers=num_workers)
tpc_energy_protons = ddas_interface.get_histogram(run, (4500, 0, 9000), 'mesh', 'mesh', 'tpc_energy', 'tpc_particle_id==1', num_workers=num_workers)
_2 = root_vis_tools.draw_overlaid_histograms({'mesh':mesh_energy_protons, 'tpc':tpc_energy_protons})

mesh_vs_tpc = ddas_interface.get_histogram(run, (4500, 0, 9000, 4500, 0, 9000), 'mesh_vs_tpc_energy', 'mesh vs tpc_energy', 'mesh_pre_amp_e:tpc_energy', 'tpc_should_veto==0', num_workers=num_workers)
mesh_vs_tpc_protons = ddas_interface.get_histogram(run, (4500, 0, 9000, 4500, 0, 9000), 'mesh_vs_tpc_energy_protons', 'mesh vs tpc_energy', '(mesh_pre_amp_e/2.5)+52:tpc_energy', \
                                                   'tpc_should_veto==0 && tpc_particle_id==1', num_workers=num_workers)
mesh_vs_tpc_protons.SetXTitle('gain matched energy (keV)')
mesh_vs_tpc_protons.SetYTitle('approx mesh energy calibration (keV)')
cprotons = ROOT.TCanvas()
mesh_vs_tpc_protons.Draw()


mesh_vs_tpc_alphas = ddas_interface.get_histogram(run, (500, 0, 9000, 500, 0, 9000), 'mesh_vs_tpc_energy_alphas', 'mesh vs tpc_energy', '(mesh_pre_amp_e/2.5)+52:tpc_energy', \
                                                  'tpc_particle_id==2', num_workers=num_workers)
mesh_vs_tpc_alphas.SetXTitle('gain matched energy (keV)')
mesh_vs_tpc_alphas.SetYTitle('approx mesh energy calibration (keV)')
caphas = ROOT.TCanvas()
mesh_vs_tpc_alphas.Draw()

mesh_rve = ddas_interface.get_histogram(run, (900, 0, 9000, 200, 0, 200), 'mesh_RvE', 'Range vs Mesh Energy', 'tpc_track_length:(mesh_pre_amp_e/2.5)+52', 
                                        '!tpc_should_veto',num_workers=num_workers)
cmesh_rve = ROOT.TCanvas()
mesh_rve.Draw('colz')
tpc_rve = ddas_interface.get_histogram(run, ( 900, 0, 9000, 200, 0, 200), 'tpc_RvE', 'Range vs TPC Energy', 'tpc_track_length:tpc_energy', 
                                        '!tpc_should_veto',num_workers=num_workers)
ctpc_rve = ROOT.TCanvas()
tpc_rve.Draw('colz')

mesh_vs_angle = ddas_interface.get_histogram(run, (900, 0, 9000, 90, 0, 90), 'mesh_vs_angle', 'Track Angle vs Mesh Energy', 'tpc_track_angle:(mesh_pre_amp_e/2.5)+52', 
                                        '!tpc_should_veto && tpc_track_length>140 && tpc_track_length<180',num_workers=num_workers)
cmesh_vs_angle = ROOT.TCanvas()
mesh_vs_angle.Draw('colz')
cmesh_vs_angle.SetLogz(1)
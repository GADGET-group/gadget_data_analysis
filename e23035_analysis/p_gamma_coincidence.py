import os

import ROOT
import numpy as np

from raw_viewer import ddas_interface, process_runs
from e23035_analysis import e23035_runs, fitting_tools


runs = range(241,255)

n_workers = len(runs)

gammas = ddas_interface.get_histogram(runs, (12000-1,1,12000), 'gammas', 'summed gamma spectrum', ddas_interface.get_summed_gamma_e_str(), num_workers=n_workers)
s = ddas_interface.get_summed_gamma_e_str()
gamma_gated_on_proton = ddas_interface.get_histogram(runs, (12000-1,1,12000), 'gamma_gated_on_protons', 'gamma rays gated on protons', 
                ddas_interface.get_summed_gamma_e_str(), 'tpc_particle_id==1', num_workers=n_workers)

protons = ddas_interface.get_histogram(runs, (3000, 0, 3000), 'protons', 'proton energy (keV)', 'tpc_energy', 'tpc_particle_id==1', num_workers=n_workers)
protons_gated_on_gammas = ddas_interface.get_histogram(runs, (3000, 0, 3000), 'protons_gated_on_gammas', 'proton energy (keV) gated on gamma rays',
                                                        'tpc_energy', 'tpc_particle_id==1 && (%s>0)'%s, num_workers=n_workers)

protons_gated_on_914 = ddas_interface.get_histogram(runs, (3000, 0, 3000), 'protons_gated_on_914', 'protons gated on 914 keV gamma',
                                                        'tpc_energy', 'tpc_particle_id==1 && (%s>913) && (%s < 917)'%(s,s), num_workers=n_workers)


c1,c2 = ROOT.TCanvas(), ROOT.TCanvas()
c1.cd()
gammas.Draw()
ROOT.gPad.SetLogy(1)
c2.cd()
gamma_gated_on_proton.Draw()
ROOT.gPad.SetLogy(1)
'''
Minimal example of a coincidence between TPC protons and gamma rays of any energy.

For each germanium crystal, histogram the time between a proton's mesh signal and that crystal's hit in the same
DDAS event. Gamma rays emitted with the proton make a peak at positive mesh time - gamma time (the electron drift
time); random gamma rays make a flat background, which is subtracted using a window where the gamma comes after the mesh.
Counts are proton-crystal pairs, so a gamma ray which scatters between crystals is counted more than once.
'''
import ctypes
import os

import ROOT
import numpy as np

from tpc_and_decay_station import ddas_interface

experiment = 'e25058'
runs = [171]
tpc_config = 'smart1_veto2_rpr_fzp.csv'
n_workers = min(len(runs), 150) #cap at 150 so as not to murder the TPCGPU

crystals = [name for name in ddas_interface.read_channel_map(experiment)[1] if name.startswith('clover')]

#windows on mesh time - gamma time (us). The merged files rebuild each GET event's entry from the hits within 15 us of
#the trigger, which comes 4.3 us after the mesh, so the entry holds gammas from 10.7 us before to 19.3 us after the mesh.
#The prompt peak (the drift time) runs from 0.5 to 7.4 us in run 136.
prompt_window = (0, 7.5)
accidental_window = (-18, -2)

#proton energy vs mesh time - gamma time, summed over crystals
hists = [ddas_interface.get_histogram(experiment, runs, (120, -20, 10, 300, 0, 3000), f'dt_vs_proton_energy_{crystal}', crystal,
                                      f'tpc_energy:(mesh_pre_amp_t - {crystal}_t)*1e6', f'tpc_particle_id==1 && {crystal}_m>0',
                                      num_workers=n_workers, tpc_ini_filename=tpc_config) for crystal in crystals]
dt_vs_proton_energy = hists[0].Clone('dt_vs_proton_energy')
dt_vs_proton_energy.SetTitle('proton-gamma pairs;mesh time - gamma time (#mus);proton energy (keV)')
for h in hists[1:]:
    dt_vs_proton_energy.Add(h)

def protons_in_window(window, name):
    axis = dt_vs_proton_energy.GetXaxis()
    return dt_vs_proton_energy.ProjectionY(name, axis.FindBin(window[0]), axis.FindBin(window[1]) - 1, 'e')

coincident_protons = protons_in_window(prompt_window, 'coincident_protons')
accidental_scale = (prompt_window[1] - prompt_window[0])/(accidental_window[1] - accidental_window[0])
coincident_protons.Add(protons_in_window(accidental_window, 'accidental_protons'), -accidental_scale)
coincident_protons.SetTitle('protons in coincidence with a gamma ray, accidentals subtracted;proton energy (keV);proton-gamma pairs')

err = ctypes.c_double()
n_pairs = coincident_protons.IntegralAndError(1, coincident_protons.GetNbinsX(), err)
print(f'{n_pairs:.0f} +- {err.value:.0f} proton-gamma pairs above accidentals')

c1 = ROOT.TCanvas()
dt = dt_vs_proton_energy.ProjectionX('dt')
dt.SetTitle('protons;mesh time - gamma time (#mus);proton-gamma pairs')
dt.Draw()
c2 = ROOT.TCanvas()
coincident_protons.Draw()

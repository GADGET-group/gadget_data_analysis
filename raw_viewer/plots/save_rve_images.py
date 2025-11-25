import os
import pickle

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.path import Path
import matplotlib.colors
import ROOT

from raw_viewer import process_runs
from  raw_viewer import raw_h5_file

if False: #background runs before experiment
    experiment = 'e23035_prep_vault'
    run_range = (68, 73)
else: #during experiment
    experiment = 'e23035'
    run_range = (0,1000)#(101, 143)

exclude_runs = [1,9, 73, 113]
runs = []
for run in range(run_range[0], run_range[1]+1):
    if run not in exclude_runs and os.path.exists(process_runs.get_h5_path(experiment, run)):
        runs.append(run)

for r in runs:
    run = [r]
    veto_thresh = 400
    rve_bins = (300, 300)

    #load pad gain match
    gain_match_path = '/egr/research-tpc/shared/e23035_prep/vault/gm.pkl'
    with open(gain_match_path, 'rb') as f:
        gain_match_result = pickle.load(f)
    pad_gains = gain_match_result.x

    lengths = process_runs.get_lengths(experiment, run)
    cpp = process_runs.get_quantity('pad_charge', experiment, run)
    #veto_counts = process_runs.get_veto_counts(exp, runs)
    veto_max = process_runs.get_max_veto_counts(experiment, run)
    charge_widths = process_runs.get_quantity('charge_width', experiment,run)
    energy = process_runs.get_gm_ic(experiment, run, pad_gains)

    veto_mask = (veto_max < veto_thresh)

    fig, ax = plt.subplots()
    plt_mask = veto_mask&(lengths<400)&(lengths>1)
    ax.set_title('runs: '+str(run))
    ax.hist2d(energy[plt_mask], lengths[plt_mask], bins=rve_bins, norm=matplotlib.colors.LogNorm())
    plt.xlabel('energy (MeV)')
    plt.savefig('/egr/research-tpc/shared/e23035/images/rve_run%d.png'%r)
    plt.close(fig)


    # self.poly_selector = matplotlib.widgets.PolygonSelector(ax,self.set_cut_polygon)
    # self.poly_selector = matplotlib.widgets.PolygonSelector(ax,self.set_cut_polygon)
    # if len(self.rve_cut_verticies) > 0:
    #     self.poly_selector.verts = self.rve_cut_verticies
    # fig.show()
    m = (108.4-32.2)/(3.628-2.25)
    alpha_mask = veto_mask&(lengths<(energy*m+32.2 - m*2.25))

    m = (159.2-26.2)/(2.81-0.619)
    proton_mask = veto_mask&(~alpha_mask)&(lengths<(energy*m+26.6 - m*0.619))

    plt.figure()
    plt.hist(energy[proton_mask], 1500)
    plt.title('proton energy spectrum, runs: '+str(run))
    plt.xlabel('energy (MeV)')
    plt.savefig('/egr/research-tpc/shared/e23035/images/proton_spectra/proton_spectrum_run%d.png'%r)
    plt.close()

    plt.figure()
    plt.title('protons selected in RVE, runs: '+str(run))
    plt.hist2d(energy[plt_mask], lengths[plt_mask], bins=rve_bins, norm=matplotlib.colors.LogNorm())
    plt.scatter(energy[proton_mask], lengths[proton_mask], marker='.', alpha=0.5, color='red')
    print(str(r) + "has " + str(len(proton_mask[proton_mask])) + " protons")
    plt.savefig('/egr/research-tpc/shared/e23035/images/proton_spectra/selected_protons_run%d.png'%r)
    plt.close()

    plt.figure()
    plt.hist(energy[alpha_mask], 200)
    plt.title('alpha energy spectrum, runs: '+str(run))
    plt.xlabel('energy (MeV)')
    plt.savefig('/egr/research-tpc/shared/e23035/images/alpha_spectra/alpha_spectrum_run%d.png'%r)
    plt.close()

    plt.figure()
    plt.title('alphas selected in RVE, runs: '+str(run))
    plt.hist2d(energy[plt_mask], lengths[plt_mask], bins=rve_bins, norm=matplotlib.colors.LogNorm())
    plt.scatter(energy[alpha_mask], lengths[alpha_mask], marker='.', alpha=0.5, color='red')
    print(str(run) + "has " + str(len(alpha_mask[alpha_mask])) + " alphas")
    plt.savefig('/egr/research-tpc/shared/e23035/images/alpha_spectra/selected_alphas_run%d.png'%r)
    plt.close()




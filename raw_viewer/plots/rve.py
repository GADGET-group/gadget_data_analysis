import os
import pickle

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.path import Path
import matplotlib.colors
import ROOT
import numpy as np
from tqdm import tqdm
from scipy import optimize

from raw_viewer import process_runs
from  raw_viewer import raw_h5_file
from raw_viewer import ddas_interface
from e23035_analysis import e23035_runs

experiment = 'e23035'#_prep_vault'
 
if experiment == 'e23035':
    #run_range = e23035_runs.run_df['GET'][(e23035_runs.run_df['Run Type']=='60Ga')]# & (e23035_runs.run_df['Field Cage Functional?'] == 'yes')]
    run_range = [145]
#     exclude_runs = [1,9, 19, 73, 113,
#                     132, #run missing some CoBos
#                     210,216, 225, 226, 227, 228, 229, #210 needs to be transfered by Tyler
#                     289,290, 291, 292, 293, 294, 295, 296, 297, 298]#41 deg angle runs
if experiment == 'e23035_prep_vault': # runs before experiment
    #run_range = (17,20,21)
    #run_range = np.arange(61, 63+1) #calibration before experiment
    #run_range = [17, 20, 21, 38, 49, 60, 61, 62, 63] #runs used for GM
    #run_range = np.arange(68, 73+1) #background before experiemnt 
    run_range = [49]#[61, 62, 63]
    

# else: #during experiment
#     #run_range = np.concatenate((np.arange(140,219+1), np.arange(263, 279+1)))#60Ga with SCA set ~300 keV
#     #run_range = np.arange(263, 279+1)#60Ga with SCA set ~300 keV & 0.5 us gate delay
#     #run_range = np.arange(220, 263+1)#60Ga with SCA set ~1000 keV
#     #run_range = np.concatenate((np.arange(281,286+1), np.arange(296,301+1)))#59Zn with field cage on
#     run_range = [285]

exclude_runs = []#[1,9, 73, 113]

#get_runs=range(145,151)
#runs=[280] #background after experiment
#runs=range(145, 150+1)#152+1)
#
#srun_range = [299,300,301]
get_runs = []
for run in run_range:
    if not np.isnan(run):
        if run not in exclude_runs and os.path.exists(process_runs.get_h5_path(experiment, run)):
            get_runs.append(run)

get_runs = np.sort(get_runs)
print('get_runs:', get_runs)

load_ddas = False

if load_ddas:
    ddas_runs = e23035_runs.get_DDAS_run_number(get_runs)
    print(ddas_runs)
    print('getting times since beam off for each event')
    times_since_beam_off = []
    for ddas_run, get_run in tqdm(zip(ddas_runs, get_runs)):
        get_ts = process_runs.get_quantity('timestamps', experiment, [get_run])
        times_since_beam_off.append(ddas_interface.get_time_since_beam_off(experiment, ddas_run)[:-1])
        print(get_run, len(times_since_beam_off[-1]), len(get_ts))
    times_since_beam_off = np.concatenate(times_since_beam_off)

veto_thresh = 500#np.inf
rve_bins = (300, 300)
phist_bins = np.linspace(0, 4, 1001)
alphahist_bins = 100

print('loading stuff')


lengths = process_runs.get_lengths(experiment, get_runs)
print('lengths loaded')
cpp = process_runs.get_quantity('pad_charge', experiment, get_runs)
#veto_counts = process_runs.get_veto_counts(exp, runs)
veto_max = process_runs.get_max_veto_counts(experiment, get_runs)
charge_widths = process_runs.get_quantity('charge_width', experiment,get_runs)
#energy = process_runs.get_gm_ic(experiment, get_runs, pad_gains)
if experiment == 'e23035':
    energy = e23035_runs.get_energy_MeV(get_runs)
else:
    gain_match_path = '/egr/research-tpc/adamsa52/gadget_analysis/fft6_res3.pkl'
    with open(gain_match_path, 'rb') as f:
        gain_match_result = pickle.load(f)
    #pad_gains = gain_match_result.x[:1024]
    pad_gains = gain_match_result.pad_gains
    energy = process_runs.get_gm_ic(experiment, get_runs, pad_gains)
angles = process_runs.get_angle(experiment, get_runs)

##&(angles>np.radians(5))#process_runs.get_outer_ring_counts(experiment, runs)<113#
pads_railed = process_runs.get_quantity('railed_pads', experiment, get_runs)
num_pads_railed = np.array([len(prl) for prl in pads_railed])
if experiment == 'e23035':
    veto_mask = e23035_runs.get_veto_mask(get_runs)
    endpoints = process_runs.get_quantity('endpoints', experiment, get_runs)
    min_z = np.min(endpoints[:,:,2], axis=1)
    #veto_mask = veto_mask&(min_z>5)
else:
    veto_mask = (veto_max < veto_thresh)
veto_mask = veto_mask & (num_pads_railed==0)    

    
if load_ddas:
    print(len(times_since_beam_off), len(energy))
    assert len(times_since_beam_off) == len(energy)

print('starting plotting')
plt.figure()
plt_mask = veto_mask&(lengths>1)&(lengths<400)
plt.title('runs: '+str(get_runs))
plt.hist2d(energy[plt_mask], lengths[plt_mask], bins=rve_bins, norm=matplotlib.colors.LogNorm())
plt.colorbar()
plt.xlabel('energy (MeV)')
plt.ylabel('range (mm)')


# self.poly_selector = matplotlib.widgets.PolygonSelector(ax,self.set_cut_polygon)
# self.poly_selector = matplotlib.widgets.PolygonSelector(ax,self.set_cut_polygon)
# if len(self.rve_cut_verticies) > 0:
#     self.poly_selector.verts = self.rve_cut_verticies
# fig.show()
m = (108.4-32.2)/(3.628-2.25)
m2 = (23.3-35.5)/(1-2.4)
#alpha_mask = veto_mask&(lengths<(energy*m2+35.5 - m2*2.4))&(energy>2.7)#((lengths<(energy*m+32.2 - m*2.25))|((lengths<(energy*m2+35.5 - m2*2.4))&(energy<2.4)))
if experiment == 'e23035':
    alpha_mask = e23035_runs.get_alpha_mask(get_runs)

    m = (159.2-26.2)/(2.81-0.619)
    #proton_mask = veto_mask&(~alpha_mask)&(lengths<(energy*m+26.6 - m*0.619))&(energy>0.3)
    proton_mask = e23035_runs.get_proton_mask(get_runs)
    #proton_mask = veto_mask&(~alpha_mask)&(lengths<(energy*m+26.6 - m*0.619))&(energy>0.95)&(energy<2.2)&(energy>1.5)&(lengths>55)
    # palpha_cut = veto_mask&(energy>1.6)&(energy<1.8)&(lengths>27.5)&(lengths<40)
    # print(np.where(palpha_cut))

    plt.figure()
    plt.hist(energy[proton_mask], phist_bins)
    plt.title('proton energy spectrum, runs: '+str(get_runs))
    plt.xlabel('energy (MeV)')
    plt.ylabel('counts/keV')
    #plt.yscale('log')

    plt.figure()
    plt.title('protons selected in RVE, runs: '+str(get_runs))
    plt.hist2d(energy[plt_mask], lengths[plt_mask], bins=rve_bins, norm=matplotlib.colors.LogNorm())
    plt.scatter(energy[proton_mask], lengths[proton_mask], marker='.', alpha=0.5, color='red')
    plt.colorbar()
    print(str(get_runs) + "has " + str(len(proton_mask[proton_mask])) + " protons")

    timestamps = process_runs.get_quantity('timestamps', experiment, get_runs)
    time_since_last_event = timestamps - np.roll(timestamps, 1)
    time_since_last_event[0] = .15 #we don't actuallly know what this is for the first event, so just putting a typical value for start of window

    plt.figure()
    plt.hist(energy[alpha_mask], alphahist_bins)
    plt.title('alpha energy spectrum, runs: '+str(get_runs))
    plt.xlabel('energy (MeV)')

    plt.figure()
    plt.title('alphas selected in RVE, runs: '+str(get_runs))
    plt.hist2d(energy[plt_mask], lengths[plt_mask], bins=rve_bins, norm=matplotlib.colors.LogNorm())
    plt.scatter(energy[alpha_mask], lengths[alpha_mask], marker='.', alpha=0.5, color='red')
    plt.colorbar()
    print(str(get_runs) + "has " + str(len(alpha_mask[alpha_mask])) + " alphas")


    #TODO: correct for times runs were not instantly started again after previous run ended
    run_t_offset = [0]
    run_ts = []
    for ddas_run in get_runs:
        run_ts.append(process_runs.get_quantity('timestamps', experiment, [ddas_run]))
    for i in range(1, len(get_runs)):
        if run_ts[i][0] <= run_ts[i-1][-1]:
            run_t_offset.append(run_t_offset[-1] + run_ts[i-1][-1])
        else:
            run_t_offset.append(run_t_offset[-1])
    for i in range(len(get_runs)):
        run_ts[i] = run_ts[i] + run_t_offset[i]
    run_ts = np.concatenate(run_ts)

    plt.figure()
    plt.title('alphas')
    tve_bins = (100, 15)
    plt.hist2d(energy[alpha_mask], run_ts[alpha_mask]/3600, bins=tve_bins, norm=matplotlib.colors.LogNorm())
    plt.xlabel('energy (MeV)')
    plt.ylabel('time since start of experiment (hours)')
    plt.colorbar()

if load_ddas:
    plt.figure()
    plt.title('protons')
    tsbo=times_since_beam_off*1e3#process_runs.get_time_since_beam_off(experiment, runs)
    tve_bins = (50, 50)
    plt.hist2d(energy[proton_mask&(tsbo>0)], tsbo[proton_mask&(tsbo>0)], bins=tve_bins)#, norm=matplotlib.colors.LogNorm())
    plt.xlabel('energy (MeV)')
    plt.ylabel('time since beam off (ms)')
    plt.colorbar()

    plt.figure()
    plt.title('alphas')
    tve_bins = (50, 10)
    plt.hist2d(energy[alpha_mask&(tsbo>0)], tsbo[alpha_mask&(tsbo>0)], bins=tve_bins)#, norm=matplotlib.colors.LogNorm())
    plt.xlabel('energy (MeV)')
    plt.ylabel('time since beam off (ms)')
    plt.colorbar()

    plt.figure()
    plt.hist(tsbo, bins=100)
    plt.xlabel('time since beam off')

plt.show(block=False)

# hist = ROOT.TH1D('Ep', 'Ep', 4000, 0, 4)
# hist.Fill(energy[proton_mask])
# hist.Draw()

# np.save('to_fit.npy', energy[proton_mask])

plt.show(block=False)

def fit_decay_exponential(mask, time_bin_edges, guess=(100,100)):
    plt.figure()
    counts, bins, patches = plt.hist(tsbo[mask], time_bin_edges)
    bin_centers = (bins[:-1] + bins[1:])/2
    sigma = np.sqrt(counts)
    sigma[sigma==1] = 1
    to_fit = lambda t, A, tau: A*np.exp(-t/tau)
    #plt.plot(bin_centers, to_fit(bin_centers, *guess))
    popt, pcov = optimize.curve_fit(to_fit, bin_centers, counts, guess, sigma)
    plt.plot(bin_centers, to_fit(bin_centers, *popt))
    plt.figure()
    plt.plot(bin_centers, counts - to_fit(bin_centers,*popt))
    plt.show(block=False)
    print('half life: ', popt[1]*np.log(2), "+/-", np.sqrt(pcov[1,1])*np.log(2), 'ms')
    return popt, pcov

def fit_double_decay_exponential(mask, time_bin_edges, guess=(100,60, 100, 100)):
    plt.figure()
    counts, bins, patches = plt.hist(tsbo[mask], time_bin_edges)
    bin_centers = (bins[:-1] + bins[1:])/2
    sigma = np.sqrt(counts)
    sigma[sigma==0] = 1
    to_fit = lambda t, A1, tau1, A2, tau2: A1*np.exp(-t/tau1) + A2*np.exp(-t/tau2)
    #plt.plot(bin_centers, to_fit(bin_centers, *guess))
    popt, pcov = optimize.curve_fit(to_fit, bin_centers, counts, guess, sigma)
    plt.plot(bin_centers, to_fit(bin_centers, *popt))
    plt.figure()
    plt.plot(bin_centers, counts - to_fit(bin_centers,*popt))
    plt.show(block=False)
    print('half lives: ', popt[1]*np.log(2), "+/-", np.sqrt(pcov[1,1])*np.log(2), 'ms, ',
           popt[3]*np.log(2), "+/-", np.sqrt(pcov[3,3])*np.log(2), 'ms')
    return popt, pcov

rve_cut_select_mask = None
def set_cut_polygon(verticies):
        '''
        verticies: (counts, ranges)
        '''
        global rve_cut_select_mask
        print(verticies)
        rve_cut_verticies = verticies
        selected_rve_path = matplotlib.path.Path(rve_cut_verticies)
        rve_points = np.vstack((energy[plt_mask], lengths[plt_mask])).transpose()
        rve_cut_select_mask = selected_rve_path.contains_points(rve_points)

poly_selector = None
def define_cut_on_gui():
    global poly_selector
    #open a RvE histogram with the current settings
    fig, ax = plt.subplots()
    ax.hist2d(energy[plt_mask], lengths[plt_mask], bins=rve_bins, norm=matplotlib.colors.LogNorm())
    ax.set_xlabel('adc counts')
    ax.set_ylabel('range (mm)')
    poly_selector = matplotlib.widgets.PolygonSelector(ax,set_cut_polygon)
    # if len(self.rve_cut_verticies) > 0:
    #     self.poly_selector.verts = self.rve_cut_verticies
    fig.show()

evt_runs, evt_nums = process_runs.get_run_and_event_numbers(experiment, get_runs)
def show_selected_event(i):
    run = evt_runs[plt_mask][rve_cut_select_mask][i]
    evt = evt_nums[plt_mask][rve_cut_select_mask][i]
    show_event(run,evt)
    

def show_event(run, evt):
    h5file = process_runs.get_h5_file(experiment, run)
    h5file.show_2d_projection(evt, block=False)
    h5file.plot_3d_traces(evt, threshold=h5file.length_counts_threshold, block=False)
    h5file.plot_traces(evt, block=False)
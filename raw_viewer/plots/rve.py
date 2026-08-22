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
from track_fitting import srim_interface, build_sim

experiment = 'e23035'
GPUs_to_use = [0,1,2,3]
max_workers=100
tpc_config = 'smart2_nrpr.csv'
 
if experiment == 'e23035':
    if True:
        run_range = e23035_runs.run_df['GET'][(e23035_runs.run_df['Run Type']=='60Ga')  & (e23035_runs.run_df['final beam settings?'] == 'yes')] 
        #run_range = range(263, 280)
        #run_range = [170, 171, 172, 173]
    else:
        run_range = e23035_runs.run_df['GET'][(e23035_runs.run_df['Run Type']=='59Zn') & (e23035_runs.run_df['Field Cage Functional?'] == 'yes')]
    #run_range=[148,149,150]
    #run_range = [145]
#     exclude_runs = [1,9, 19, 73, 113,
#                     132, #run missing some CoBos
#                     210,216, 225, 226, 227, 228, 229, #210 needs to be transfered by Tyler
#                     289,290, 291, 292, 293, 294, 295, 296, 297, 298]#41 deg angle runs
if experiment == 'e23035_prep_vault': # runs before experiment
    #run_range = (17,20,21)
    #run_range = np.arange(61, 63+1) #calibration before experiment
    #run_range = [17, 20, 21, 38, 49, 60, 61, 62, 63] #runs used for GM
    #run_range = np.arange(68, 73+1) #background before experiemnt 
    run_range = [49, 16,20,35,61, 62, 63]
    

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

print('loading stuff')
num_workers = min(max_workers, len(get_runs))

quantities_to_get = ['charge_width', 'endpoints', 'timestamps']
if experiment != 'e23035':
    quantities_to_get.append('railed_pads')
    
results = process_runs.get_quantity(quantities_to_get, experiment, get_runs, show_load_progress=False, num_workers=num_workers, gpus_to_use=GPUs_to_use, config_filename=tpc_config)
charge_widths = results[0]
endpoints = results[1]
timestamps = results[2]
if experiment != 'e23035':
    pads_railed = results[3]
print('root data loaded')

run_numbers, event_numbers = process_runs.get_run_and_event_numbers(experiment, get_runs, config_filename=tpc_config)

if load_ddas:
    ddas_runs = e23035_runs.get_DDAS_run_number(get_runs)
    print(ddas_runs)
    print('getting times since beam off for each event')
    times_since_beam_off = []
    for ddas_run, get_run in tqdm(zip(ddas_runs, get_runs)):
        get_ts = timestamps[run_numbers == get_run]
        times_since_beam_off.append(ddas_interface.get_time_since_beam_off(experiment, ddas_run)[:-1])
        print(get_run, len(times_since_beam_off[-1]), len(get_ts))
    times_since_beam_off = np.concatenate(times_since_beam_off)

veto_thresh = 500#np.inf
rve_bins = (600, 600)
phist_bins = np.linspace(0, 4, 1001)
alphahist_bins = 100
lengths = process_runs.get_lengths(endpoints)
angles = process_runs.get_angle(endpoints)
print('lengths and angles calculated')

#cpp = process_runs.get_quantity('pad_charge', experiment, get_runs)
#veto_counts = process_runs.get_veto_counts(exp, runs)
veto_max = process_runs.get_max_veto_counts(experiment, get_runs, num_workers=num_workers, config_filename=tpc_config)
print('veto max loaded')


#energy = process_runs.get_gm_ic(experiment, get_runs, pad_gains)
if experiment == 'e23035':
    if True: #use default gain match
        energy = e23035_runs.get_energy_MeV(get_runs, num_workers=num_workers, tpc_ini_filename=tpc_config)
    else:
        # gain_match_path = '/egr/research-tpc/adamsa52/gadget_analysis/raw_viewer/pad_gain_match/gain_match_results/fft6_res3.pkl'
        # with open(gain_match_path, 'rb') as f:
        #     gain_match_result = pickle.load(f)
        # pad_gains = gain_match_result.pad_gains
        energy = process_runs.get_gm_ic(experiment, get_runs, pad_gains, num_workers=num_workers, config_filename=tpc_config) 
else:
    gain_match_path = '/egr/research-tpc/adamsa52/gadget_analysis/raw_viewer/pad_gain_match/gain_match_results/gm_old/fft6_res3.pkl'
    with open(gain_match_path, 'rb') as f:
        gain_match_result = pickle.load(f)
    #pad_gains = gain_match_result.x[:1024]
    pad_gains = gain_match_result.pad_gains
    #pad_gains = np.ones(np.shape(gain_match_result.pad_gains))*np.mean(gain_match_result.pad_gains)
    energy = process_runs.get_gm_ic(experiment, get_runs, pad_gains, config_filename=tpc_config)
print('energies loaded')

##&(angles>np.radians(5))#process_runs.get_outer_ring_counts(experiment, runs)<113#


if experiment == 'e23035':
    veto_mask = e23035_runs.get_veto_mask(endpoints=endpoints, max_veto_counts=veto_max, tpc_ini_filename=tpc_config)
else:
    num_pads_railed = np.array([len(prl) for prl in pads_railed])
    veto_mask = (veto_max < veto_thresh)
    veto_mask = veto_mask #& (num_pads_railed==0)# & (angles>np.radians(8)) 

    
if load_ddas:
    print(len(times_since_beam_off), len(energy))
    assert len(times_since_beam_off) == len(energy)

print('starting plotting')
plt.figure()
plt_mask = veto_mask&(lengths>1)&(lengths<250)&(energy<10)
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
#alpha_mask = veto_mask&(lengths<(energy*m2+35.5 - m2*2.4))&(energy>2.7)#((lengths<(energy*m+32.2 - m*2.25))|((lengths<(energy*m2+35.5 - m2*2.4))&(energy<2.4)))
if experiment == 'e23035':
    alpha_mask = e23035_runs.get_alpha_mask(get_runs, lengths=lengths, energy=energy, veto_mask=veto_mask, tpc_ini_filename=tpc_config)

    #proton_mask = veto_mask&(~alpha_mask)&(lengths<(energy*m+26.6 - m*0.619))&(energy>0.3)
    proton_mask = e23035_runs.get_proton_mask(get_runs, lengths=lengths, energy=energy, veto_mask=veto_mask, tpc_ini_filename=tpc_config)
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
    #plt.scatter(energy[proton_mask], lengths[proton_mask], marker='.', alpha=0.5, color='red')
    stopping_power_path = 'track_fitting/stopping_powers/%s_in_%s.txt'%('1H', 'P10')
    proton_srim_table = srim_interface.SRIM_Table(stopping_power_path, build_sim.get_gas_density('e23035', get_runs[0]))
    es_to_plot_ranges = np.linspace(0, 3.5, 1000)
    expected_proton_length = proton_srim_table.get_stopping_distance(es_to_plot_ranges)
    lower_proton_band, upper_proton_band = e23035_runs.get_proton_mask_min_max_range(get_runs, es_to_plot_ranges)
    plt.plot(es_to_plot_ranges, expected_proton_length, 'r')
    plt.plot(es_to_plot_ranges, lower_proton_band, 'k')
    plt.plot(es_to_plot_ranges, upper_proton_band, 'k')
    
    plt.colorbar()
    print(str(get_runs) + "has " + str(len(proton_mask[proton_mask])) + " protons")

    time_since_last_event = timestamps - np.roll(timestamps, 1)
    time_since_last_event[0] = .15 #we don't actuallly know what this is for the first event, so just putting a typical value for start of window

    plt.figure()
    plt.hist(energy[alpha_mask], alphahist_bins)
    plt.title('alpha energy spectrum, runs: '+str(get_runs))
    plt.xlabel('energy (MeV)')

    plt.figure()
    plt.title('alphas selected in RVE, runs: '+str(get_runs))
    plt.hist2d(energy[plt_mask], lengths[plt_mask], bins=rve_bins, norm=matplotlib.colors.LogNorm())
    es_to_plot_ranges = np.linspace(0, 9, 1000)
    lower_alpha_band, upper_alpha_band = e23035_runs.get_alpha_mask_min_max_range(get_runs, es_to_plot_ranges)
    stopping_power_path = 'track_fitting/stopping_powers/%s_in_%s.txt'%('4He', 'P10')
    alpha_srim_table = srim_interface.SRIM_Table(stopping_power_path, build_sim.get_gas_density('e23035', get_runs[0]))
    expected_alpha_length = alpha_srim_table.get_stopping_distance(es_to_plot_ranges)
    plt.plot(es_to_plot_ranges, expected_alpha_length, 'r')
    plt.plot(es_to_plot_ranges, lower_alpha_band, 'k')
    plt.plot(es_to_plot_ranges, upper_alpha_band, 'k')
    plt.colorbar()
    print(str(get_runs) + "has " + str(len(alpha_mask[alpha_mask])) + " alphas")


    #TODO: correct for times runs were not instantly started again after previous run ended
    run_t_offset = [0]
    run_ts = []
    for get_run in get_runs:
        run_ts.append(timestamps[run_numbers == get_run])
    for i in range(1, len(get_runs)):
        if len(run_ts[i]) > 0 and len(run_ts[i-1]) > 0:
            if run_ts[i][0] <= run_ts[i-1][-1]:
                run_t_offset.append(run_t_offset[-1] + run_ts[i-1][-1])
            else:
                run_t_offset.append(run_t_offset[-1])
        else:
            run_t_offset.append(run_t_offset[-1])
    for i in range(len(get_runs)):
        if len(run_ts[i]) > 0:
            run_ts[i] = run_ts[i] + run_t_offset[i]
    if len(run_ts) > 0:
        run_ts = np.concatenate(run_ts)
    else:
        run_ts = np.array([])

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

evt_runs, evt_nums = process_runs.get_run_and_event_numbers(experiment, get_runs, config_filename=tpc_config)
def show_selected_event(i):
    run = evt_runs[plt_mask][rve_cut_select_mask][i]
    evt = evt_nums[plt_mask][rve_cut_select_mask][i]
    print('run %d evt %d energy %f MeV'%(run, evt, energy[plt_mask][rve_cut_select_mask][i]))
    show_event(run,evt)
    

def show_event(run, evt):
    h5file = process_runs.get_h5_file(experiment, run, config_filename=tpc_config)
    h5file.show_2d_projection(evt, block=False)
    h5file.plot_3d_traces(evt, threshold=h5file.length_counts_threshold, block=False)
    h5file.plot_traces(evt, block=False)

if experiment != 'e23035_prep_vault' and len(get_runs) > 1:
    plt.figure()
    proton_mask = proton_mask & (energy<3.5)
    plt.title('protons')
    plt.hist2d(energy[proton_mask],evt_runs[proton_mask],bins=(300,np.arange(min(get_runs)-0.5, max(get_runs)+0.5)),  norm=matplotlib.colors.LogNorm())
    plt.xlabel('energy (MeV)')
    plt.ylabel('run number')
    plt.colorbar()
    plt.show(block=False)

    plt.figure()
    alpha_mask = alpha_mask & (energy<10)
    plt.title('alphas')
    plt.hist2d(energy[alpha_mask],evt_runs[alpha_mask],bins=(50,np.arange(min(get_runs)-0.5, max(get_runs)+0.5)),  norm=matplotlib.colors.LogNorm())
    plt.xlabel('energy (MeV)')
    plt.ylabel('run number')
    plt.colorbar()
    plt.show(block=False)
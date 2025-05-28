import os
import pickle
import sys
import time

import numpy as np
from tqdm import tqdm
import matplotlib.pylab as plt
import matplotlib.colors
import scipy.optimize as opt

from track_fitting.field_distortion import extract_track_axis_info
from track_fitting import build_sim

experiment, run, N = 'e21072', 124, 1

#list of (wieght, peak label) tuples. Objective function will include minimizing sum_i weight_i * std(peak i range)^2
peak_widths_to_minimize = [(1, 'p1596'),  (1, 'a4434'), (1, 'p770'), (1, 'a2153')]
#list of (weight, peak 1, peak 2) tuples.
#Objective function will minimize sum_i weight_i ((mean(peak i1 range) - mean(peaki2 range) - (true peak i2 range - true peak i2 range))^2
peak_spacings_to_preserve = [(1, 'a2153', 'a4434'), (1, 'p770', 'p1596'), (1, 'p1596', 'a2153')]

use_pca_for_width = False #if false, uses standard deviation of charge along the 2nd pca axis
exploit_symmetry = False #Assumes positive ions spread out quickly: f(r,w,t)=f0(r, sqrt(w^2 - kt))
phi_dependence = False #currently only works if exploit symetry is True
allow_beam_off_axis = True #if false, will assume electric field is centered at (0,0)
opt_method = 'local'#annealing, shgo, or local
t_bounds = False
t_lower = 0.0
t_upper = 0.02

offset_endpoints = True



#include up to 4 particles to make scatter plots and histograms for
particles_to_plot = ['p1596', 'p770', 'a2153', 'p1927']#'a4434']


track_info_dict = extract_track_axis_info.get_track_info(experiment, run)
endpoints = np.array(track_info_dict['endpoints'])

processed_directory = '/egr/research-tpc/shared/Run_Data/run_%04d_raw_viewer/run_%04dsmart'%(run, run)

#load histogram arrays
counts = np.load(os.path.join(processed_directory, 'counts.npy'))
dxys = np.load(os.path.join(processed_directory, 'dxy.npy'))
dts = np.load(os.path.join(processed_directory, 'dt.npy'))
max_veto_counts = np.load(os.path.join(processed_directory, 'veto.npy'))
timestamps = np.load(os.path.join(processed_directory, 'timestamps.npy'))

h5file = build_sim.get_rawh5_object(experiment, run)
dzs = dts*h5file.zscale
MeV = build_sim.get_integrated_charge_energy_offset(experiment, run) + counts/build_sim.get_adc_counts_per_MeV(experiment, run)
#ranges = np.sqrt(dzs**2 + dxys**2) #why is this different than ds = np.linalg.norm(endpoints[:,0] - endpoints[:,1], axis=1)?
ds = np.linalg.norm(endpoints[:,0] - endpoints[:,1], axis=1)
ranges = ds

if run == 124:
    veto_mask = max_veto_counts<300
elif run == 212:
    veto_mask = max_veto_counts<150

#use track angle from pca rather than that exported by raw event viewer
angles = []
for axes in track_info_dict['principle_axes']:
    dir = axes[0]
    angles.append(np.arctan2(np.sqrt(dir[0]**2 + dir[1]**2), np.abs(dir[2])))
angles = np.array(angles)

#estimate time since start of decay window to be time since first event in the window
time_since_last_event = timestamps - np.roll(timestamps, 1)
time_since_last_event[0] = .15 #we don't actuallly know what this is for the first event, so just putting a typical value for start of window
start_of_current_winow = 0

pca_widths = []
for event_index in range(len(timestamps)):
    if len(track_info_dict['variance_along_axes'][event_index]) == 3:
        pca_widths.append(track_info_dict['variance_along_axes'][event_index][1]**0.5)
    else:
        pca_widths.append(0)
pca_widths = np.array(pca_widths)
if use_pca_for_width:
    track_widths = pca_widths
else:
    track_widths = np.array(track_info_dict['charge_width'])

print('calculating event times in decay window')
times_since_start_of_window = []
for t, dt in tqdm(zip(timestamps, time_since_last_event)):
    if dt > 0.1:
        start_of_current_winow = t
    times_since_start_of_window.append(t - start_of_current_winow)
times_since_start_of_window = np.array(times_since_start_of_window)


if t_bounds:
    veto_mask = veto_mask & (times_since_start_of_window > t_lower) & (times_since_start_of_window < t_upper)

cut_mask_dict = {}
label_dict = {}
if experiment == 'e21072':
    true_range_dict = {'p1596': 51.6, 'p1596pp': 51.6, 'p770':16.8, 'p770pp':16.8, 
                       'a4434wr':30.6, 'a4434wor':30.6, 'a4434':30.6,'a4434pp':30.6,
                        'a2153':11.8, 'a2153wr':11.8, 'a2153wor':11., 'p1927':0}
    label_dict['p770'] = '770 keV protons'
    label_dict['p770pp'] = '770 keV protons within 20 deg of pad plane'
    label_dict['p1596'] = '~1596 keV protons'
    label_dict['p1596pp'] = '~1596 keV protons within 20 deg of pad plane'
    label_dict['a4434'] = 'all 4434 keV alpha'
    label_dict['a4434wr'] = '4434 keV alpha with recoil'
    label_dict['a4434wor'] = '4434 keV alpha without recoil'
    label_dict['a4434pp'] = '4434 keV alpha within 20 deg of pad plane'
    label_dict['a2153'] = 'all 2153 keV alpha'
    label_dict['a2153wor'] = '2153 keV alpha without recoil'
    label_dict['a2153wr'] = '2153 keV alpha with recoil'
    label_dict['p1927']='>1900 keV protons'
    if run==124:
        cut_mask_dict['p1596'] = (ranges > 31) & (ranges < 65) & (counts > 1.64e5) & (counts < 2.15e5) & veto_mask
        cut_mask_dict['p770'] = (ranges>19) & (ranges<26) & (counts>8.67e4) & (counts<9.5e4) & veto_mask
        cut_mask_dict['a4434'] = (ranges>25) & (ranges<50) & (counts>4.5e5) & (counts < 7e5) & veto_mask
        cut_mask_dict['a2153'] = (ranges>18) & (ranges<28) & (counts>2.25e5) & (counts<3.4e5) & veto_mask
        cut_mask_dict['a4434wr'] = (ranges>25) & (ranges<50) & (counts>5.9e5) & (counts < 7e5) & veto_mask
        cut_mask_dict['a4434wor'] = (ranges>25) & (ranges<50) & (counts>4.5e5) & (counts < 5.7e5) & veto_mask
        cut_mask_dict['a2153wr'] = (ranges>18) & (ranges<28) & (counts>2.83e5) & (counts<3.4e5) & veto_mask
        cut_mask_dict['a2153wor'] = (ranges>18) & (ranges<26) & (counts>2.3e5) & (counts<2.7e5) & veto_mask
        cut_mask_dict['p1927'] = (ranges>70) &(MeV>1.9) & veto_mask
    elif run==212:
        cut_mask_dict['p1596'] = (ranges > 32) & (ranges < 65) & (counts > 3.05e5) & (counts < 3.5e5) & veto_mask
        cut_mask_dict['p770'] = (ranges>20) & (ranges<26) & (counts>1.45e5) & (counts< 1.67e5)&veto_mask
        cut_mask_dict['a4434'] = (ranges>22) & (ranges<50) & (counts>0.6e6) & (counts <1.1e6) & veto_mask
        cut_mask_dict['a2153'] = (ranges>19) & (ranges<26) & (counts>3e5) & (counts<5e5) & veto_mask
        cut_mask_dict['a2153wor'] = (ranges>20.5) & (ranges<23) & (counts>3.15e5) & (counts<3.5e5) & veto_mask
    #cuts within 20 degrees of pad plan
    cut_mask_dict['p1596pp'] = cut_mask_dict['p1596']&(angles>np.radians(70))
    cut_mask_dict['p770pp'] = cut_mask_dict['p770']&(angles>np.radians(70))
    cut_mask_dict['a4434pp'] = cut_mask_dict['a4434']&(angles>np.radians(70))

if offset_endpoints:
    total_track_widths = np.array(track_info_dict['width_above_threshold'])
    endpoints_offset_dir1 = endpoints[:, 0, :] - endpoints[:, 1, :]
    endpoints_offset_dir1 /= np.linalg.norm(endpoints_offset_dir1, axis=1)[:, np.newaxis]
    endpoints[:, 0, :] -= (total_track_widths[:, np.newaxis]/2)*endpoints_offset_dir1
    endpoints_offset_dir2 = endpoints[:, 1, :] - endpoints[:, 0, :]
    endpoints_offset_dir2 /= np.linalg.norm(endpoints_offset_dir2, axis=1)[:, np.newaxis]
    endpoints[:, 1, :] -= (total_track_widths[:, np.newaxis]/2)*endpoints_offset_dir2

#plot showing selected events of each type
rve_plt_mask = (ranges>0)&(ranges<150)&(counts>0)&veto_mask&(MeV<8)
fig, axs = plt.subplots(2,2)
fig.set_figheight(10)
fig.set_figwidth(10)
for ax, ptype in zip(axs.reshape(-1), particles_to_plot):
    ax.set_title(label_dict[ptype])
    mask = cut_mask_dict[ptype]
    hist,xbins,ybins,plot = ax.hist2d(MeV[rve_plt_mask], ranges[rve_plt_mask], 200, norm=matplotlib.colors.LogNorm(), alpha=0.25)
    hist,xbins,ybins,plot = ax.hist2d(MeV[rve_plt_mask&mask], ranges[rve_plt_mask&mask], bins=[xbins, ybins],
                                       norm=matplotlib.colors.LogNorm(), alpha=1, cmin=np.min(hist), cmax=np.max(hist))
    ax.set(xlabel='track width (mm)', ylabel='range (mm)')
    fig.colorbar(plot, ax=ax)

plt.figure()
width_hist_bins = np.linspace(1,5,100)
plt.hist(track_widths[veto_mask], bins=width_hist_bins)
for ptype in particles_to_plot:
    plt.hist(track_widths[cut_mask_dict[ptype]], label=label_dict[ptype], alpha=0.75, bins=width_hist_bins)
plt.legend()
plt.xlabel('track_width (mm)')

fig, axs = plt.subplots(2,2)
fig.set_figheight(10)
fig.set_figwidth(10)
fig.suptitle('uncorrected, all angles')
for ax, ptype in zip(axs.reshape(-1), particles_to_plot):
    ax.set_title(label_dict[ptype])
    mask = cut_mask_dict[ptype]
    plot = ax.scatter(track_widths[mask], ranges[mask], c=times_since_start_of_window[mask], marker='.')
    ax.set(xlabel='track width (mm)', ylabel='range (mm)')
    fig.colorbar(plot, ax=ax)

fig, axs = plt.subplots(2,2)
fig.set_figheight(10)
fig.set_figwidth(10)
fig.suptitle('uncorrected, within 20 deg of pad plane')
theta_mask = angles>np.radians(70)
for ax, ptype in zip(axs.reshape(-1), particles_to_plot):
    ax.set_title(label_dict[ptype])
    mask = cut_mask_dict[ptype]
    plot = ax.scatter(track_widths[mask&theta_mask], ranges[mask&theta_mask], c=times_since_start_of_window[mask&theta_mask], marker='.')
    ax.set(xlabel='track width (mm)', ylabel='range (mm)')
    fig.colorbar(plot, ax=ax)

fig, axs = plt.subplots(2,2)
fig.set_figheight(10)
fig.set_figwidth(10)
fig.suptitle('uncorrected, within 20 deg of beam axis')
theta_mask = angles<np.radians(20)
for ax, ptype in zip(axs.reshape(-1), particles_to_plot):
    ax.set_title(label_dict[ptype])
    mask = cut_mask_dict[ptype]
    plot = ax.scatter(track_widths[mask&theta_mask], ranges[mask&theta_mask], c=times_since_start_of_window[mask&theta_mask], marker='.')
    ax.set(xlabel='track width (mm)', ylabel='range (mm)')
    fig.colorbar(plot, ax=ax)


rscale, wscale, tscale = 20, 3, 0.05
zscale = 200
if exploit_symmetry:
    ijk_array = []
    for i in range(0, N+1):
        for j in range(0, N-np.abs(i) + 1): 
            if phi_dependence:
                for k in range(0, N-np.abs(i) - np.abs(j) + 1):
                    ijk_array.append((i,j,k),)
            else:
                ijk_array.append((i,j))
    ijk_array = np.array(ijk_array)

    def map_r(a_ij, r, t, w, cosphi=0):
        #phi is only has any effect if phi_dependence is True
        r_scaled= r/rscale
        D, v, b = a_ij[-3:] #diffusion constant, ion drift velocity, and charge spreading width

        w = np.copy(w)
        w[w<b] = 0
        z_eff = ((w - b)/D)**2 + v*t
        z_scaled = z_eff/zscale
        #use absolute value of cos_phi since we expect beam distribution to be symetric under reflections across the x-z and y-z planes
        cos_phi = np.abs(cosphi) 
        
        new_r = np.copy(r)
        for ijk, a in zip(ijk_array, a_ij[:-1]):
            if phi_dependence:
                i,j,k = ijk
                new_r += a*(r_scaled**i)*(z_scaled**j)*(cos_phi**k)
            else:
                i, j = ijk
                new_r += a*(r_scaled**i)*(z_scaled**j)
        new_r[new_r < r] = r[new_r < r] #don't allow Efield to move electrons away from the beam axis
        new_r[new_r > 61] = 61#charge can't be deposited outside the field cage
        return new_r


else:
    #try mapping r->r + r'(r, t, w)= r + sum_{i,j,k s.t i+j+k < N} a_ijk r^i t^j w^k
    ijk_array = []
    for n in range(N+1):
        for i in range(n+1):
            for j in range(n - i +1):
                k = n - i - j
                ijk_array.append((i,j,k))
    ijk_array = np.array(ijk_array)
    i_array = ijk_array[:,0]
    j_array = ijk_array[:,1]
    k_array = ijk_array[:,2]

    def map_r(a_ijk, r, t, w, cosphi=0):
        r_scaled = r/rscale
        t_scaled = t/tscale
        w_scaled = w/wscale

        new_r = np.copy(r)
        for ijk, a in zip(ijk_array, a_ijk):
            i,j,k = ijk
            new_r += a*(r_scaled**i)*(t_scaled**j)*(w_scaled**k)

        #new_r[new_r < 0] = 0
        new_r[new_r < r] = r[new_r < r] #don't allow Efield to move electrons away from the beam axis
        new_r[new_r > 61] = 61#charge can't be deposited outside the field cage
        return new_r

def map_endpoints(a_ijk, event_select_mask, beam_xy=(0,0)):
    beam_xy = np.array(beam_xy)
    selected_endpoints = endpoints[event_select_mask]
    p1_init = selected_endpoints[:,0,:2] - beam_xy
    p2_init = selected_endpoints[:,1,:2] - beam_xy
    t = times_since_start_of_window[event_select_mask]
    w = track_widths[event_select_mask]
    r1_init = np.einsum('ij,ij->i', p1_init, p1_init)**0.5
    r2_init = np.einsum('ij,ij->i', p2_init, p2_init)**0.5
    cosphi1 = p1_init[:,0]/r1_init
    cosphi2 = p2_init[:,0]/r2_init
    r1_final = map_r(a_ijk, r1_init, t, w, cosphi1)
    r2_final = map_r(a_ijk, r2_init, t, w, cosphi2)
    to_return =np.copy(selected_endpoints)
    rscale1 = r1_final/r1_init
    rscale1[r1_init==0] = 0
    rscale2 = r2_final/r2_init
    rscale2[r2_init==0] = 0
    to_return[:,0,:2] = np.einsum('ij,i ->ij', p1_init, rscale1) + beam_xy
    to_return[:,1,:2] = np.einsum('ij,i ->ij', p2_init, rscale2) + beam_xy
    return to_return


def map_ranges(a_ijk, event_select_mask, beam_xy=(0,0)):
    #map ranges as range -> range_from_mapped_r - c*width
    new_endpoints = map_endpoints(a_ijk, event_select_mask, beam_xy)
    return np.linalg.norm(new_endpoints[:,0,:] - new_endpoints[:, 1,:], axis=1) #track_widths[event_select_mask]


def to_minimize(a_ijk, beam_xy=(0,0)):
    range_hist_dict = {} #dict to avoid doing the same rmap twice
    to_return = 0
    for weight, ptype in peak_widths_to_minimize:
        range_hist_dict[ptype] = map_ranges(a_ijk, cut_mask_dict[ptype], beam_xy)
        to_return += weight*np.std(range_hist_dict[ptype])**2
    for weight, ptype1, ptype2 in peak_spacings_to_preserve:
        if ptype1 not in range_hist_dict:
            range_hist_dict[ptype1] = map_ranges(a_ijk, cut_mask_dict[ptype1], beam_xy)
        if ptype2 not in range_hist_dict:
            range_hist_dict[ptype2] = map_ranges(a_ijk, cut_mask_dict[ptype2], beam_xy)
        to_return += weight*(np.mean(range_hist_dict[ptype1]) - np.mean(range_hist_dict[ptype2]) - (true_range_dict[ptype1] - true_range_dict[ptype2]))**2
    return to_return

fname_template = '%s_run%d_rmap_order%d.pkl'
#Make string representation of the optimization function to use when saving/loading results.
#<weight>w<particle> will minimize standard deviation of range peaks(eg sp1sp2 will minimize standard deviation of the p1 and p2 peaks)
#<weight>d<particle1>-<particle2> will minimize the difference beteen the spacing between the specified peaks and true difference between these particles ranges
#The "weights" are numbers which will multiply each term in the objective function.
#seperate each terms with underscores (eg sp1596_sp750_dp1596-p750)
#
for weight, ptype in peak_widths_to_minimize:
    fname_template = ('%gw%s_'%(weight, ptype))+fname_template
for weight, ptype1, ptype2 in peak_spacings_to_preserve:
    fname_template = ('%gd%s%s_'%(weight, ptype1, ptype2))+fname_template

if phi_dependence:
    fname_template = 'phidep_'+fname_template
if use_pca_for_width:
    fname_template = 'pca_width_'+fname_template
if exploit_symmetry:
    fname_template = 'sym_'+fname_template
if allow_beam_off_axis:
    fname_template = 'beam_' + fname_template
if t_bounds:
    fname_template = 't%gand%g_'%(t_lower, t_upper)+fname_template
if offset_endpoints:
    fname_template = 'offset_points_'+fname_template
fname_template = opt_method + '_' +fname_template
package_directory = os.path.dirname(os.path.abspath(__file__))
fname = os.path.join(package_directory,fname_template%(experiment, run, N))
print('pickle file name: ', fname)
if os.path.exists(fname):
    print('optimizer previously run, loading saved result')
    with open(fname, 'rb') as file:
        res =  pickle.load(file)
else:
    print('optimizing a_ijk parameters')
    previous_fname = os.path.join(package_directory, fname_template%(experiment, run, N-1))
    #if a solution for N-1 exists, use this as starting guess. Otherwise guess r->r.
    guess = [0 for i in range(len(ijk_array))]
    if exploit_symmetry:
        guess.append(0.09) #D, sqrt(mm)
        guess.append(3000) #v, mm/s
        guess.append(2) #mm  
    if allow_beam_off_axis:
        guess.append(0)
        guess.append(0)

    
    # if os.path.exists(previous_fname):
    #     print('rmap exists for N-1, using as intial guess')
    #     with open(previous_fname, 'rb') as file:
    #         prev_res = pickle.load(file)
    #         prev_ijk_array = []
    #         for n in range(N):
    #             for i in range(n+1):
    #                 for j in range(n - i +1):
    #                     k = n - i - j
    #                     prev_ijk_array.append((i,j,k))
    #         prev_ijk_array = np.array(prev_ijk_array)
    #         for prev_a_ijk, prev_ijk in zip(prev_res.x, prev_ijk_array):
    #             for new_index, ijk in enumerate(ijk_array):
    #                 if np.all(ijk == prev_ijk):
    #                     guess[new_index] = prev_a_ijk
    if allow_beam_off_axis:
        f_to_min = lambda x: to_minimize(x[:-2], x[-2:])
        
    else:
        f_to_min = lambda x: to_minimize(x)

    bounds = [[-50, 50] for i in range(len(ijk_array))]
    if exploit_symmetry:
        bounds.append([1e-10, 1]) #D, sqrt(mm)
        bounds.append([1e-10, 10000])#v, mm/s
        bounds.append([1e-10, 3]) #charge spreading, mm

        #charge spread

    tlast = time.time()
    def callback(x, fig='%s update', save_intermediate_res=True):
        global tlast
        print(x, to_minimize(x))
        print('%f s'%(time.time() - tlast))
        tlast = time.time()
    if allow_beam_off_axis:
        bounds.append([-10, 10])
        bounds.append([-10, 10])
    if opt_method == 'shgo':
        res = opt.shgo(f_to_min, bounds, sampling_method='halton')
    elif opt_method == 'annealing':
        def callback(x, f, context):
            print(x, f, context)
            return False
        res = opt.dual_annealing(f_to_min, bounds, x0=guess, maxiter=10000, callback=callback)
    elif opt_method == 'local':
        res = opt.minimize(f_to_min, guess, callback=callback)
    with open(fname, 'wb') as file:
        pickle.dump(res, file)
if allow_beam_off_axis:
    a_ijk_best = res.x[:-2]
    beam_xy_best = res.x[-2:]
else:
    a_ijk_best = res.x
    beam_xy_best = np.zeros(2)
print(res)

        

plt.figure()
plt.title('run %d uncorrected RvE'%run)
plt.hist2d(MeV[rve_plt_mask], ranges[rve_plt_mask], 200, norm=matplotlib.colors.LogNorm())
plt.xlabel('Energy (MeV)')
plt.ylabel('Range (mm)')
plt.colorbar()

mapped_ranges = map_ranges(a_ijk_best, ranges==ranges, beam_xy_best)
plt.figure()
plt.title('run %d RvE corrected using r-map'%run)
rve_plt_mask = (mapped_ranges>0)&(mapped_ranges<150)&(counts>0)&(MeV<8)  & veto_mask
plt.hist2d(MeV[rve_plt_mask], mapped_ranges[rve_plt_mask], 200, norm=matplotlib.colors.LogNorm())
plt.xlabel('Energy (MeV)')
plt.ylabel('Range (mm)')
plt.colorbar()

fig, axs = plt.subplots(2,2)
fig.set_figheight(10)
fig.set_figwidth(10)
for ax, ptype in zip(axs.reshape(-1), particles_to_plot):
    mask, label, true_range  = cut_mask_dict[ptype], label_dict[ptype], true_range_dict[ptype]
    range_hist_bins = np.linspace(true_range-25, true_range+25, 100)
    ax.set_title(label)
    ax.hist(ranges[mask], bins=range_hist_bins, alpha=0.6, label='uncorrected range; std=%g'%np.std(ranges[mask]))
    ax.hist(mapped_ranges[mask], bins=range_hist_bins, alpha=0.6, label='corrected range; std=%g'%np.std(mapped_ranges[mask]))
    ax.legend()


fig, axs = plt.subplots(2,2)
fig.set_figheight(10)
fig.set_figwidth(10)
r_obs = np.linspace(0, 40, 100)#radius at which charge was observed
for ax, t in zip(axs.reshape(-1), [0,0.025,0.05,0.075]): 
    ax.set_title('r map for tracks at t=%g s'%t)
    for w in np.arange(2, 3.51, 0.25):
        ax.plot(r_obs, map_r(a_ijk_best, r_obs, t, w) - r_obs, label='%f mm'%w)
    ax.set(xlabel='position charge was observed (mm)', ylabel='r_dep - r_obs (mm)')
    ax.legend()

plt.figure()
plt.title('t=0 s')
w_squared = np.linspace(2**2, 3.5**2)
w = w_squared**0.5
for r in [1.,10., 20., 30., 40.]:
    plt.plot(w_squared, map_r(a_ijk_best, np.array([r]*len(w_squared)), np.array([0.]*len(w_squared)), w)-r, label='r=%g mm'%r)
plt.legend()
plt.xlabel('w squared (mm^2)')
plt.ylabel('r_dep - r_obs (mm)')

fig, axs = plt.subplots(2,2)
fig.set_figheight(10)
fig.set_figwidth(10)
r_obs = np.linspace(0, 40, 100)#radius at which charge was observed
for ax, w in zip(axs.reshape(-1), [2, 2.25, 2.5, 2.75]): 
    ax.set_title('r map for track with %f mm width'%w)
    for t in np.linspace(0, 0.1, 10):
        ax.plot(r_obs, map_r(a_ijk_best, r_obs, t, w) - r_obs, label='%f s'%t)
    ax.set(xlabel='position charge was observed (mm)', ylabel='r_dep - r_obs (mm)')
    ax.legend()

fig, axs = plt.subplots(2,2)
fig.set_figheight(10)
fig.set_figwidth(10)
fig.suptitle('corrected range, all angles')
for ax, ptype in zip(axs.reshape(-1), particles_to_plot):
    ax.set_title(label_dict[ptype])
    mask = cut_mask_dict[ptype]
    plot = ax.scatter(track_widths[mask], mapped_ranges[mask], c=times_since_start_of_window[mask], marker='.')
    ax.set(xlabel='track width (mm)', ylabel='range (mm)')
    fig.colorbar(plot, ax=ax)

#plt.show(block=False)

if allow_beam_off_axis:
    print('beam axis at:', beam_xy_best)
if exploit_symmetry:
    print('D, v, charge spreading width = ',a_ijk_best[-3:])
print('a_ijk', a_ijk_best)

track_centers = track_info_dict['track_center']
rad_dist = np.sqrt(track_centers[:,0]**2 + track_centers[:,1]**2)
plt.figure()
mask = cut_mask_dict['p1596'] & (angles < np.radians(20)) & (track_widths < 3.2)
plt.scatter(rad_dist[mask], ranges[mask], c=track_widths[mask], vmin=2., vmax=2.75)#c=times_since_start_of_window[mask])
plt.colorbar()
plt.xlabel('track centroid radial distance from beam axis (mm)')
plt.ylabel('track length (mm)')
plt.show(block=False)
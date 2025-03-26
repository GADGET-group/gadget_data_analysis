import os
import pickle
import sys

import numpy as np
from tqdm import tqdm
import matplotlib.pylab as plt
import matplotlib.colors
import scipy.optimize as opt

from track_fitting.field_distortion import extract_track_axis_info
from track_fitting import build_sim

experiment, run, N = 'e21072', 212, 4
particles_for_fit='proton only'
use_pca_for_width = False
exploit_symmetry = True #Assumes positive ions spread out quickly: f(r,w,t)=f0(r, sqrt(w^2 - kt))
allow_beam_off_axis = True


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
#ranges = np.sqrt(dzs**2 + dxys**2) #why is this different than ds = np.linalg.norm(endpoints[:,0] - endpoints[:,1], axis=1)?
ds = np.linalg.norm(endpoints[:,0] - endpoints[:,1], axis=1)
ranges = ds

veto_mask = max_veto_counts<400

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
times_since_start_of_window = []

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
for t, dt in tqdm(zip(timestamps, time_since_last_event)):
    if dt > 0.1:
        start_of_current_winow = t
    times_since_start_of_window.append(t - start_of_current_winow)
times_since_start_of_window = np.array(times_since_start_of_window)

if run==124:
    mask_1500keV_protons = (ranges > 31) & (ranges < 65) & (counts > 1.64e5) & (counts < 2.15e5) & veto_mask
    mask_750keV_protons = (ranges>20) & (ranges<30) & (counts>8.67e4) & (counts<9.5e4) & veto_mask
    #these cuts include both events w/ and w/o recoil
    #mask_4434keV_alphas = (ranges>25) & (ranges<50) & (counts>4.5e5) & (counts < 7e5) & veto_mask
    #mask_2153keV_alphas = (ranges>18) & (ranges<28) & (counts>2.25e5) & (counts<3.4e5) & veto_mask
    #only include events w/ recoil
    mask_4434keV_alphas = (ranges>25) & (ranges<50) & (counts>5.9e5) & (counts < 7e5) & veto_mask
    mask_2153keV_alphas = (ranges>18) & (ranges<28) & (counts>2.83e5) & (counts<3.4e5) & veto_mask
elif run==212:
    mask_1500keV_protons = (ranges > 32) & (ranges < 65) & (counts > 3e5) & (counts < 3.5e5) & veto_mask
    mask_750keV_protons = (ranges>24) & (ranges<30) & (counts>1.5e5) & veto_mask
    mask_4434keV_alphas = (ranges>25) & (ranges<50) & (counts>6.5e5) & (counts < 8.5e5) & veto_mask
    mask_2153keV_alphas = (ranges>22) & (ranges<28) & (counts>3e5) & (counts<5e5) & veto_mask
true_range_1500keV_proton = 46.7
true_range_750keV_protons = 16.1
true_range_4434keV_alphas = 30.6
true_range_2153keV_alphas = 11.8

pcut1_mask, pcut1_true_range = mask_1500keV_protons, true_range_1500keV_proton
pcut2_mask, pcut2_true_range = mask_750keV_protons, true_range_750keV_protons
acut1_mask, acut1_true_range = mask_4434keV_alphas, true_range_4434keV_alphas
acut2_mask, acut2_true_range = mask_2153keV_alphas, true_range_2153keV_alphas

masks=[pcut1_mask, pcut2_mask, acut1_mask, acut2_mask]
mask_labels=['1500 keV protons', '750 keV protons', '4434 keV alphas', '2153 keV alphas']

plt.figure()
width_hist_bins = np.linspace(1,5,100)
plt.hist(track_widths[veto_mask], bins=width_hist_bins)
for mask, label in zip(masks, mask_labels):
    plt.hist(track_widths[mask], label=label, alpha=0.75, bins=width_hist_bins)
plt.legend()
plt.xlabel('track_width (mm)')

fig, axs = plt.subplots(2,2)
fig.set_figheight(10)
fig.set_figwidth(10)
for ax, mask, label in zip(axs.reshape(-1), masks, mask_labels):
    ax.set_title(label)
    plot = ax.scatter(track_widths[mask], ranges[mask], c=times_since_start_of_window[mask])
    ax.set(xlabel='track width (mm)', ylabel='range (mm)')
    fig.colorbar(plot, ax=ax)


rscale, wscale, tscale = 25, 4, 0.05
if exploit_symmetry:
    ijk_array = []
    for i in range(0, N+1):
        for j in range(0, N-np.abs(i) + 1): 
            ijk_array.append((i,j))
    ijk_array = np.array(ijk_array)

    def map_r(a_ij, r, t, w):
        #a_ij[-1] will contain drift speed constant
        r_scaled, t_scaled, w_scaled = r/rscale, t/tscale, w/wscale
        k = a_ij[-1]
        wsquared = w_scaled**2
        if type(w) == np.ndarray:
            w_eff = np.zeros(wsquared.shape)
            valid_width_mask = wsquared>k*t_scaled #avoid negative numbers in square root
            w_eff[valid_width_mask] = np.sqrt(wsquared[valid_width_mask] - k*t_scaled[valid_width_mask])
        else:
            if wsquared>k*t_scaled:
                w_eff = np.sqrt(wsquared - k*t_scaled)
            else:
                w_eff = 0
        
        new_r = np.copy(r)
        for ij, a in zip(ijk_array, a_ij[:-1]):
            i, j = ij
            new_r += a*(r_scaled**i)*(w_eff**j)
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

    def map_r(a_ijk, r, t, w):
        # if type(t) != np.ndarray:
        #     t = np.array([t]*len(r))
        # if type(w) != np.ndarray:
        #     w = np.array([w]*len(r))
        r_scaled = r/rscale
        t_scaled = t/tscale
        w_scaled = w/wscale
        # r_to_i = np.tile(r_scaled, (len(i_array), 1))**np.tile(np.transpose([i_array]), (1,len(r_scaled))) #[i index, r index]
        # t_to_j = np.tile(t_scaled, (len(j_array), 1))**np.tile(np.transpose([j_array]), (1,len(t_scaled))) #[i index, r index]
        # w_to_k = np.tile(w_scaled, (len(k_array), 1))**np.tile(np.transpose([k_array]), (1,len(w_scaled))) #[i index, r index]

        # new_r = r + np.einsum('i,ij,ij,ij->j', a_ijk, r_to_i, t_to_j, w_to_k)

        new_r = np.copy(r)
        for ijk, a in zip(ijk_array, a_ijk):
            i,j,k = ijk
            new_r += a*(r_scaled**i)*(t_scaled**j)*(w_scaled**k)

        new_r[new_r < 0] = 0
        #new_r[new_r < r] = r[new_r < r] #don't allow Efield to move electrons away from the beam axis
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
    r1_final = map_r(a_ijk, r1_init, t, w)
    r2_final = map_r(a_ijk, r2_init, t, w)
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
    #try to minimize spread  in proton ranges within each peak, while preserving the distance between the two peaks
    pranges1 = map_ranges(a_ijk, pcut1_mask, beam_xy)
    pranges2 = map_ranges(a_ijk, pcut2_mask, beam_xy)
    p1mean, p2mean = np.mean(pranges1), np.mean(pranges2)
    if particles_for_fit != 'proton_only':
        aranges1 = map_ranges(a_ijk, acut1_mask, beam_xy)
        aranges2 = map_ranges(a_ijk, acut2_mask, beam_xy)
        a1mean, a2mean = np.mean(aranges1), np.mean(aranges2)
    #minimize width of each peak
    to_return = np.std(pranges1)**2 + np.std(pranges2)**2
    if particles_for_fit != 'proton only':
        to_return +=np.std(aranges1)**2  +  np.std(aranges2)**2

    
    #preserve distance between proton peaks
    to_return += np.abs(p1mean - p2mean - (pcut1_true_range - pcut2_true_range))**2#/np.abs((pcut1_true_range - pcut2_true_range))**2
    if particles_for_fit != 'proton only':
        #preserve distance between alpha peaks
        to_return += np.abs(a1mean - a2mean - (acut1_true_range - acut2_true_range))**2#/np.abs((acut1_true_range - acut2_true_range))**2
        #preserve distance between proton and alpha bands
        to_return += np.abs(p1mean - a1mean - (pcut1_true_range - acut1_true_range))**2
        #and try to keep everything at roughly the correct true range
        to_return += (p1mean - pcut1_true_range)**2
    print(to_return)
    return to_return


print('calculating order %d'%N)
if use_pca_for_width:
    fname_template = '%s_run%d_rmap_order%d_pca_width.pkl'
else:
    fname_template = '%s_run%d_rmap_order%d.pkl'
if particles_for_fit == 'proton only':
    fname_template = 'proton_only_'+fname_template
if exploit_symmetry:
    fname_template = 'sym_'+fname_template
if allow_beam_off_axis:
    fname_template = 'offcenter_' + fname_template
package_directory = os.path.dirname(os.path.abspath(__file__))
fname = os.path.join(package_directory,fname_template%(experiment, run, N))
if os.path.exists(fname):
    print('optimizer previously run, loading saved result')
    with open(fname, 'rb') as file:
        res =  pickle.load(file)
else:
    print('optimizing a_ijk parameters')
    previous_fname = os.path.join(package_directory, fname_template%(experiment, run, N-1))
    #if a solution for N-1 exists, use this as starting guess. Otherwise guess r->r.
    guess_length = len(ijk_array)
    if exploit_symmetry:
        guess_length += 1
    if allow_beam_off_axis:
        guess_length += 2
    guess = np.zeros(guess_length)
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
        res = opt.minimize(lambda x: to_minimize(x[:-2], x[-2:]), guess)
    else:
        res = opt.minimize(lambda x: to_minimize(x), guess)
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
plt_mask = (ranges>0)&(ranges<150)&(counts>0)&veto_mask
plt.hist2d(counts[plt_mask], ranges[plt_mask], 200, norm=matplotlib.colors.LogNorm())
plt.colorbar()

mapped_ranges = map_ranges(a_ijk_best, ranges==ranges, beam_xy_best)
plt.figure()
plt.title('run %d RvE corrected using r-map'%run)
plt_mask = (mapped_ranges>0)&(mapped_ranges<150)&(counts>0)  & veto_mask
plt.hist2d(counts[plt_mask], mapped_ranges[plt_mask], 200, norm=matplotlib.colors.LogNorm())
#plt.colorbar()

fig, axs = plt.subplots(2,2)
fig.set_figheight(10)
fig.set_figwidth(10)
for ax, mask, label, true_range in zip(axs.reshape(-1), masks, mask_labels, [pcut1_true_range, pcut2_true_range, acut1_true_range, acut2_true_range]):
    range_hist_bins = np.linspace(true_range-25, true_range+25, 100)
    ax.set_title(label)
    ax.hist(ranges[mask], bins=range_hist_bins, alpha=0.6, label='uncorrected range; std=%f'%np.std(ranges[mask]))
    ax.hist(mapped_ranges[mask], bins=range_hist_bins, alpha=0.6, label='corrected range; std=%f'%np.std(mapped_ranges[mask]))
    ax.legend()


fig, axs = plt.subplots(2,2)
fig.set_figheight(10)
fig.set_figwidth(10)
r_obs = np.linspace(0, 40, 100)#radius at which charge was observed
for ax, t in zip(axs.reshape(-1), [0,0.025,0.05,0.075]): 
    ax.set_title('r map for tracks at t=%f s'%t)
    for w in np.arange(2, 3.51, 0.25):
        ax.plot(r_obs, map_r(a_ijk_best, r_obs, t, w) - r_obs, label='%f mm'%w)
    ax.set(xlabel='position charge was observed (mm)', ylabel='r_dep - r_obs (mm)')
    ax.legend()

plt.figure()
plt.title('t=0 s')
w_squared = np.linspace(2**2, 3.5**2)
w = w_squared**0.5
for r in [1.,10., 20., 30., 40.]:
    plt.plot(w_squared, map_r(a_ijk_best, np.array([r]*len(w_squared)), np.array([0.]*len(w_squared)), w)-r, label='r=%f mm'%r)
plt.legend()
plt.xlabel('w squared (mm^2)')
plt.ylabel('r_dep - r_obs (mm)')

fig, axs = plt.subplots(2,2)
fig.set_figheight(10)
fig.set_figwidth(10)
r_obs = np.linspace(0, 40, 100)#radius at which charge was observed
for ax, w in zip(axs.reshape(-1), [2, 2.5, 3, 3.5]): 
    ax.set_title('r map for track with %f mm width'%w)
    for t in np.linspace(0, 0.1, 10):
        ax.plot(r_obs, map_r(a_ijk_best, r_obs, t, w) - r_obs, label='%f s'%t)
    ax.set(xlabel='position charge was observed (mm)', ylabel='r_dep - r_obs (mm)')
    ax.legend()

plt.show(block=False)

if allow_beam_off_axis:
    print('beam axis at:', beam_xy_best)
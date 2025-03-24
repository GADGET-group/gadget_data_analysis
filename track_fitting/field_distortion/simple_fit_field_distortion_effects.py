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

experiment, run, N = 'e21072', 124, 3


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

# track_widths = []
# for event_index in range(len(timestamps)):
#     if len(track_info_dict['variance_along_axes'][event_index]) == 3:
#         track_widths.append(track_info_dict['variance_along_axes'][event_index][1]**0.5)
#     else:
#         track_widths.append(0)
# track_widths = np.array(track_widths)
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
    mask_4434keV_alphas = (ranges>25) & (ranges<50) & (counts>4.5e5) & (counts < 7e5) & veto_mask
    mask_2153keV_alphas = (ranges>18) & (ranges<28) & (counts>2.25e5) & (counts<3.4e5) & veto_mask
elif run==212:
    mask_1500keV_protons = (ranges > 40) & (ranges < 65) & (counts > 3e5) & (counts < 3.5e5) & veto_mask
    mask_750keV_protons = (ranges>24) & (ranges<30) & (counts>1.5e5) & veto_mask
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
#try mapping r->r + r'(r, t, w)= r + sum_{i,j,k s.t i+j+k < N} a_ijk r^i t^j w^k
ijk_array = []
for n in range(N+1):
    for i in range(n+1):
        for j in range(n - i +1):
            k = n - i - j
            ijk_array.append((i,j,k))
ijk_array = np.array(ijk_array)

def map_r(a_ijk, r, t, w):
    new_r = np.copy(r)
    for ijk, a in zip(ijk_array, a_ijk):
        i,j,k = ijk
        new_r += a*((r/rscale)**i)*((t/tscale)**j)*((w/wscale)**k)
    if type(new_r) == np.ndarray:
        #new_r[new_r < 0] = 0
        new_r[new_r < r] = r[new_r < r] #don't allow Efield to point away from beam axis
        new_r[new_r > 61] = 61#charge can't be deposited outside the field cage
    elif new_r < r:
        new_r = r
    elif new_r > 61:
        new_r = 61
    return new_r

def map_endpoints(a_ijk, event_select_mask):
    selected_endpoints = endpoints[event_select_mask]
    p1_init = selected_endpoints[:,0,:]
    p2_init = selected_endpoints[:,1,:]
    t = times_since_start_of_window[event_select_mask]
    w = track_widths[event_select_mask]
    r1_init = np.einsum('ij,ij->i', p1_init[:,:2], p1_init[:,:2])**0.5
    r2_init = np.einsum('ij,ij->i', p2_init[:,:2], p2_init[:,:2])**0.5
    r1_final = map_r(a_ijk, r1_init, t, w)
    r2_final = map_r(a_ijk, r2_init, t, w)
    to_return =np.zeros(selected_endpoints.shape)
    to_return[:,0,:] = np.einsum('ij,i ->ij', selected_endpoints[:,0,:], r1_final/r1_init)
    to_return[:,1,:] = np.einsum('ij,i ->ij', selected_endpoints[:,1,:], r2_final/r2_init)
    return to_return


def map_ranges(a_ijk, event_select_mask):
    new_endpoints = map_endpoints(a_ijk, event_select_mask)
    return np.linalg.norm(new_endpoints[:,0,:] - new_endpoints[:, 1,:], axis=1)


def to_minimize(a_ijk):
    #try to minimize spread  in proton ranges within each peak, while preserving the distance between the two peaks
    pranges1 = map_ranges(a_ijk, pcut1_mask)
    pranges2 = map_ranges(a_ijk, pcut2_mask)
    to_return = np.std(pranges1) + ((np.mean(pranges1) - np.mean(pranges2)) - (pcut1_true_range - pcut2_true_range))**2  + np.std(pranges2)
    aranges1 = map_ranges(a_ijk, acut1_mask)
    aranges2 = map_ranges(a_ijk, acut2_mask)
    to_return += np.std(aranges1) + ((np.mean(aranges1) - np.mean(aranges2)) - (acut1_true_range - acut2_true_range))**2  + np.std(aranges2)
    return to_return

package_directory = os.path.dirname(os.path.abspath(__file__))
fname = os.path.join(package_directory, '%s_run%d_rmap_order%d.pkl'%(experiment, run, N))
if os.path.exists(fname):
    print('optimizer previously run, loading saved result')
    with open(fname, 'rb') as file:
        res =  pickle.load(file)
else:
    print('optimizing a_ijk parameters')
    previous_fname = os.path.join(package_directory, '%s_run%d_rmap_order%d.pkl'%(experiment, run, N-1))
    #if a solution for N-1 exists, use this as starting guess. Otherwise guess r->r.
    a_ijk_guess = np.zeros(len(ijk_array))
    if os.path.exists(previous_fname):
        print('rmap exists for N-1, using as intial guess')
        with open(previous_fname, 'rb') as file:
            prev_res = pickle.load(file)
            prev_ijk_array = []
            for n in range(N):
                for i in range(n+1):
                    for j in range(n - i +1):
                        k = n - i - j
                        prev_ijk_array.append((i,j,k))
            prev_ijk_array = np.array(prev_ijk_array)
            for prev_a_ijk, prev_ijk in zip(prev_res.x, prev_ijk_array):
                for new_index, ijk in enumerate(ijk_array):
                    if np.all(ijk == prev_ijk):
                        a_ijk_guess[new_index] = prev_a_ijk
        
    res = opt.minimize(to_minimize, a_ijk_guess)
    with open(fname, 'wb') as file:
        pickle.dump(res, file)
print(res)
a_ijk_best = res.x
        

plt.figure()
plt.title('run %d uncorrected RvE'%run)
plt_mask = (ranges>0)&(ranges<150)&(counts>0)
plt.hist2d(counts[plt_mask], ranges[plt_mask], 200, norm=matplotlib.colors.LogNorm())
plt.colorbar()

mapped_ranges = map_ranges(a_ijk_best, ranges==ranges)
plt.figure()
plt.title('run %d RvE corrected using r-map'%run)
plt_mask = (mapped_ranges>0)&(mapped_ranges<150)&(counts>0)  & veto_mask
plt.hist2d(counts[plt_mask], mapped_ranges[plt_mask], 200, norm=matplotlib.colors.LogNorm())
#plt.colorbar()

fig, axs = plt.subplots(2,2)
fig.set_figheight(10)
fig.set_figwidth(10)
for ax, mask, label, true_range in zip(axs.reshape(-1), masks, mask_labels, [pcut1_true_range, pcut2_true_range, acut1_true_range, acut2_true_range]):
    range_hist_bins = np.linspace(true_range-20, true_range+20, 80)
    ax.set_title(label)
    ax.hist(ranges[mask], bins=range_hist_bins, alpha=0.6, label='uncorrected range')
    ax.hist(mapped_ranges[mask], bins=range_hist_bins, alpha=0.6, label='corrected range')
    ax.legend()

fig, axs = plt.subplots(2,2)
fig.set_figheight(10)
fig.set_figwidth(10)
r_obs = np.linspace(0, 50, 100)#radius at which charge was observed
for ax, w in zip(axs.reshape(-1), [2, 2.5, 3, 3.5]): 
    ax.set_title('r map for track with %f mm width'%w)
    for t in [0, 0.02, 0.05, 0.1]:
        ax.plot(r_obs, map_r(a_ijk_best, r_obs, t, w) - r_obs, label='%f s'%t)
    ax.set(xlabel='position charge was observed (mm)', ylabel='r_dep - r_obs (mm)')
    ax.legend()

plt.show(block=False)

print('standard deviation of 1500 keV proton ranges: %f mm'%np.std(mapped_ranges[mask_1500keV_protons]))
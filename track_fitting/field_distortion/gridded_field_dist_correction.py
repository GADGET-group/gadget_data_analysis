'''
Find rbar_dep(x_det, y_det, width, time)->(x_dep, y_dep)
and dz_dep(dz_det, x_dep, y_dep, width, time)
'''

import os
import pickle
import sys
import multiprocessing as mp

import numpy as np
from tqdm import tqdm
import matplotlib.pylab as plt
import matplotlib.colors
import scipy.optimize as opt
import scipy.interpolate as interp

from track_fitting.field_distortion import extract_track_axis_info
from track_fitting import build_sim

load_intermediate_result = False # if True, then load saved pickle file of best result found so far, and display data with no further optimization

'''
Configuration for fit.
'''
experiment, run = 'e21072', 212
#list of (wieght, peak label) tuples. Objective function will include minimizing sum_i weight_i * std(peak i range)^2
peak_widths_to_minimize = [(1, 'p1596'),  (1, 'a4434'), (1, 'p770'), (1, 'a2153')]
#list of (weight, peak 1, peak 2) tuples.
#Objective function will minimize sum_i weight_i ((mean(peak i1 range) - mean(peaki2 range) - (true peak i2 range - true peak i2 range))^2
peak_spacings_to_preserve = [(1, 'a2153', 'a4434'), (1, 'p770', 'p1596'), (1, 'p1596', 'a2153')]
use_pca_for_width = False #if false, uses standard deviation of charge along the 2nd pca axis
#include up to 4 particles to make scatter plots and histograms for
particles_to_plot = ['p1596', 'p770', 'a2153', 'a4434']
t_bounds = False 
t_lower, t_upper = 0, 0.1
offset_endpoints = True

if True:
    xgrid_len = ygrid_len = 5
    zgrid_len = 2
    wgrid_len = 5
    tgrid_len = 7
    x_grid_guess = np.linspace(-40, 40, xgrid_len)
    y_grid_guess = np.linspace(-40, 40, ygrid_len)
    z_grid_guess = np.linspace(-40, 40, zgrid_len)
    w_grid_guess = np.linspace(2, 3.5, wgrid_len)
    t_grid_guess = np.linspace(0, 0.09, tgrid_len)

'''
Load data and do pre-processing
'''
track_info_dict = extract_track_axis_info.get_track_info(experiment, run)
endpoints = np.array(track_info_dict['endpoints'])

processed_directory = '/egr/research-tpc/shared/Run_Data/run_%04d_raw_viewer/run_%04dsmart'%(run, run)

#load histogram arrays
counts = np.load(os.path.join(processed_directory, 'counts.npy'))
max_veto_counts = np.load(os.path.join(processed_directory, 'veto.npy'))
timestamps = np.load(os.path.join(processed_directory, 'timestamps.npy'))

h5file = build_sim.get_rawh5_object(experiment, run)
MeV = build_sim.get_integrated_charge_energy_offset(experiment, run) + counts/build_sim.get_adc_counts_per_MeV(experiment, run)
#ranges = np.sqrt(dzs**2 + dxys**2) #why is this different than ds = np.linalg.norm(endpoints[:,0] - endpoints[:,1], axis=1)?
ranges = np.linalg.norm(endpoints[:,0] - endpoints[:,1], axis=1)

if run == 124:
    veto_mask = max_veto_counts<200
elif run == 212:
    veto_mask = max_veto_counts<150

#use track angle from pca rather than that exported by raw event viewer
angles = []
track_vects = []
for axes in track_info_dict['principle_axes']:
    dir = axes[0]
    angles.append(np.arctan2(np.sqrt(dir[0]**2 + dir[1]**2), np.abs(dir[2])))
angles = np.array(angles)
track_vects = np.array(track_vects)

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

if t_bounds:
    veto_mask = veto_mask & (times_since_start_of_window > t_lower) & (times_since_start_of_window < t_upper)

cut_mask_dict = {}
label_dict = {}
if experiment == 'e21072':
    true_range_dict = {'p1596': 51.6, 'p1596pp': 51.6, 'p770':16.8, 'p770pp':16.8, 
                       'a4434wr':30.6, 'a4434wor':30.6, 'a4434':30.6,'a4434pp':30.6,
                        'a2153':11.8, 'a2153wr':11.8, 'a2153wor':11.8}
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
    if run==124:
        cut_mask_dict['p1596'] = (ranges > 31) & (ranges < 60) & (counts > 1.69e5) & (counts < 2.08e5) & veto_mask
        cut_mask_dict['p770'] = (ranges>16.8) & (ranges<23) & (counts>8.2e4) & (counts<9.5e4) & veto_mask
        cut_mask_dict['a4434'] = (ranges>21) & (ranges<47) & (counts>4.5e5) & (counts < 7e5) & veto_mask
        cut_mask_dict['a2153'] = (ranges>16) & (ranges<26) & (counts>2.25e5) & (counts<3.4e5) & veto_mask
        #TODO: need to update these
        # cut_mask_dict['a4434wr'] = (ranges>25) & (ranges<50) & (counts>5.9e5) & (counts < 7e5) & veto_mask
        # cut_mask_dict['a4434wor'] = (ranges>25) & (ranges<50) & (counts>4.5e5) & (counts < 5.7e5) & veto_mask
        # cut_mask_dict['a2153wr'] = (ranges>18) & (ranges<28) & (counts>2.83e5) & (counts<3.4e5) & veto_mask
        # cut_mask_dict['a2153wor'] = (ranges>18) & (ranges<26) & (counts>2.3e5) & (counts<2.7e5) & veto_mask
    elif run==212:
        cut_mask_dict['p1596'] = (ranges > 30) & (ranges < 61) & (counts > 3.05e5) & (counts < 3.5e5) & veto_mask
        cut_mask_dict['p770'] = (ranges>16.5) & (ranges<26) & (counts>1.45e5) & (counts< 1.67e5)&veto_mask
        cut_mask_dict['a4434'] = (ranges>22) & (ranges<50) & (counts>0.6e6) & (counts <1.1e6) & veto_mask
        cut_mask_dict['a2153'] = (ranges>17) & (ranges<26) & (counts>2.8e5) & (counts<5.1e5) & veto_mask
        #TODO: update this one
        #cut_mask_dict['a2153wor'] = (ranges>20.5) & (ranges<23) & (counts>3.15e5) & (counts<3.5e5) & veto_mask
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



#translate endpoint pairs in z so z=0 is the average of the z values
z_ave_init = (endpoints[:,0,2] + endpoints[:,1,2])/2
endpoints[:,0,2] -= z_ave_init
endpoints[:,1,2] -= z_ave_init


def map_endpoints(endpoints_to_map, w, t, xparams, yparams, zparams,  x_grid, y_grid, z_grid, w_grid, t_grid):

    to_return = np.copy(endpoints_to_map)
    #map first set of endpoints
    x, y, z = endpoints_to_map[:,0,0], endpoints_to_map[:,0,1], endpoints_to_map[:,0,2]
    to_return[:,0,0] = interp.interpn(points=(x_grid, y_grid, w_grid, t_grid), values=xparams, xi=(x,y, w, t), bounds_error=False, fill_value=None)
    to_return[:,0,1] = interp.interpn(points=(x_grid, y_grid, w_grid, t_grid), values=yparams, xi=(x,y, w, t), bounds_error=False, fill_value=None)
    to_return[:,0,2] = interp.interpn(points=(x_grid, y_grid, z_grid, w_grid, t_grid), values=zparams, xi=(x,y, z, w, t), bounds_error=False, fill_value=None)
    #map second set
    x, y, z = endpoints_to_map[:,1,0], endpoints_to_map[:,1,1], endpoints_to_map[:,1,2] 
    to_return[:,1,0] = interp.interpn(points=(x_grid, y_grid, w_grid, t_grid), values=xparams, xi=(x,y, w, t), bounds_error=False, fill_value=None)
    to_return[:,1,1] = interp.interpn(points=(x_grid, y_grid, w_grid, t_grid), values=yparams, xi=(x,y, w, t), bounds_error=False, fill_value=None)
    to_return[:,1,2] = interp.interpn(points=(x_grid, y_grid,  z_grid, w_grid, t_grid), values=zparams, xi=(x,y, z, w, t), bounds_error=False, fill_value=None)
    return to_return


def map_ranges(xparams, yparams, zparams, x_grid, y_grid, z_grid, w_grid, t_grid, event_select_mask):
    #map ranges as range -> range_from_mapped_r - c*width
    new_endpoints = map_endpoints(endpoints[event_select_mask], track_widths[event_select_mask],
                                   times_since_start_of_window[event_select_mask],
                                   xparams, yparams, zparams, x_grid, y_grid, z_grid, w_grid, t_grid)
    return np.linalg.norm(new_endpoints[:,0,:] - new_endpoints[:, 1,:], axis=1) #track_widths[event_select_mask]

#minimize events that need to be passed to other threads
type_endpoints = {}
type_widths = {}
type_times = {}
for weight, ptype in peak_widths_to_minimize:
    type_endpoints[ptype] = endpoints[cut_mask_dict[ptype]]
    type_widths[ptype] = track_widths[cut_mask_dict[ptype]]
    type_times[ptype] = times_since_start_of_window[cut_mask_dict[ptype]]
for weight, ptype1, ptype2 in peak_spacings_to_preserve:
    type_endpoints[ptype1] = endpoints[cut_mask_dict[ptype1]]
    type_widths[ptype1] = track_widths[cut_mask_dict[ptype1]]
    type_times[ptype1] = times_since_start_of_window[cut_mask_dict[ptype1]]
    type_endpoints[ptype2] = endpoints[cut_mask_dict[ptype2]]
    type_widths[ptype2] = track_widths[cut_mask_dict[ptype2]]
    type_times[ptype2] = times_since_start_of_window[cut_mask_dict[ptype2]]


def map_type_range(xparams, yparams, zparams,  x_grid, y_grid, z_grid, w_grid, t_grid, ptype):
    new_endpoints = map_endpoints(type_endpoints[ptype], type_widths[ptype],
                                   type_times[ptype],
                                   xparams, yparams, zparams, x_grid, y_grid, z_grid, w_grid, t_grid)
    return np.linalg.norm(new_endpoints[:,0,:] - new_endpoints[:, 1,:], axis=1)


def convert_fit_params(params):
    '''
    Takes 1D array of parameters used for fitting, and maps to xparams, yparams, zparams used by range mapping.
    xparams[0,0,0,0]=yparams[0,0,0,0]=grid value 0
    '''
    start, end = 0, xgrid_len
    x_grid = params[start:end]
    start, end= end, end+ygrid_len
    y_grid = params[start:end]
    start, end = end, end+zgrid_len
    z_grid = params[start:end]
    start, end = end, end+wgrid_len
    w_grid = params[start:end]
    start, end = end, end+tgrid_len
    t_grid = params[start:end]

    xparam_length = xgrid_len*ygrid_len*wgrid_len*tgrid_len
    start, end = end, end + xparam_length-1
    xparams_flat = np.zeros(xparam_length)
    xparams_flat[0] = x_grid[0]
    xparams_flat[1:] = params[start:end]
    xparams = np.reshape(xparams_flat, (xgrid_len, ygrid_len, wgrid_len, tgrid_len))
    
    yparam_length = xgrid_len*ygrid_len*wgrid_len*tgrid_len
    start, end = end, end + yparam_length-1
    yparams_flat = np.zeros(yparam_length)
    yparams_flat[0] = y_grid[0]
    yparams_flat[1:] = params[start:end]
    yparams = np.reshape(yparams_flat,(xgrid_len, ygrid_len, wgrid_len, tgrid_len))

    z_param_length = xgrid_len*ygrid_len*zgrid_len*wgrid_len*tgrid_len
    start, end = end, end+z_param_length-1
    zparams_flat = np.zeros(z_param_length)
    zparams_flat[0] = z_grid[0]
    zparams_flat[1:] = params[start:end]
    zparams = np.reshape(zparams_flat, (xgrid_len, ygrid_len, zgrid_len, wgrid_len, tgrid_len))
    return xparams, yparams, zparams, x_grid, y_grid, z_grid, w_grid, t_grid

xguess = np.zeros((xgrid_len, ygrid_len, wgrid_len, tgrid_len))
yguess = np.zeros((xgrid_len, ygrid_len, wgrid_len, tgrid_len))
zguess = np.zeros((xgrid_len, ygrid_len, zgrid_len, wgrid_len, tgrid_len))
for i in range(ygrid_len):
    for j in range(wgrid_len):
        for k in range(tgrid_len):
            xguess[:,i,j,k] = x_grid_guess
for i in range(xgrid_len):
    for j in range(wgrid_len):
        for k in range(tgrid_len):
            yguess[i,:,j,k] = y_grid_guess
for i in range(xgrid_len):
    for j in range(ygrid_len):
        for k in range(wgrid_len):
            for l in range(tgrid_len):
                zguess[i,j, :,k, l] = z_grid_guess
guess = np.concatenate([x_grid_guess, y_grid_guess, z_grid_guess, w_grid_guess, t_grid_guess, xguess.flatten()[1:], yguess.flatten()[1:], zguess.flatten()[1:]])

def to_minimize(params):
    xparams, yparams, zparams, x_grid, y_grid, z_grid, w_grid, t_grid = convert_fit_params(params)
    #confirm all grid spacings are strictly accending
    if np.any(np.diff(x_grid) <= 0) or np.any(np.diff(y_grid) <= 0) or np.any(np.diff(z_grid) <= 0)  \
            or np.any(np.diff(t_grid) <= 0) or np.any(np.diff(w_grid) <= 0):
        return np.inf
    range_hist_dict = {} #dict to avoid doing the same rmap twice
    to_return = 0
    for weight, ptype in peak_widths_to_minimize:
        range_hist_dict[ptype] = map_type_range(xparams, yparams, zparams, x_grid, y_grid, z_grid, w_grid, t_grid, ptype) 
        to_return += weight*np.std(range_hist_dict[ptype])**2
    for weight, ptype1, ptype2 in peak_spacings_to_preserve:
        if ptype1 not in range_hist_dict:
            range_hist_dict[ptype1] = map_type_range(xparams, yparams, zparams, x_grid, y_grid, z_grid, w_grid, t_grid, ptype1) 
        if ptype2 not in range_hist_dict:
            range_hist_dict[ptype2] = map_type_range(xparams, yparams, zparams, x_grid, y_grid, z_grid, w_grid, t_grid, ptype2) 
        to_return += weight*(np.mean(range_hist_dict[ptype1]) - np.mean(range_hist_dict[ptype2]) - (true_range_dict[ptype1] - true_range_dict[ptype2]))**2
    #print(to_return)
    return to_return

fname_template = 'gridcor_%s_run%d_x%d_y%d_z%d_w%d_t%d.pkl'
if t_bounds:
    fname_template = 't%gand%g_'%(t_lower, t_upper)+fname_template
if use_pca_for_width:
    fname_template = 'pca_width_'+fname_template
if offset_endpoints:
    fname_template = 'offset_points_'+fname_template

for weight, ptype in peak_widths_to_minimize:
    fname_template = ('%gw%s_'%(weight, ptype))+fname_template
for weight, ptype1, ptype2 in peak_spacings_to_preserve:
    fname_template = ('%gd%s%s_'%(weight, ptype1, ptype2))+fname_template

package_directory = os.path.dirname(os.path.abspath(__file__))
fname = os.path.join(package_directory,fname_template%(experiment, run, xgrid_len, ygrid_len, zgrid_len, wgrid_len, tgrid_len))
inter_fname = os.path.join(package_directory,'inter_' + fname_template%(experiment, run, xgrid_len, ygrid_len, zgrid_len, wgrid_len, tgrid_len))
print('pickle file name: ', fname)
if load_intermediate_result:
    with open(inter_fname, 'rb') as file:
        x =  pickle.load(file)
    print(x, to_minimize(x))
    xparams, yparams, zparams, x_grid, y_grid, z_grid, w_grid, t_grid = convert_fit_params(x)    
elif os.path.exists(fname):
    print('optimizer previously run, loading saved result')
    with open(fname, 'rb') as file:
        res =  pickle.load(file)
    print(res)
    xparams, yparams, zparam, x_grid, y_grid, z_grid, w_grid, t_grids = convert_fit_params(res.x)
else:
    print('performing optimization')   

    def callback(x, fig='%s update', save_intermediate_res=True):
        print(x,to_minimize(x))
        if save_intermediate_res:
            with open(inter_fname, 'wb') as f:
                pickle.dump(x, f)

    callback(guess, '%s init')
    print('number of parameters to fit:', len(guess))
    res = opt.minimize(to_minimize, guess, callback=callback, method='BFGS')
    with open(fname, 'wb') as file:
        pickle.dump(res, file)
    print(res)
    xparams, yparams, zparams = convert_fit_params(res.x)

'''
make plots
'''
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

#histogram of track widths, for different particle types
plt.figure()
width_hist_bins = np.linspace(1,5,100)
plt.hist(track_widths[veto_mask], bins=width_hist_bins)
for ptype in particles_to_plot:
    plt.hist(track_widths[cut_mask_dict[ptype]], label=label_dict[ptype], alpha=0.75, bins=width_hist_bins)
plt.legend()
plt.xlabel('track_width (mm)')

#scatter plot of range vs track width color coded by time
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

plt.figure()
plt.title('run %d uncorrected RvE'%run)
plt.hist2d(MeV[rve_plt_mask], ranges[rve_plt_mask], 200, norm=matplotlib.colors.LogNorm())
plt.xlabel('Energy (MeV)')
plt.ylabel('Range (mm)')
plt.colorbar()

init_ranges = map_ranges(xguess, yguess, zguess, x_grid_guess, y_grid_guess, z_grid_guess, w_grid_guess,t_grid_guess, ranges==ranges)
plt.figure()
plt.title('run %d init guess RvE'%run)
rve_plt_mask = (init_ranges>0)&(init_ranges<150)&(counts>0)&(MeV<8)  & veto_mask
plt.hist2d(MeV[rve_plt_mask], init_ranges[rve_plt_mask], 200, norm=matplotlib.colors.LogNorm())
plt.xlabel('Energy (MeV)')
plt.ylabel('Range (mm)')
plt.colorbar()

mapped_ranges = map_ranges(xparams, yparams, zparams, x_grid, y_grid, z_grid, w_grid, t_grid, ranges==ranges)
plt.figure()
plt.title('run %d corrected RvE'%run)
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
    range_hist_bins = np.linspace(max(true_range-35, 0), true_range+25, 100)
    ax.set_title(label)
    ax.hist(ranges[mask], bins=range_hist_bins, alpha=0.6, label='uncorrected range; std=%g'%np.std(ranges[mask]))
    ax.hist(init_ranges[mask], bins=range_hist_bins, alpha=0.6, label='guess range; std=%g'%np.std(init_ranges[mask]))
    ax.hist(mapped_ranges[mask], bins=range_hist_bins, alpha=0.6, label='corrected range; std=%g'%np.std(mapped_ranges[mask]))
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


plt.figure()
plt.title('1550-1650 keV event ranges')
mask = veto_mask & (MeV>1.55) & (MeV<1.65)
range_hist_bins = np.linspace(0, 70, 100)
plt.hist(ranges[mask], bins=range_hist_bins, alpha=1, label='uncorrected range; std=%g'%np.std(ranges[mask]))
plt.hist(init_ranges[mask], bins=range_hist_bins, alpha=0.6, label='guess range; std=%g'%np.std(init_ranges[mask]))
plt.hist(mapped_ranges[mask], bins=range_hist_bins, alpha=0.6, label='corrected range; std=%g'%np.std(mapped_ranges[mask]))
plt.xlabel('range (mm)')
plt.legend()

#make interactive figure for viewing results
#fig, axs = plt.subplots(1,2)


plt.show(block=False)

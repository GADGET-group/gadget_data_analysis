USE_GPU = True

import tqdm
import os
import time
import pickle

import numpy as np
if USE_GPU:
    import cupy as cp
else:
    cp = np
    cp.asnumpy = lambda x: x
    class Device:
         def __init__(self, x):
              pass
         def __enter__(self):
              pass
         def __exit__(self, exc_type, exc_value, traceback):
              pass
    cp.cuda = Device(0)
    cp.cuda.Device = Device
import scipy.optimize as optimize

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.path import Path
import matplotlib.colors

from raw_viewer.pad_gain_match import process_runs

gpu_device = 1
load_first_result = True
load_second_result = False


runs = (121, 122, 123, 124, 125, 126, 127, 128)
veto_thresh = 10000
exp = 'e21072'
rve_bins = 1000
offset = 'constant'

lengths = process_runs.get_lengths(exp, runs)
cpp = process_runs.get_quantity('pad_charge', exp, runs)
veto_counts = process_runs.get_veto_counts(exp, runs)
veto_mask = veto_counts < veto_thresh
with cp.cuda.Device(gpu_device):
    cpp_gpu = cp.array(cpp)





def get_gm_ic(gains, counts_per_pad=cpp_gpu, return_gpu=False):
    #counts per pad needs to already be on the gpu
    with cp.cuda.Device(gpu_device):
        gains_gpu = cp.array(gains)
        to_return = cp.einsum('ij, j', counts_per_pad, gains_gpu)
    if return_gpu:
        return to_return
    else:
        return cp.asnumpy(to_return)

no_gm_ic = get_gm_ic(np.ones(1024))

#set up initial gain match cuts
cuts1 = []
true_energies = [1.633, 0.7856]# only includes energy deposited as ionization
cuts1.append((no_gm_ic>1.9e5) & (no_gm_ic<2.1e5) & (lengths>30) & (lengths<55) & veto_mask)
cuts1.append((no_gm_ic>8.8e4) & (no_gm_ic<1.025e5) & (lengths>16.5) & (lengths<20.2) & veto_mask)



plt.figure()
plt.hist(veto_counts, 100)

fig = plt.figure()
plt_mask = veto_mask&(lengths<100)
plt.title('without gain match, runs: '+str(runs))
plt.hist2d(no_gm_ic[plt_mask], lengths[plt_mask], bins=rve_bins, norm=matplotlib.colors.LogNorm())

fig = plt.figure()
plt.title('gain match cuts')
plt.hist2d(no_gm_ic[plt_mask], lengths[plt_mask], bins=rve_bins, norm=matplotlib.colors.LogNorm())
for cut in cuts1:
    plt.scatter(no_gm_ic[cut],lengths[cut], marker='.', alpha=0.5)
plt.colorbar()
plt.show(block=(not load_first_result))


def do_gain_match(cut_masks, true_energies, init_guess=None, offset="none", ):
    gm_slices = []
    default_guess = []
    num_in_slice = []
    with cp.cuda.Device(gpu_device):
        for cut_mask, true_energy in zip(cut_masks, true_energies):
            gm_slices.append(cpp_gpu[cut_mask, :])
            default_guess.append(cp.asnumpy(true_energy/cp.mean(cp.sum(gm_slices[-1], axis=1))))
            num_in_slice.append(cp.shape(gm_slices[-1])[0])
            print('cut with true energy of %f MeV has %d events'%(true_energy, num_in_slice[-1]))
        if offset == 'none':
            default_guess = np.ones(1024)*np.average(default_guess)
            bounds=[(0, np.inf)]*1024
        elif offset == 'constant':
            default_guess = np.ones(1025)*np.average(default_guess)
            default_guess[-1] = 0
            bounds=[(0, np.inf)]*1024
            bounds.append((-np.inf, np.inf))
        if init_guess == None:
            init_guess = default_guess
        
        def obj_func(x):
            if offset == 'none':
                gains = x
            elif offset == 'constant':
                gains = x[:-1]
                offset_constant = x[-1]*1e4
            e_list = []
            for gm_slice in gm_slices:
                e_list.append(get_gm_ic(gains, gm_slice, True))
                if offset == 'constant':
                    e_list[-1] += offset_constant
            to_return = 0
            with cp.cuda.Device(gpu_device):
                for es, true_e, num in zip(e_list, true_energies, num_in_slice):
                    to_return += np.sqrt(cp.asnumpy(cp.sum((es - true_e)**2))/num)/true_energy*2.355
            return to_return
        
        def callback(intermediate_result):
            print(intermediate_result)
            if offset == 'none':
                gains = intermediate_result.x
            elif offset == 'constant':
                gains = intermediate_result.x[:-1]
            print(np.mean(gains), np.std(gains), np.min(gains), np.max(gains))

        print('objective function for initial guess: ', obj_func(init_guess))
        start_time = time.time()
        print('starting minimization')
        res =  optimize.minimize(obj_func, init_guess, callback=callback, bounds=bounds, options={'maxfun':1000000})
        print('time to perform minimization: %f s'%(time.time() - start_time))
        return res


if load_first_result:
    with open('res1_%s.pkl'%offset, 'rb') as f:
        res1 = pickle.load(f)
else:
    res1 = do_gain_match(cuts1, true_energies, offset=offset)
    with open('res1_%s.pkl'%offset, 'wb') as f:
        pickle.dump(res1, f)
print(res1)

def show_plots(res, cuts):
    if offset == 'none':
        gm_ic = get_gm_ic(res.x)
    elif offset == 'constant':
        gm_ic = get_gm_ic(res.x[:-1]) + 1e4*res.x[-1]
    plt.figure()
    plt.title('gain match applied, runs: '+str(runs))
    plt.hist2d(gm_ic[plt_mask], lengths[plt_mask], bins=rve_bins, norm=matplotlib.colors.LogNorm())
    plt.colorbar()
    plt.xlabel('Energy (MeV)')
    plt.ylabel('range (mm)')

    # np.save('energy_cb37815dc81a8e0abe11b70e577d05143ea7b5ab', gm_ic[plt_mask])
    # np.save('length_cb37815dc81a8e0abe11b70e577d05143ea7b5ab', lengths[plt_mask])

    plt.figure()
    plt.title('events used in gain match')
    plt.hist2d(gm_ic[plt_mask], lengths[plt_mask], bins=rve_bins, norm=matplotlib.colors.LogNorm())
    plt.colorbar()
    for cut in cuts:
        plt.scatter(gm_ic[cut],lengths[cut], marker='.', alpha=0.5)
    plt.xlabel('Energy (MeV)')
    plt.ylabel('range (mm)')

    #show pad plane image
    plt.figure()
    plt.title('gain matched cut')
    h5 = process_runs.get_h5_file(exp, runs[0])
    d = {}
    for i in range(len(res.x)):
        if i in h5.pad_to_xy_index:
            d[i] = res.x[i]
    im = h5.get_2d_image(d)
    plt.imshow(im)
    plt.colorbar()
    plt.show()

show_plots(res1, cuts1)

if offset == 'none':
    gm_ic = get_gm_ic(res1.x)
elif offset == 'constant':
    gm_ic = get_gm_ic(res1.x[:-1]) + 1e4*res1.x[-1]
cuts2 = []
verticies = [(1.75,57.13),(1.434,44.66),(1.615,26.58),(1.736,27)]
path = matplotlib.path.Path(verticies)
rve_points = np.vstack((gm_ic, lengths)).transpose()
cuts2.append(path.contains_points(rve_points)&veto_mask)
verticies = [(0.837,13.9),(0.8782,21.53),(0.8007, 21.84),(0.7192, 17.78),(0.7676, 13.86)]
path = matplotlib.path.Path(verticies)
rve_points = np.vstack((gm_ic, lengths)).transpose()
cuts2.append(path.contains_points(rve_points)&veto_mask)

fig = plt.figure()
plt.title('gain match cuts')
plt.hist2d(gm_ic[plt_mask], lengths[plt_mask], bins=rve_bins, norm=matplotlib.colors.LogNorm())
for cut in cuts2:
    plt.scatter(gm_ic[cut],lengths[cut], marker='.', alpha=0.5)
plt.colorbar()
plt.show()

if load_second_result:
    with open('res2_%s.pkl'%offset, 'rb') as f:
        res2 = pickle.load(f)
else:
    res2 = do_gain_match(cuts2, true_energies, offset=offset)
    with open('res2_%s.pkl'%offset, 'wb') as f:
        pickle.dump(res2, f)
print(res2)
show_plots(res2, cuts2)

if offset == 'none':
    gm_ic = get_gm_ic(res2.x)
elif offset == 'constant':
    gm_ic = get_gm_ic(res2.x[:-1]) + 1e4*res2.x[-1]


rve_points = np.vstack((gm_ic, lengths)).transpose()
verticies = [(0.728, 11.5), (0.843, 11),(0.849,15.9), (2.772,50.5), (2.772, 100), (2.401,100), (0.607,17.3)]
path = matplotlib.path.Path(verticies)
proton_cut_mask = path.contains_points(rve_points)&veto_mask
verticies = [(1.51,19.3),(9.17,86.9), (9.41,47.1), (1.51, 3)]
path = matplotlib.path.Path(verticies)
alpha_cut_mask = path.contains_points(rve_points)&veto_mask

if True:

    ##########################################
    #do polynomial field distortion correction
    ##########################################
    #list of (wieght, peak label) tuples. Objective function will include minimizing sum_i weight_i * std(peak i range)^2
    peak_widths_to_minimize = [(1, 'p1596'),  (1, 'a4434'), (1, 'p770'), (1, 'a2153')]
    #list of (weight, peak 1, peak 2) tuples.
    #Objective function will minimize sum_i weight_i ((mean(peak i1 range) - mean(peaki2 range) - (true peak i2 range - true peak i2 range))^2
    peak_spacings_to_preserve = [(1, 'a2153', 'a4434'), (1, 'p770', 'p1596'), (1, 'p1596', 'a2153')]

    N=1
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
    particles_to_plot = ['p1596', 'p770', 'protons', 'alphas']

    endpoints = process_runs.get_quantity('endpoints', exp, runs)

    timestamps = []
    for run in runs:
        processed_directory = '/egr/research-tpc/shared/Run_Data/run_%04d_raw_viewer/run_%04dsmart'%(run, run)
        timestamps.append(np.load(os.path.join(processed_directory, 'timestamps.npy')))
    timestamps = np.concatenate(timestamps, axis=0)

    MeV = gm_ic
    #ranges = np.sqrt(dzs**2 + dxys**2) #why is this different than ds = np.linalg.norm(endpoints[:,0] - endpoints[:,1], axis=1)?
    ranges = lengths

    #use track angle from pca rather than that exported by raw event viewer
    angles = []
    for axes in process_runs.get_quantity('principle_axes', exp, runs):
        dir = axes[0]
        angles.append(np.arctan2(np.sqrt(dir[0]**2 + dir[1]**2), np.abs(dir[2])))
    angles = np.array(angles)

    #estimate time since start of decay window to be time since first event in the window
    time_since_last_event = timestamps - np.roll(timestamps, 1)
    time_since_last_event[0] = .15 #we don't actuallly know what this is for the first event, so just putting a typical value for start of window
    start_of_current_winow = 0
    track_widths = process_runs.get_quantity('charge_width', exp, runs)

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
    label_dict['protons']=proton_cut_mask
    label_dict['alphas']=alpha_cut_mask

    cut_mask_dict['p1596'] = cuts2[0]
    cut_mask_dict['p770'] = cuts2[1]
    cut_mask_dict['protons']=proton_cut_mask
    cut_mask_dict['alphas']=alpha_cut_mask
    # cut_mask_dict['a4434'] = (ranges>25) & (ranges<50) & (MeV > 4.434) & (counts < 7e5) & veto_mask
    # cut_mask_dict['a2153'] = (ranges>18) & (ranges<28) & (counts>2.25e5) & (counts<3.4e5) & veto_mask
    # cut_mask_dict['a4434wr'] = (ranges>25) & (ranges<50) & (counts>5.9e5) & (counts < 7e5) & veto_mask
    # cut_mask_dict['a4434wor'] = (ranges>25) & (ranges<50) & (counts>4.5e5) & (counts < 5.7e5) & veto_mask
    # cut_mask_dict['a2153wr'] = (ranges>18) & (ranges<28) & (counts>2.83e5) & (counts<3.4e5) & veto_mask
    # cut_mask_dict['a2153wor'] = (ranges>18) & (ranges<26) & (counts>2.3e5) & (counts<2.7e5) & veto_mask
    # #cuts within 20 degrees of pad plan
    # cut_mask_dict['p1596pp'] = cut_mask_dict['p1596']&(angles>np.radians(70))
    # cut_mask_dict['p770pp'] = cut_mask_dict['p770']&(angles>np.radians(70))
    # cut_mask_dict['a4434pp'] = cut_mask_dict['a4434']&(angles>np.radians(70))

    if offset_endpoints:
        total_track_widths = np.array(process_runs.get_quantity('width_above_threshold', exp, runs))
        endpoints_offset_dir1 = endpoints[:, 0, :] - endpoints[:, 1, :]
        endpoints_offset_dir1 /= np.linalg.norm(endpoints_offset_dir1, axis=1)[:, np.newaxis]
        endpoints[:, 0, :] -= (total_track_widths[:, np.newaxis]/2)*endpoints_offset_dir1
        endpoints_offset_dir2 = endpoints[:, 1, :] - endpoints[:, 0, :]
        endpoints_offset_dir2 /= np.linalg.norm(endpoints_offset_dir2, axis=1)[:, np.newaxis]
        endpoints[:, 1, :] -= (total_track_widths[:, np.newaxis]/2)*endpoints_offset_dir2

    #plot showing selected events of each type
    rve_plt_mask = (ranges>0)&(ranges<150)&veto_mask&(MeV<10)
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
    fname = os.path.join(package_directory,fname_template%(exp, run, N))
    print('pickle file name: ', fname)
    if os.path.exists(fname):
        print('optimizer previously run, loading saved result')
        with open(fname, 'rb') as file:
            res =  pickle.load(file)
    else:
        print('optimizing a_ijk parameters')
        previous_fname = os.path.join(package_directory, fname_template%(exp, run, N-1))
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
            res = optimize.shgo(f_to_min, bounds, sampling_method='halton')
        elif opt_method == 'annealing':
            def callback(x, f, context):
                print(x, f, context)
                return False
            res = optimize.dual_annealing(f_to_min, bounds, x0=guess, maxiter=10000, callback=callback)
        elif opt_method == 'local':
            res = optimize.minimize(f_to_min, guess, callback=callback)
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
        #ax.hist(mapped_ranges[mask], bins=range_hist_bins, alpha=0.6, label='corrected range; std=%g'%np.std(mapped_ranges[mask]))
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

    #save results
    to_save = {}
    to_save['length'] = lengths
    to_save['energy'] = gm_ic
    to_save['endpoints'] = endpoints
    to_save['charge_width'] = process_runs.get_quantity('charge_width', exp, runs)
    to_save['counts_per_pad'] = cpp
    to_save['veto_pad_counts'] = veto_counts
    save_fname = 'e21072_121to128.pkl'
    with open(save_fname, 'wb') as f:
        pickle.dump(to_save, f)
USE_GPU = True

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
from scipy import fftpack

from raw_viewer import process_runs

gpu_device = 1
load_result1 = False
load_result2 = False
load_result3 = False


runs = (49,)
exp = 'e23035_prep_vault'

veto_thresh = 400
rve_bins = 400

lengths = process_runs.get_lengths(exp, runs)
cpp = process_runs.get_quantity('pad_charge', exp, runs)
veto_max = process_runs.get_max_veto_counts(exp, runs)
charge_widths = process_runs.get_quantity('charge_width', exp,runs)

veto_thresholds = np.ones(process_runs.raw_h5_file.NUM_PADS)*np.inf
for pad in process_runs.raw_h5_file.VETO_PADS:
    veto_thresholds[pad] = 500

max_pad_counts = process_runs.get_quantity('pad_max', exp, runs)
pads_railed = process_runs.get_quantity('railed_pads', exp, runs)
num_pads_railed = np.array([len(prl) for prl in pads_railed])
angles = process_runs.get_angle(exp, runs)

veto_mask = np.all(max_pad_counts<veto_thresholds, axis=1)&(num_pads_railed==0)&(np.degrees(angles)>8)

run_numbers, event_numbers = process_runs.get_run_and_event_numbers(exp, runs)

h5 = process_runs.get_h5_file(exp, runs[0])
freqs_to_use = 6
freq_bins_to_cut=len(h5.pad_plane) - freqs_to_use

with cp.cuda.Device(gpu_device):
    cpp_gpu = cp.array(cpp)

def get_gm_ic(gains, counts_per_pad=cpp_gpu, return_gpu=False):
    with cp.cuda.Device(gpu_device):
        gains_gpu = cp.array(gains)
        to_return = cp.einsum('ij, j', counts_per_pad, gains_gpu)
    if return_gpu:
        return to_return
    else:
        return cp.asnumpy(to_return)
    

def apply_gm_result(gm_result):
    return get_gm_ic(get_pad_gains(gm_result.x))
        
no_gm_ic = get_gm_ic(np.ones(1024))

# --- MODIFIED --- 
# Set your target energy. 
true_energies = [1.240]

cuts1 = []
# --- MODIFIED ---
# YOU MUST LOOK AT THE FIRST 2D HISTOGRAM TO FIND THESE RAW CHARGE AND LENGTH BOUNDS
# Replace the raw charge (e.g., 200000, 300000) and lengths (e.g., 10, 30) with your actual proton data bounds.
cuts1.append((no_gm_ic>200000) & (no_gm_ic<300000) & (lengths>10) & (lengths<30) & veto_mask)


plt.figure()
plt.hist(veto_max, 100)

fig = plt.figure()
plt_mask = veto_mask&(lengths<150)&(lengths>1)
plt.title('without gain match, runs: '+str(runs))
plt.hist2d(no_gm_ic[plt_mask], lengths[plt_mask], bins=rve_bins, norm=matplotlib.colors.LogNorm())
plt.xlabel('integrated charge')

fig = plt.figure()
plt.title('gain match cuts')
plt.hist2d(no_gm_ic[plt_mask], lengths[plt_mask], bins=rve_bins, norm=matplotlib.colors.LogNorm())
for cut in cuts1:
    plt.scatter(no_gm_ic[cut],lengths[cut], marker='.', alpha=0.5)
    print('counts in cut: ', len(no_gm_ic[cut]))
plt.colorbar()
plt.show(block=(not load_result1))

def get_init_spectrum_guess(init_gain_guess):  
    image = np.ones(np.shape(h5.pad_plane))*init_gain_guess
    to_return = fftpack.dctn(image, norm='ortho')[:-freq_bins_to_cut, :-freq_bins_to_cut]
    return to_return.flatten()

def get_pad_gains(x):
    sp = np.zeros(np.shape(h5.pad_plane))
    sp[:-freq_bins_to_cut, :-freq_bins_to_cut] = np.reshape(x,np.array(np.shape(sp)) - freq_bins_to_cut)
    gain_image =  fftpack.idctn(sp, norm='ortho')
    gains = np.zeros(1024)
    for pad in h5.pad_to_xy_index:
        if pad not in process_runs.raw_h5_file.VETO_PADS:
            gains[pad] = gain_image[h5.pad_to_xy_index[pad]]
    return gains


def do_gain_match(cut_masks, true_energies):
    h5 = process_runs.get_h5_file(exp, runs[0]) 
    gm_slices = []
    default_guess = []
    num_in_slice = []

    with cp.cuda.Device(gpu_device):
        for cut_mask, true_energy in zip(cut_masks, true_energies):
            gm_slices.append(cpp_gpu[cut_mask, :])
            default_guess.append(cp.asnumpy(true_energy/cp.mean(cp.sum(gm_slices[-1], axis=1))))
            num_in_slice.append(cp.shape(gm_slices[-1])[0])
            print('cut with true energy of %f MeV has %d events'%(true_energy, num_in_slice[-1]))
        
        init_guess = get_init_spectrum_guess(np.average(default_guess))
        print('num params:', len(init_guess))
        bounds=np.array([(-np.inf, np.inf)]*len(init_guess))
       
        def obj_func(x):
            gains = get_pad_gains(x)
            e_list = []
            for gm_slice in gm_slices:
                e_list.append(get_gm_ic(gains, gm_slice, True))
            to_return = 0
            with cp.cuda.Device(gpu_device):
                for es, true_e, num in zip(e_list, true_energies, num_in_slice):
                    to_return += np.sqrt(cp.asnumpy(cp.sum((es - true_e)**2))/num)/true_energy*2.355
            return to_return
        
        def callback(intermediate_result):
            gains = get_pad_gains(intermediate_result.x)
            print('gains',np.mean(gains), np.std(gains), np.min(gains), np.max(gains))
            
        print('objective function for initial guess: ', obj_func(init_guess))
        start_time = time.time()
        print('starting minimization')
        res =  optimize.minimize(obj_func, init_guess, callback=callback, bounds=bounds, options={'maxfun':1000000})
        print('time to perform minimization: %f s'%(time.time() - start_time))
        res.cut_masks = cut_masks
        res.true_energies = true_energies
        res.runs = runs
        res.freq_to_cut = freq_bins_to_cut
        res.pad_gains = get_pad_gains(res.x)
        return res

if load_result1:
    with open('fft%d_res1.pkl'%(freqs_to_use), 'rb') as f:
        res1 = pickle.load(f)
else:
    res1 = do_gain_match(cuts1, true_energies)
    with open('fft%d_res1.pkl'%(freqs_to_use), 'wb') as f:
        pickle.dump(res1, f)

def show_plots(res,block=False):
    gm_ic = apply_gm_result(res)
    plt.figure()
    plt.title('gain match applied, runs: '+str(runs))
    plt.hist2d(gm_ic[plt_mask], lengths[plt_mask], bins=rve_bins, norm=matplotlib.colors.LogNorm())
    plt.colorbar()
    plt.xlabel('Energy (MeV)')
    plt.ylabel('range (mm)')

    try:
        plt.figure()
        plt.title('events used in gain match')
        plt.hist2d(gm_ic[plt_mask], lengths[plt_mask], bins=rve_bins, norm=matplotlib.colors.LogNorm())
        plt.colorbar()
        for cut in res.cut_masks:
            plt.scatter(gm_ic[cut],lengths[cut], marker='.', alpha=0.1)
        plt.xlabel('Energy (MeV)')
        plt.ylabel('range (mm)')
    except:
        pass

    plt.figure()
    plt.title('pad gains')
    h5 = process_runs.get_h5_file(exp, runs[0])
    pad_gains = get_pad_gains(res.x)
    d = {}
    for i in range(len(pad_gains)):
        if i in h5.pad_to_xy_index:
            d[i] = pad_gains[i]
    im = h5.get_2d_image(d)
    
    plt.imshow(im,vmin = np.min(pad_gains[pad_gains>0]), vmax = np.max(pad_gains))
    plt.colorbar()
    plt.show(block=False)

show_plots(res1)

if True:
    gm_ic = apply_gm_result(res1)
    cuts2 = []
    # --- MODIFIED ---
    # Tighten the bounds around the 1.24 MeV peak now that it is roughly calibrated
    cuts2.append((gm_ic > 1.15) & (gm_ic < 1.35) & (lengths > 10) & (lengths < 30) & veto_mask)
    true_energies2 = [1.240]

    fig = plt.figure()
    plt.title('gain match cuts')
    plt.hist2d(gm_ic[plt_mask], lengths[plt_mask], bins=rve_bins, norm=matplotlib.colors.LogNorm())
    for cut in cuts2:
        plt.scatter(gm_ic[cut],lengths[cut], marker='.', alpha=0.5)
    plt.colorbar()
    plt.show()

    if load_result2:
        with open('fft%d_res2.pkl'%(freqs_to_use), 'rb') as f:
            res2 = pickle.load(f)
    else:
        res2 = do_gain_match(cuts2, true_energies2)
        with open('fft%d_res2.pkl'%(freqs_to_use), 'wb') as f:
            pickle.dump(res2, f)
    print(res2)
    show_plots(res2)


if True:
    gm_ic2 = apply_gm_result(res2)
    cuts3 = []
    # --- MODIFIED ---
    # Final, tightest cut around the 1.24 MeV peak
    cuts3.append((gm_ic > 1.20) & (gm_ic < 1.28) & (lengths > 10) & (lengths < 30) & veto_mask)
    true_energies3 = [1.240]

    fig = plt.figure()
    plt.title('gain match cuts')
    plt.hist2d(gm_ic2[plt_mask], lengths[plt_mask], bins=rve_bins, norm=matplotlib.colors.LogNorm())
    for cut in cuts3:
        plt.scatter(gm_ic2[cut],lengths[cut], marker='.', alpha=0.25)
    plt.colorbar()
    plt.show()

    if load_result3:
        with open('fft%d_res3.pkl'%(freqs_to_use), 'rb') as f:
            res3 = pickle.load(f)
    else:
        res3 = do_gain_match(cuts3, true_energies3)
        with open('fft%d_res3.pkl'%(freqs_to_use), 'wb') as f:
            pickle.dump(res3, f)
    print(res3)
    show_plots(res3)


gm_ic3 = apply_gm_result(res3)
import ROOT
c1 = ROOT.TCanvas()
c1.cd()
# --- MODIFIED ---
# Adjust the ROOT fit window and parameters for a single 1.24 MeV peak
emin, emax = 1.0, 1.5
energy_hist = ROOT.TH1D("h1", "h1", 100,  emin, emax)
gm2_ic = apply_gm_result(res3)
energy_hist.Fill(gm_ic3[veto_mask])
energy_hist.Draw()

# Fit function: Constant background [0] + single Gaussian [1, 2, 3]
func1 = ROOT.TF1('f1', '[0] + gaus(1)',  emin, emax)
# background, height, mu, sigma
func1.SetParameters(0, 10, 1.24, 0.05)
func1.SetParLimits(0, 0, np.inf)
func1.SetParLimits(1, 0, np.inf)
func1.SetParLimits(2, 1.15, 1.35)
func1.SetParLimits(3, 0.01, 0.2)

energy_hist.Fit(func1, "L")
energy_hist.Fit(func1, "L")
energy_hist.Fit(func1, "L")
fit_params1 = np.zeros(4)
func1.GetParameters(fit_params1)
print('fwhm value:', 2.355*fit_params1[3]/fit_params1[2])

angles = process_runs.get_angle(exp, runs)
plt.figure()
plt.scatter(gm_ic3[veto_mask],lengths[veto_mask],c=np.degrees(angles[veto_mask]), marker='.')
plt.colorbar()
plt.show(block=False)
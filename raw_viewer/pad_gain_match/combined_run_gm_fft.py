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


runs = (61,62,63)#(20,)#(20,)#(38,49)#17,20,21,38,49,60,
exp = 'e23035_prep_vault'

veto_thresh = 400
rve_bins = 400

lengths = process_runs.get_lengths(exp, runs)
cpp = process_runs.get_quantity('pad_charge', exp, runs)
#veto_counts = process_runs.get_veto_counts(exp, runs)
veto_max = process_runs.get_max_veto_counts(exp, runs)
charge_widths = process_runs.get_quantity('charge_width', exp,runs)
#veto_mask = (veto_max < veto_thresh)&(charge_widths>3.25)#(veto_counts < veto_thresh)&(charge_widths>2.5)
veto_thresholds = np.ones(process_runs.raw_h5_file.NUM_PADS)*np.inf
veto_thresholds[253]=170
veto_thresholds[254]=170
veto_thresholds[508]=200
veto_thresholds[509]=600
veto_thresholds[763]=280
veto_thresholds[764]=260
veto_thresholds *= 400/600.
max_pad_counts = process_runs.get_quantity('pad_max', exp, runs)
veto_mask = np.all(max_pad_counts<veto_thresholds, axis=1)
run_numbers, event_numbers = process_runs.get_run_and_event_numbers(exp, runs)

h5 = process_runs.get_h5_file(exp, runs[0])
freqs_to_use = 10
freq_bins_to_cut=len(h5.pad_plane) - freqs_to_use


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
    

def apply_gm_result(gm_result):
    return get_gm_ic(get_pad_gains(gm_result.x))
        

no_gm_ic = get_gm_ic(np.ones(1024))

#set up initial gain match cuts
cuts1 = []
true_energies = [6.28808, 6.7883,8.784 ]#, 0.7856]# only includes energy deposited as ionization
cuts1.append((no_gm_ic>1.1e6) & (no_gm_ic<1.31e6) & (lengths>50) & (lengths<65.5) & veto_mask)
cuts1.append((no_gm_ic>1.13e6) & (no_gm_ic<1.4e6) & (lengths>66) & (lengths<76) & veto_mask)
cuts1.append((no_gm_ic>1.55e6) & (no_gm_ic<1.85e6) & (lengths>90) & (lengths< 105) & veto_mask)



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
    # plt.figure()
    # plt.title('desired init pad gains')
    # plt.imshow(fftpack.dctn(image, norm='ortho'))
    # plt.show()
    
    # plt.figure()
    # plt.title('desired init pad gains')
    # plt.imshow(fftpack.idctn(fftpack.dctn(image, norm='ortho'), norm='ortho'))
    # plt.show()

    to_return = fftpack.dctn(image, norm='ortho')[:-freq_bins_to_cut, :-freq_bins_to_cut]
    return to_return.flatten()

def get_pad_gains(x):
    sp = np.zeros(np.shape(h5.pad_plane))
    sp[:-freq_bins_to_cut, :-freq_bins_to_cut] = np.reshape(x,np.array(np.shape(sp)) - freq_bins_to_cut)

    gain_image =  fftpack.idctn(sp, norm='ortho')
    # plt.figure()
    # plt.title('pad gains assigned')
    # plt.imshow(gain_image)
    # plt.show()

    gains = np.zeros(1024)
    for pad in h5.pad_to_xy_index:
        if pad not in process_runs.raw_h5_file.VETO_PADS:
            gains[pad] = gain_image[h5.pad_to_xy_index[pad]]
    return gains


def do_gain_match(cut_masks, true_energies):
    h5 = process_runs.get_h5_file(exp, runs[0]) #used to make pad plane images
    
    gm_slices = []
    default_guess = []
    num_in_slice = []
    run_indexs = [] 

    with cp.cuda.Device(gpu_device):
        for cut_mask, true_energy in zip(cut_masks, true_energies):
            gm_slices.append(cpp_gpu[cut_mask, :])
            default_guess.append(cp.asnumpy(true_energy/cp.mean(cp.sum(gm_slices[-1], axis=1))))
            num_in_slice.append(cp.shape(gm_slices[-1])[0])
            print('cut with true energy of %f MeV has %d events'%(true_energy, num_in_slice[-1]))
            run_indexs.append(cp.array(run_numbers[cut_mask]))
            for i, r in enumerate(runs):
                run_indexs[-1][r] = i
        
    
        init_guess = get_init_spectrum_guess(np.average(default_guess))
        print('num params:', len(init_guess))
        bounds=np.array([(-np.inf, np.inf)]*len(init_guess))

        # plt.figure()
        # print(np.average(default_guess))
        # print(get_pad_gains(init_guess))
        # plt.hist2d(get_gm_ic(get_pad_gains(init_guess))[plt_mask], lengths[plt_mask], bins=rve_bins, norm=matplotlib.colors.LogNorm())
        # plt.show()
       
        def obj_func(x):
            gains = get_pad_gains(x)
            e_list = []
            
            for gm_slice, ri in zip(gm_slices, run_indexs):
                e_list.append(get_gm_ic(gains, gm_slice, True))

            to_return = 0
            with cp.cuda.Device(gpu_device):
                for es, true_e, num in zip(e_list, true_energies, num_in_slice):
                    to_return += np.sqrt(cp.asnumpy(cp.sum((es - true_e)**2))/num)/true_energy*2.355
            return to_return
        
        def callback(intermediate_result):
            print(intermediate_result)
           
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
#print(res1)

def show_plots(res,block=False):
    gm_ic = apply_gm_result(res)
    plt.figure()
    plt.title('gain match applied, runs: '+str(runs))
    plt.hist2d(gm_ic[plt_mask], lengths[plt_mask], bins=rve_bins, norm=matplotlib.colors.LogNorm())
    plt.colorbar()
    plt.xlabel('Energy (MeV)')
    plt.ylabel('range (mm)')

    # np.save('energy_cb37815dc81a8e0abe11b70e577d05143ea7b5ab', gm_ic[plt_mask])
    # np.save('length_cb37815dc81a8e0abe11b70e577d05143ea7b5ab', lengths[plt_mask])

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

    #show pad plane image
    plt.figure()
    plt.title('gain matched cut')
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

#show_plots(res1)

#redo gain match using selection based on original gain match
if True:
    gm_ic = apply_gm_result(res1)
    cuts2 = []
    cuts2.append((gm_ic >6.0)&(gm_ic<6.58)&(lengths>50)&(lengths<69) & veto_mask)
    cuts2.append((gm_ic >6.58)&(gm_ic<7.5)&(lengths>59)&(lengths<80) & veto_mask)
    cuts2.append((gm_ic>8)&(gm_ic<9.25)&(lengths>90)&(lengths<105) & veto_mask)
    true_energies2 = [6.288, 6.7783, 8.78486]#[6.7783]

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
    #show_plots(res2)



    #redo gain match using selection based on original gain match
if True:
    gm_ic2 = apply_gm_result(res2)
    cuts3 = []
    cuts3.append(((gm_ic2 >6.0)&(gm_ic2<6.6)&(lengths>50)&(lengths<64)|
                  (gm_ic2 >6.1)&(gm_ic2<6.53)&(lengths>50)&(lengths<69.4)) & veto_mask)
    cuts3.append((gm_ic2 >6.6)&(gm_ic2<7.35)&(lengths>60)&(lengths<77) & veto_mask)
    cuts3.append((gm_ic2>8.6)&(gm_ic2<9.1)&(lengths>92)&(lengths<105) & veto_mask)
    true_energies3 = [6.288, 6.7783, 8.78486]#[6.7783]

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

if True:
    import ROOT
    c1 = ROOT.TCanvas()
    c1.cd()
    emin, emax = 6,7
    energy_hist = ROOT.TH1D("h1", "h1", 100,  emin, emax)
    gm2_ic = apply_gm_result(res3)
    energy_hist.Fill(gm_ic2[veto_mask])
    energy_hist.Draw()
    func1 = ROOT.TF1('f1', '[0] + gaus(1) + gaus(4)',  emin, emax)
    #background, height, mu, sigam, height, mu, sigma
    func1.SetParameters(0, 10, 6.28, 0.05, 10, 6.88, 0.05)
    func1.SetParLimits(0,0,np.inf)
    func1.SetParLimits(1,0,np.inf)
    func1.SetParLimits(2,6.1,6.5)
    func1.SetParLimits(3, 0, 0.5)
    func1.SetParLimits(4, 0, np.inf)
    func1.SetParLimits(5, 6.6, 7)
    func1.SetParLimits(6, 0, 0.5)
    energy_hist.Fit(func1, "L")
    energy_hist.Fit(func1, "L")
    energy_hist.Fit(func1, "L")
    fit_params1 = np.zeros(7)
    func1.GetParameters(fit_params1)
    print('fwhm values:', 2.355*fit_params1[3]/fit_params1[2], 2.355*fit_params1[6]/fit_params1[5])
    ROOT.gStyle.SetOptFit(1111)


    c2 = ROOT.TCanvas()
    c2.cd()
    emin, emax= 8.3, 9.2
    energy_hist2 = ROOT.TH1D("h2", "h2", 20, emin, emax)
    energy_hist2.Fill(gm_ic2[veto_mask])
    energy_hist2.Draw()
    func2 = ROOT.TF1('f1', '[0] + gaus(1)',  emin, emax)
    func2.SetParameters(0, 10, 8.78, 0.05, )
    func2.SetParLimits(0,0,np.inf)
    func2.SetParLimits(1,0,np.inf)
    func2.SetParLimits(2,emin,emax)
    func2.SetParLimits(3, 0, 0.5)

    energy_hist2.Fit(func2, "L")
    energy_hist2.Fit(func2, "L")
    energy_hist2.Fit(func2, "L")
    fit_params2 = np.zeros(4)
    func2.GetParameters(fit_params2)

    print('fwhm values:', 2.355*fit_params1[3]/fit_params1[2], 2.355*fit_params1[6]/fit_params1[5], 2.355*fit_params2[3]/fit_params2[2])
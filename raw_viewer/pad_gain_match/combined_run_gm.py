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

from raw_viewer import process_runs

gpu_device = 1
load_result1 = False
load_result2 = False
load_result3 = False


runs = (61,62,63)#(20,)#(20,)#(38,49)#17,20,21,38,49,60,
exp = 'e23035_prep_vault'

veto_thresh = 400
rve_bins = 400
offset = 'constant' #'constant' or 'none'
per_run_variation = False #allow gain and offset (if applicable) to vary per run

lengths = process_runs.get_lengths(exp, runs)
cpp = process_runs.get_quantity('pad_charge', exp, runs)
#veto_counts = process_runs.get_veto_counts(exp, runs)
veto_max = process_runs.get_max_veto_counts(exp, runs)
charge_widths = process_runs.get_quantity('charge_width', exp,runs)
veto_mask = (veto_max < veto_thresh)&(charge_widths>3.25)#(veto_counts < veto_thresh)&(charge_widths>2.5)
run_numbers, event_numbers = process_runs.get_run_and_event_numbers(exp, runs)

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
    if gm_result.per_run_variation:
        run_indexes = np.copy(run_numbers)
        for i, r in enumerate(runs):
            run_indexes[run_numbers == r] = i
        if gm_result.offset == 'none':
            gains = gm_result.x[:-1*(len(runs)-1)]
            per_run_gain = np.array(gm_result.x[-1*(len(runs)-1):])
            per_run_gain = np.concatenate((per_run_gain, np.array([1.0])))
            return get_gm_ic(gains)*per_run_gain[run_indexes]
        elif gm_result.offset == 'constant':
            per_run_gain = gm_result.x[-2*len(runs):-1*len(runs)]
            per_run_offset = gm_result.x[-1*len(runs):]
            return (get_gm_ic(gm_result.x[:-2*len(runs)])*per_run_gain[run_indexes] +
                    1e4*per_run_offset[run_indexes])
    else:
        if gm_result.offset == 'none':
            return get_gm_ic(gm_result.x)
        elif gm_result.offset == 'constant':
            return get_gm_ic(gm_result.x[:-1]) + 1e4*gm_result.x[-1]

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


def do_gain_match(cut_masks, true_energies, init_guess=None, offset='none'):
    gm_slices = []
    default_guess = []
    num_in_slice = []
    run_indexs = [] 
    num_params = 1024
    if per_run_variation:
        if offset == 'none':
            num_params += len(runs)-1
        elif offset == 'constant':
            num_params += 2*(len(runs)-1)
    else:
        if offset == 'constant':
            num_params += 1
    with cp.cuda.Device(gpu_device):
        for cut_mask, true_energy in zip(cut_masks, true_energies):
            gm_slices.append(cpp_gpu[cut_mask, :])
            default_guess.append(cp.asnumpy(true_energy/cp.mean(cp.sum(gm_slices[-1], axis=1))))
            num_in_slice.append(cp.shape(gm_slices[-1])[0])
            print('cut with true energy of %f MeV has %d events'%(true_energy, num_in_slice[-1]))
            run_indexs.append(cp.array(run_numbers[cut_mask]))
            for i, r in enumerate(runs):
                run_indexs[-1][r] = i
        if offset == 'none':
            default_guess = np.ones(num_params)*np.average(default_guess)
            bounds=np.array([(0, np.inf)]*num_params)
            if per_run_variation:
                default_guess[-1*len(runs):] = 1
                bounds[-1*len(runs):] = (0,2)
        elif offset == 'constant':
            default_guess = np.ones(num_params)*np.average(default_guess)
            if per_run_variation:
                default_guess[-2*len(runs):-1*len(runs)] = 1
                default_guess[-1*len(runs):] = 0
            default_guess[-1] = 0
            bounds=np.array([(0, np.inf)]*num_params)
            if per_run_variation:
                bounds[-2*len(runs):-1*len(runs)] = (0,2)
                bounds[-1*len(runs):] = (-np.inf, np.inf)
            else:
                bounds[-1] = (-np.inf, np.inf)
        if type(init_guess) == type(None):
            init_guess = default_guess
        
        def obj_func(x):
            if offset == 'none':
                if per_run_variation:
                    gains = x[:-1*(len(runs)-1)]
                    per_run_gain = cp.array(x[-1*(len(runs)-1):])
                    per_run_gain = cp.concatenate((per_run_gain, cp.array([1.0])))
                else:
                    gains = x
            elif offset == 'constant':
                if per_run_variation:
                    gains = x[:-2*len(runs)]
                    per_run_gain =cp.array(x[-2*len(runs):-1*len(runs)])
                    per_run_offset = cp.array(x[-1*len(runs):])
                else:
                    gains = x[:-1]
                offset_constant = x[-1]*1e4
            e_list = []
            for gm_slice, ri in zip(gm_slices, run_indexs):
                e_list.append(get_gm_ic(gains, gm_slice, True))
                if offset == 'none' and per_run_variation:
                    e_list[-1] *= per_run_gain[ri]
                if offset == 'constant':
                    if per_run_variation:
                        e_list[-1] += per_run_offset[ri]*1e4
                    else:
                        e_list[-1] += offset_constant
            to_return = 0
            with cp.cuda.Device(gpu_device):
                for es, true_e, num in zip(e_list, true_energies, num_in_slice):
                    to_return += np.sqrt(cp.asnumpy(cp.sum((es - true_e)**2))/num)/true_energy*2.355
            return to_return
        
        def callback(intermediate_result):
            print(intermediate_result)
            if offset == 'none':
                if per_run_variation:
                    gains = intermediate_result.x[:-1*len(runs)]
                    per_run_gains = intermediate_result.x[-1*(len(runs)-1):]
                else:
                    gains = intermediate_result.x
            elif offset == 'constant':
                if per_run_variation:
                    gains = intermediate_result.x[:-2*len(runs)]
                    per_run_gains = intermediate_result.x[-2*len(runs):-1*len(runs)]
                else:
                    gains = intermediate_result.x[:-1]
            print('gains',np.mean(gains), np.std(gains), np.min(gains), np.max(gains))
            if per_run_variation:
                print('per run gains:', np.mean(per_run_gains), np.std(per_run_gains), np.min(per_run_gains), np.max(per_run_gains))

        print('objective function for initial guess: ', obj_func(init_guess))
        start_time = time.time()
        print('starting minimization')
        res =  optimize.minimize(obj_func, init_guess, callback=callback, bounds=bounds, options={'maxfun':1000000})
        print('time to perform minimization: %f s'%(time.time() - start_time))
        res.cut_masks = cut_masks
        res.true_energies = true_energies
        res.runs = runs
        res.offset = offset
        res.per_run_variation = per_run_variation
        return res


if load_result1:
    with open('res1_%s_%s.pkl'%(offset, per_run_variation), 'rb') as f:
        res1 = pickle.load(f)
else:
    res1 = do_gain_match(cuts1, true_energies, offset=offset)
    with open('res1_%s_%s.pkl'%(offset, per_run_variation), 'wb') as f:
        pickle.dump(res1, f)
print(res1)

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
    d = {}
    for i in range(len(res.x)):
        if i in h5.pad_to_xy_index:
            d[i] = res.x[i]
    im = h5.get_2d_image(d)
    
    plt.imshow(im,vmin = np.min(res.x), vmax = np.max(res.x))
    plt.colorbar()
    plt.show(block=False)

show_plots(res1)

#redo gain match using selection based on original gain match
if True:
    offset2 = offset
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
        with open('res2_%s_%s.pkl'%(offset2, per_run_variation), 'rb') as f:
            res2 = pickle.load(f)
    else:
        res2 = do_gain_match(cuts2, true_energies2, offset=offset2)
        with open('res2_%s_%s.pkl'%(offset2, per_run_variation), 'wb') as f:
            pickle.dump(res2, f)
    print(res2)
    show_plots(res2)

if False:
    import ROOT
    energy_hist = ROOT.TH1D("h1", "h1", 100, 6.0, 7)
    if res2.offset == 'none':
        gm_ic2 = get_gm_ic(res2.x)
    if res2.offset == 'constant':
        gm_ic2 = get_gm_ic(res2.x[:-1])
        gm_ic2 += 1e4*res2.x[-1]
    energy_hist.Fill(gm_ic2[veto_mask])
    energy_hist.Draw()

    energy_hist2 = ROOT.TH1D("h2", "h2", 20, 8.2, 9.2)
    energy_hist2.Fill(gm_ic2[veto_mask])
    energy_hist2.Draw()

    #redo gain match using selection based on original gain match
if True:
    offset3 = offset2
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
        with open('res3_%s_%s.pkl'%(offset3, per_run_variation), 'rb') as f:
            res3 = pickle.load(f)
    else:
        
        res3 = do_gain_match(cuts3, true_energies3, offset=offset3, init_guess=None)
        with open('res3_%s_%s.pkl'%(offset3, per_run_variation), 'wb') as f:
            pickle.dump(res3, f)
    print(res3)
    show_plots(res3)
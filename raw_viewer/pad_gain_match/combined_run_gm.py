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

from raw_viewer.pad_gain_match import process_runs

gpu_device = 2
load_first_result = False
load_second_result = True


runs = (121, 122, 123, 124, 125, 126, 127, 128)
veto_thresh = 10000
exp = 'e21072'
rve_bins = 500

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
    plt.scatter(no_gm_ic[cut],lengths[cut], marker='.')
plt.colorbar()
plt.show()


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
        elif offset == 'constant':
            default_guess = np.ones(1024)*np.average(default_guess)
            default_guess[-1] = 0
        if init_guess == None:
            init_guess = default_guess
        
        def obj_func(x):
            if offset == 'none':
                gains = x
            elif offset == 'constant':
                gains = x[:-1]
                offset_constant = x[-1]
            e_list = []
            for gm_slice in gm_slices:
                e_list.append(get_gm_ic(gains, gm_slice, True))
                if offset == 'constant':
                    e_list[-1] += offset_constant
            to_return = 0
            with cp.cuda.Device(gpu_device):
                for es, true_e, num in zip(e_list, true_energies, num_in_slice):
                    to_return += np.sqrt(cp.asnumpy(cp.sum((es - true_energy)**2))/num)/true_energy*2.355
            return to_return
        
        def callback(intermediate_result):
            print(intermediate_result)
            gains = intermediate_result.x
            print(np.mean(gains), np.std(gains), np.min(gains), np.max(gains))
        start_time = time.time()
        res =  optimize.minimize(obj_func, init_guess, bounds=[(0, np.inf)]*1024, callback=callback, options={'maxfun':1000000})
        print('time to perform minimization: %f s'%(time.time() - start_time))


if load_first_result:
    with open('res1.pkl', 'rb') as f:
        res = pickle.load(f)
else:
    res = do_gain_match(cuts1, true_energies)
    with open('res1.pkl', 'wb') as f:
        pickle.dump(f, res)
print(res)

def show_plots():
    gm_ic = get_gm_ic(res.x)
    plt.figure()
    plt.title('gain match applied, runs: '+str(runs))
    plt.hist2d(gm_ic[plt_mask], lengths[plt_mask], bins=rve_bins, norm=matplotlib.colors.LogNorm())
    plt.colorbar()
    plt.xlabel('Energy (MeV)')
    plt.ylabel('range (mm)')

    # np.save('energy_cb37815dc81a8e0abe11b70e577d05143ea7b5ab', gm_ic[plt_mask])
    # np.save('length_cb37815dc81a8e0abe11b70e577d05143ea7b5ab', lengths[plt_mask])

    #show selected events
    plt.figure()
    plt.title('events used for gain match')
    plt.hist2d(gm_ic[plt_mask], lengths[plt_mask], bins=rve_bins, norm=matplotlib.colors.LogNorm())
    plt.scatter(get_gm_ic(res.x, cpp_gpu[gm_cut_mask]), lengths[gm_cut_mask], marker='.', c='r')

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

show_plots()

gm_ic = get_gm_ic(res.x)
gm_cut_mask = (lengths>25)&(lengths<56)&(gm_ic > 1.6)&(gm_ic<1.7)  & veto_mask
with cp.cuda.Device(gpu_device):
    cpp_gm_cut_gpu = cpp_gpu[gm_cut_mask, :]
    print('cpp_gm_cut shape: ', cp.shape(cpp_gm_cut_gpu))
    print(cp.mean(get_gm_ic(np.ones(1024), cpp_gm_cut_gpu)), cp.std(get_gm_ic(np.ones(1024), cpp_gm_cut_gpu)))
    avg_ic_gm_cut_gpu = cp.mean(get_gm_ic(np.ones(1024), cpp_gm_cut_gpu, True))
    avg_ic_gm_cut = cp.asnumpy(avg_ic_gm_cut_gpu)
print(avg_ic_gm_cut)
num_in_cut = cp.shape(cpp_gm_cut_gpu)[0]
print('events in gm cut:', num_in_cut)

res1 = res
init_guess = res1.x
print(obj_func(init_guess))
start_time = time.time()
res = optimize.minimize(obj_func, init_guess, bounds=[(0, np.inf)]*1024, callback=callback, options={'maxfun':1000000})
print(res)
print('minimization took %f s'%time.time() - start_time)
if load_second_result:
    with open('res2.pkl', 'rb') as f:
        res = pickle.load(f)
else:
    res = optimize.minimize(obj_func, init_guess, bounds=[(0, np.inf)]*1024, callback=callback, options={'maxfun':1000000})
    with open('res2.pkl', 'wb') as f:
        pickle.dump(f, res)

show_plots()

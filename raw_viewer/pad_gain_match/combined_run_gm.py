USE_GPU = True

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

if False:
    runs = (96, 97, 98, 100, 101, 102, 104, 105, 106)
    veto_thresh = 10e3
    exp = 'e21072'
    rve_bins = 500
    gm_cut1 = ((7.7e4, 18), (9.14e4,18),(1e5,21), (9.2e4,21.9),(8.3e4,21.1), (7.7e4, 18))
    gm_cut1_path = Path(gm_cut1, closed=True)
else:
    runs = (121,121, 122, 123, 124, 125, 126, 127, 128)
    veto_thresh = 1.35e4
    exp = 'e21072'
    rve_bins = 500
    gm_cut1 = ((2.023e5,56.8), (2.19e5,56.8), (2.188e5,33.52), (1.94e5, 34.41), (1.791e5, 45.28),(2.023e5,56.8))
    gm_cut1_path = Path(gm_cut1, closed=True)

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

plt.figure()
plt.hist(veto_counts, 100)

fig = plt.figure()
plt_mask = veto_mask&(lengths<100)
plt.title('without gain match, runs: '+str(runs))
ax = fig.add_subplot(111)
no_gm_ic = get_gm_ic(np.ones(1024))
plt.hist2d(no_gm_ic[plt_mask], lengths[plt_mask], bins=rve_bins, norm=matplotlib.colors.LogNorm())
ax.add_patch(patches.PathPatch(gm_cut1_path, fill=False))
plt.colorbar()
plt.show()

gm1_cut_mask = gm_cut1_path.contains_points(np.vstack((no_gm_ic, lengths)).transpose())

with cp.cuda.Device(gpu_device):
    cpp_gm1cut_gpu = cpp_gpu[gm1_cut_mask, :]
    print('cpp_gm1cut shape: ', cp.shape(cpp_gm1cut_gpu))
    print(cp.mean(get_gm_ic(np.ones(1024), cpp_gm1cut_gpu)), cp.std(get_gm_ic(np.ones(1024), cpp_gm1cut_gpu)))
    avg_ic_gm1cut_gpu = cp.mean(get_gm_ic(np.ones(1024), cpp_gm1cut_gpu, True))
    avg_ic_gm1cut = cp.asnumpy(avg_ic_gm1cut_gpu)

print(avg_ic_gm1cut)
num_in_cut = cp.shape(cpp_gm1cut_gpu)[0]
print('events in gm cut:', num_in_cut)
def obj_func(gains):
    ics_gpu = get_gm_ic(gains, cpp_gm1cut_gpu, True)
    with cp.cuda.Device(gpu_device):
        return np.sqrt(cp.asnumpy(cp.sum((ics_gpu - avg_ic_gm1cut_gpu)**2))/num_in_cut)/avg_ic_gm1cut*2.355
    
def callback(intermediate_result):
    print(intermediate_result)
    gains = intermediate_result.x
    print(np.mean(gains), np.std(gains), np.min(gains), np.max(gains))
print(obj_func(np.ones(1024)))
res = optimize.minimize(obj_func, np.ones(1024), bounds=[(0.5,2)]*1024, callback=callback, options={'maxfun':1000000})
print(res)

gm_ic = get_gm_ic(res.x)
fig = plt.figure()
plt.title('gain match applied, runs: '+str(runs))
plt.hist2d(gm_ic[plt_mask], lengths[plt_mask], bins=rve_bins, norm=matplotlib.colors.LogNorm())
plt.colorbar()
plt.show()
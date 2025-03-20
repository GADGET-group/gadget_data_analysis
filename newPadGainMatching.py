import numpy as np
import cupy
from cupyx.profiler import benchmark
import scipy.optimize as opt
import h5py
import time
import pandas as pd

veto_pads = [253,254,508,509,763,764,1018,1019]

f = h5py.File("/egr/research-tpc/dopferjo/pad_gainmatch_events.h5",'r')
real_events = []
for event in range(100000):
    real_events.append(f['get/evt%d_pad_data'%event])

# Removing veto pads (even though all of their data is 0, we dont want the minimizer to use them to calculate the gain on the measurements pads)
df = pd.DataFrame(real_events)
df = df.drop(df.columns[[253,254,508,509,763,764,1018,1019]],axis=1)


npads = 1012

# print(np.shape(real_events))
# print(type(real_events))
# pad_gains = np.random.randn(npads)*0.07 + 1


Etrue = 8.7849

# fake_events = []
# nevents = 100#int(1e3)
# average_percent_fire = 0.1
# noise_amplitude = 0.01
# for i in range(nevents):
#     #pick pads to fire
#     e_deposited_so_far = 0
#     event = np.zeros(npads)
#     while e_deposited_so_far < Etrue:
#         pad = np.random.randint(0, npads)
#         e_dep = (1+np.random.randn()*0.1)*Etrue*average_percent_fire
#         if e_dep < 0:
#             e_dep = 0
#         if e_deposited_so_far +e_dep > Etrue:
#             e_dep = Etrue - e_deposited_so_far
#         e_deposited_so_far += e_dep
#         event[pad] += e_dep/pad_gains[pad] + noise_amplitude*np.random.randn()
#     fake_events.append(event)


real_events = cupy.array(df)
print('Finished reading in data, now finding pad gains')
def to_minimize(g):
    g = cupy.array([g]).T
    #to_return = 0
    #for event in real_events:
    #    to_return += (cupy.sum(g*event) - Etrue)**2
    e_per_event = cupy.matmul(real_events, g)
    to_return = cupy.sum((e_per_event - Etrue)**2)
    #print('%e'%to_return)
    return cupy.asnumpy(to_return)

# tstart = time.time()
# print(benchmark(to_minimize, (np.ones(npads),), n_repeat=1))
# print(time.time()-tstart)


init_guess = 1/10000

counter = 0
def display_progress(intermediate_result):
    global counter
    if (counter % 10)==0:
        print('step', counter)
        print(intermediate_result)
        changed_res = intermediate_result.x[intermediate_result.x != init_guess]
        print('mean, std:', np.mean(changed_res), np.std(changed_res))
        print(np.min(changed_res), np.max(changed_res))
    counter += 1


res = opt.minimize(to_minimize, init_guess*np.ones(npads), callback=display_progress)
np.save('padgain_noveto.npy',res.x)


# Current minimization: 4.544050e+16
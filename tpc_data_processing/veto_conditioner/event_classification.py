import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt

from raw_viewer import process_runs

e23035_run148 = np.array([
    1,0,0,0,0,0,0,0,0,0,
    1,0,0,0,1,0,1,0,0,0,
    1,0,0,0,1,1,0,0,0,0,
    1,0,1,0,0,1,0,1,0,0,
    0,0,1,0,1,0,1,0,0,0,
    0,1,0,0,1,1,0,1,1,0,
    0,1,1,0,0,0,0
    ])

outer_ring_max_counts = process_runs.get_outer_ring_max_counts('e23035', [148])[:len(e23035_run148)]
max_veto_counts = process_runs.get_max_veto_counts('e23035', [148])[:len(e23035_run148)]
track_width = process_runs.get_quantity('charge_width', 'e23035', [148])[:len(e23035_run148)]

plt.figure()
plt.title('should veto')
plt.scatter(max_veto_counts[e23035_run148==1], outer_ring_max_counts[e23035_run148==1], c=track_width[e23035_run148==1], marker='.')
plt.colorbar()
plt.xlabel('single veto pad max')
plt.ylabel('single pad outer ring max')

plt.figure()
plt.title('should not veto')
plt.scatter(max_veto_counts[e23035_run148==0], outer_ring_max_counts[e23035_run148==0], c=track_width[e23035_run148==0], marker='.')
plt.colorbar()
plt.xlabel('single veto pad max')
plt.ylabel('single pad outer ring max')

plt.figure()
plt.scatter(max_veto_counts, outer_ring_max_counts, c=e23035_run148==0, marker='.')
plt.colorbar()
plt.xlabel('single veto pad max')
plt.ylabel('single pad outer ring max')

outer_ring_total = process_runs.get_outer_ring_counts('e23035', [148])[:len(e23035_run148)]
veto_total = process_runs.get_veto_counts('e23035', [148])[:len(e23035_run148)]
plt.figure()
plt.title('should veto')
plt.scatter(veto_total[e23035_run148==1], outer_ring_total[e23035_run148==1], c=track_width[e23035_run148==1], marker='.')
plt.colorbar()
plt.xlabel('total veto pad charge')
plt.ylabel('total outer ring pad charge')

plt.figure()
plt.title('should not veto')
plt.scatter(veto_total[e23035_run148==0], outer_ring_total[e23035_run148==0], c=track_width[e23035_run148==0], marker='.')
plt.colorbar()
plt.xlabel('total veto pad charge')
plt.ylabel('total outer ring pad charge')

plt.figure()
plt.scatter(veto_total, outer_ring_total, c=e23035_run148==0, marker='.')
plt.colorbar()
plt.xlabel('total veto pad charge')
plt.ylabel('total outer ring pad charge')

plt.show(block=False)

from raw_viewer import raw_h5_file
run = 148
h5file = process_runs.get_h5_file('e23035', run)
h5file.cache_enable = False

fft_mask = np.zeros(200)
fft_mask[8:199]=1

def show_veto(evt):
    old_mode = h5file.background_subtract_mode
    h5file.background_subtract_mode = 'none'
    plt.figure('evt %d traces'%evt)
    pad_traces = {pad:trace for pad, trace in zip(*h5file.get_pad_traces(evt))}
    
    for i in range(8):
        ax = plt.subplot(2,4,i+1)
        #ax.plot(pad_traces[raw_h5_file.VETO_PADS[i]])
        ax.set_title(raw_h5_file.VETO_PADS[i])
        sp = np.fft.rfft(pad_traces[raw_h5_file.VETO_PADS[i]])
        fft_mask = np.zeros(200)
        fft_mask[1:200]=1
        ax.plot(np.fft.irfft(sp*fft_mask))
        fft_mask = np.zeros(200)
        fft_mask[1:8]=1
        ax.plot(np.fft.irfft(sp*fft_mask))
        # fft_mask = np.zeros(200)
        # fft_mask[12:]=1
        # ax.plot(np.fft.irfft(sp*fft_mask))
    plt.figure('fft evt %d'%evt)
    for i in range(8):
        ax = plt.subplot(2,4,i+1)
        sp = np.fft.rfft(pad_traces[raw_h5_file.VETO_PADS[i]])
        #freq = np.fft.fftfreq(len(pad_traces[raw_h5_file.VETO_PADS[i]]))
        ax.plot(np.abs(sp.real))
        ax.plot(np.abs(sp.imag))
        #_ = ax.plot(freq, np.abs(sp.real), freq, np.abs(sp.imag))
        ax.set_yscale('log')
        ax.set_title(raw_h5_file.VETO_PADS[i])

    h5file.background_subtract_mode = old_mode
    h5file.show_2d_projection(evt, block=False)

    plt.show(block=False)


def fit_gaussians(evt):
    old_mode = h5file.background_subtract_mode
    h5file.background_subtract_mode = 'none'
    pads, traces = h5file.get_pad_traces(evt)
    fit_params = []
    window_radius = 30
    obj_funcs = []
    for pad, trace in zip(pads, traces):
        peak_index = np.argmax(trace)
        trace_to_fit = trace[max(0, peak_index - window_radius): min(len(trace), peak_index + window_radius+1)]
        x = np.arange(len(trace_to_fit))
        fit_func = lambda A, mu, sigma, c: A*np.exp(-(x - mu)**2/2/sigma**2) + c
        error_func = lambda p: np.sum((trace_to_fit - fit_func(*p))**2)
        mu_guess = int(len(trace_to_fit/2))
        bounds = [(0, np.inf), (0, len(trace_to_fit)), (0, len(trace_to_fit)), (0,1000)]
        res = opt.minimize(error_func, (1,mu_guess,1,1), bounds=bounds)
        # plt.figure('tmp')
        # plt.clf()
        # plt.plot(trace_to_fit)
        # plt.plot(fit_func(*res.x))
        # print(res.x)
        # plt.show()
        fit_params.append(res.x)
        obj_funcs.append(res.fun)

    h5file.background_subtract_mode = old_mode
    return pads, np.array(fit_params), obj_funcs
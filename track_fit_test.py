import time
import numpy as np
import matplotlib.pylab as plt
import scipy.optimize as opt

from track_fitting import TraceFunction
from raw_viewer import raw_h5_file

run_h5_path = '/mnt/analysis/e21072/gastest_h5_files/run_0368.h5'
event_num = 5

init_position_guess = (-7, 10, 400)
E_guess = 6.2 #correct answer is 6.288
theta_guess = np.radians(80)
phi_guess = np.radians(120)

adc_scale = 376646/6.288 #counts/MeV, from fitting events with range 40-43 in run 0368 with p10_default

#use theoretical zscale
clock_freq = 50e6 #Hz
drift_speed = 54.4*1e6 #mm/s, from ruchi's paper
zscale = drift_speed/clock_freq

shaping_time = 117e-9 #s
shaping_width  = shaping_time*clock_freq

ic_threshold = 25


h5file = raw_h5_file.raw_h5_file(file_path=run_h5_path,
                                  zscale=zscale,
                                  flat_lookup_csv='raw_viewer/channel_mappings/flatlookup2cobos.csv')
h5file.background_subtract_mode='fixed window'
h5file.near_peak_window_width = 50
h5file.require_peak_within= (-np.inf, np.inf)
h5file.num_background_bins=(400,500)

gas_density = 1.56
trace_sim = TraceFunction.TraceFunction(1.56, 'alpha')
trace_sim.shaping_width = shaping_width
trace_sim.zscale = zscale
trace_sim.counts_per_MeV = adc_scale

trace_sim.initial_energy = E_guess
trace_sim.phi = phi_guess
trace_sim.theta = theta_guess
trace_sim.initial_point = init_position_guess

trace_sim.simulate_event()
pads_to_fit, traces_to_fit = h5file.get_pad_traces(event_num, include_veto_pads=False)
trace_sim.set_real_data(pads_to_fit, traces_to_fit, fit_threshold=ic_threshold)
trace_sim.align_pad_traces()
x_sim, y_sim, z_sim, e_sim = trace_sim.get_xyze('aligned', threshold = 100)
print(len(x_sim))

if False: #show plots of initial guess
    fig = plt.figure()
    plt.title('simulated data')
    ax = fig.add_subplot(111, projection='3d')
    # Plot the 3D scatter plot with energy values as color
    sc = ax.scatter(x_sim, y_sim, z_sim, c=e_sim,  alpha=0.5)#, cmap='inferno', marker='o',)# norm=matplotlib.colors.LogNorm())
    # Add colorbar
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label('Energy (adc units)')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.axes.set_xlim3d(left=-100, right=100) 
    ax.axes.set_ylim3d(bottom=-100, top=100) 
    ax.axes.set_zlim3d(bottom=0, top=200)

    def plot_traces(trace_dict, title=''):
        plt.figure()
        for pad in trace_dict:
            plt.plot(trace_dict[pad], label=str(pad))
        plt.legend()
        plt.title(title)

    plot_traces(trace_sim.traces_to_fit, 'clipped real traces')
    plot_traces(trace_sim.aligned_sim_traces, 'simulated traces')

    plt.show()

def neg_log_likelihood(params):
    E, x, y, z, theta, phi = params
    trace_sim.initial_energy = E
    trace_sim.initial_point = (x,y,z)
    trace_sim.theta = theta
    trace_sim.phi = phi
    trace_sim.simulate_event()
    trace_sim.align_pad_traces()
    to_return = -trace_sim.log_likelihood()
    print(params, to_return)
    return to_return

fit_start_time = time.time()
xopt = opt.fmin(func=neg_log_likelihood, x0=(E_guess, *init_position_guess, theta_guess, phi_guess))
print(xopt)
print('total fit time: %f s'%(time.time() - fit_start_time))
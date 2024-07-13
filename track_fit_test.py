import time

import os
os.environ['OPENBLAS_NUM_THREADS'] = '3'
import numpy as np

import matplotlib.pylab as plt
import scipy.optimize as opt

from track_fitting import TraceFunction
from raw_viewer import raw_h5_file

run_h5_path = '/mnt/analysis/e21072/gastest_h5_files/run_0368.h5'
event_num = 5

E_guess = 6.2 
if event_num == 5:
    init_position_guess = (-20, 12, 100)
    charge_spreading = 6
    theta_guess = np.radians(80)
    phi_guess = np.radians(-30)

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
h5file.data_select_mode='near peak'
h5file.remove_outliers=True
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
trace_sim.set_real_data(pads_to_fit, traces_to_fit, fit_threshold=ic_threshold, trim_pad = 20)
trace_sim.align_pad_traces()

def plot_traces(trace_dict, title=''):
        plt.figure()
        for pad in trace_dict:
            r = pad/1024*.8
            g = (pad%512)/512*.8
            b = (pad%256)/256*.8
            plt.plot(trace_dict[pad], label=str(pad), color=(r,g,b))
        plt.legend()
        plt.title(title)

def show_simulated_3d_data(mode,  threshold = 0.0001): #show plots of initial guess
    x_sim, y_sim, z_sim, e_sim = trace_sim.get_xyze(mode,threshold)
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


def neg_log_likelihood(params):
    E, x, y, z, theta, phi, charge_spread, counts_per_MeV = params
    counts_per_MeV *= 1e4
    trace_sim.initial_energy = E
    trace_sim.initial_point = (x,y,z)
    trace_sim.theta = theta
    trace_sim.phi = phi
    trace_sim.charge_spreading_sigma = charge_spread
    trace_sim.simulate_event()
    trace_sim.align_pad_traces()
    trace_sim.counts_per_MeV = counts_per_MeV
    to_return = -trace_sim.log_likelihood()
    print('E=%f MeV, (x,y,z)=(%f, %f, %f) mm, theta = %f deg, phi=%f deg, cs=%f mm, cpe=%e LL=%e'%(E, x,y,z,np.degrees(theta), np.degrees(phi), charge_spread, counts_per_MeV, to_return))
    return to_return

fit_start_time = time.time()

Ebounds = (5,9)
x_real, y_real, z_real, e_real = h5file.get_xyze(event_number=event_num)
x_bounds = (np.min(x_real), np.max(x_real))
y_bounds = (np.min(y_real), np.max(y_real))
z_bounds = (10,400)
theta_bounds = (0, np.radians(180))
phi_bounds = (0., 2*np.pi)
cs_bounds = (0,10)#mm
#opt_results = opt.shgo(func=neg_log_likelihood, bounds=[Ebounds, x_bounds, y_bounds, z_bounds, theta_bounds, phi_bounds, cs_bounds])
xopt = opt.fmin(func=neg_log_likelihood, x0=(E_guess, *init_position_guess, theta_guess, phi_guess, charge_spreading, trace_sim.counts_per_MeV/1e4), ftol=1000)
#res = opt.basinhopping(func=neg_log_likelihood, x0=(E_guess, *init_position_guess, theta_guess, phi_guess, 0))
#res = opt.minimize(fun=neg_log_likelihood, x0=(E_guess, *init_position_guess, theta_guess, phi_guess, 0), method = 'Nelder-Mead', options={'adaptive':True})
#res = opt.differential_evolution(func=neg_log_likelihood, bounds=[Ebounds, x_bounds, y_bounds, z_bounds, theta_bounds, phi_bounds, cs_bounds], workers=1)
#print(xopt)
print('total fit time: %f s'%(time.time() - fit_start_time))

plot_traces(trace_sim.traces_to_fit, 'clipped real traces')
plot_traces(trace_sim.aligned_sim_traces, 'simulated traces')

#best fit from differential evolution: E=6.034191 MeV, (x,y,z)=(26.532488, 16.171300, 39.979532) mm, theta = 149.786768deg, phi=79.963981 deg, cs=5.403833 mm, LL=6.807297e+06
#fmin w/ charge per MeV free:
#event 5: 
#event 7

trace_sim.simulate_event()
trace_sim.align_pad_traces()
show_simulated_3d_data(mode='aligned', threshold=100)
h5file.plot_3d_traces(event_num, threshold=100)

import time

import os
os.environ['OPENBLAS_NUM_THREADS'] = '3'
import numpy as np

import matplotlib.pylab as plt
import scipy.optimize as opt

from track_fitting import SingleParticleEvent
from raw_viewer import raw_h5_file

run_h5_path = '/mnt/analysis/e21072/gastest_h5_files/run_0368.h5'
event_num = 5

if event_num == 5:
    E_guess = 6.212
    init_position_guess = (-12, 13, 50)
    charge_spreading_guess = 3
    theta_guess = np.radians(90)
    phi_guess = np.radians(-30)
    P_guess = 1157

adc_scale = 376646/6.288 #counts/MeV, from fitting events with range 40-43 in run 0368 with p10_default

#use theoretical zscale
clock_freq = 50e6 #Hz
drift_speed = 54.4*1e6 #mm/s, from ruchi's paper
zscale = drift_speed/clock_freq

shaping_time = 117e-9 #s
shaping_width  = shaping_time*clock_freq*2.355

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

rho0 = 1.5256 #mg/cm^3, P10 at 300K and 760 torr
T = 20+273.15 #K
get_gas_density = lambda P: rho0*(P/760)*(300./T)
trace_sim = SingleParticleEvent.SingleParticleEvent(get_gas_density(P_guess), 'alpha')
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

def get_residuals():
    sim_trace_dict = trace_sim.aligned_sim_traces
    real_trace_dict = trace_sim.traces_to_fit
    residuals_dict = {}
    for pad in sim_trace_dict:
        if pad not in real_trace_dict:
            residuals_dict[pad] = sim_trace_dict[pad]
        else:
            residuals_dict[pad] = sim_trace_dict[pad] - real_trace_dict[pad]
    for pad in real_trace_dict:
        if pad not in sim_trace_dict:
            residuals_dict[pad] = -real_trace_dict[pad]
    return residuals_dict

def plot_residuals():
    plot_traces(get_residuals(), 'residuals')

def plot_residuals_3d(threshold=1):
    residuals_dict = get_residuals()
    xs, ys, es = [],[],[]
    for pad in residuals_dict:
        es.append(residuals_dict[pad])
        x,y = trace_sim.pad_to_xy[pad]
        xs.append(x)
        ys.append(y)
    num_z_bins = len(es[0])
    xs = np.repeat(xs, num_z_bins)
    ys = np.repeat(ys, num_z_bins)
    es = np.array(es).flatten()
    z_axis = np.arange(num_z_bins)*trace_sim.zscale
    zs = np.tile(z_axis, int(len(xs)/len(z_axis)))
    fig = plt.figure()
    plt.title('residuals')
    ax = fig.add_subplot(111, projection='3d')
    # Plot the 3D scatter plot with energy values as color
    xs = xs[np.abs(es)>threshold]
    ys = ys[np.abs(es)>threshold]
    zs = zs[np.abs(es)>threshold]
    es = es[np.abs(es)>threshold]
    sc = ax.scatter(xs, ys,zs, c=es,  alpha=0.5)#, cmap='inferno', marker='o',)# norm=matplotlib.colors.LogNorm())
    # Add colorbar
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label('Energy (adc units)')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.axes.set_xlim3d(left=-100, right=100) 
    ax.axes.set_ylim3d(bottom=-100, top=100) 
    ax.axes.set_zlim3d(bottom=0, top=200)



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
    E, x, y, z, theta, phi, charge_spread, shaping_width, P = params
    #P = P_guess
    trace_sim.load_srim_table('alpha', get_gas_density(P))
    trace_sim.initial_energy = E
    trace_sim.initial_point = (x,y,z)
    trace_sim.theta = theta
    trace_sim.phi = phi
    trace_sim.charge_spreading_sigma = charge_spread
    trace_sim.shaping_width = shaping_width
    trace_sim.simulate_event()
    trace_sim.align_pad_traces()
    to_return = -trace_sim.log_likelihood()
    print('E=%f MeV, (x,y,z)=(%f, %f, %f) mm, theta = %f deg, phi=%f deg, cs=%f mm, shaping=%f, P=%f torr,  LL=%e'%(E, x,y,z,np.degrees(theta), np.degrees(phi), charge_spread, shaping_width, P, to_return))
    return to_return


fit_start_time = time.time()

Ebounds = (5,7)
x_real, y_real, z_real, e_real = h5file.get_xyze(event_number=event_num)
x_bounds = (np.min(x_real), np.max(x_real))
y_bounds = (np.min(y_real), np.max(y_real))
z_bounds = (10,400)
theta_bounds = (0, np.radians(180))
phi_bounds = (0., 2*np.pi)
cs_bounds = (0,10)#mm
cpe_bounds = (0.8*adc_scale, 1.2*adc_scale)
initial_guess = (E_guess, *init_position_guess, theta_guess, phi_guess, charge_spreading_guess, shaping_width, P_guess)
#opt_results = opt.shgo(func=neg_log_likelihood, bounds=[Ebounds, x_bounds, y_bounds, z_bounds, theta_bounds, phi_bounds, cs_bounds, cpe_bounds])
#print(opt_results)
#xopt = opt.fmin_powell(func=neg_log_likelihood, x0=(E_guess, *init_position_guess, theta_guess, phi_guess, charge_spreading, trace_sim.shaping_width), ftol=1000)
res = opt.minimize(fun=neg_log_likelihood, x0=initial_guess, method="Powell")
#res = opt.basinhopping(func=neg_log_likelihood, x0=(E_guess, *init_position_guess, theta_guess, phi_guess, 0))
#res = opt.minimize(fun=neg_log_likelihood, x0=(E_guess, *init_position_guess, theta_guess, phi_guess, charge_spreading, trace_sim.shaping_width), options={'adaptive':True})
#res = opt.minimize(fun=neg_log_likelihood, method='BFGS', x0=(E_guess, *init_position_guess, theta_guess, phi_guess, charge_spreading, trace_sim.counts_per_MeV/1e4))
#res = opt.differential_evolution(func=neg_log_likelihood, bounds=[Ebounds, x_bounds, y_bounds, z_bounds, theta_bounds, phi_bounds, cs_bounds, cpe_bounds], workers=1)
#print(xopt)

print(res)
neg_log_likelihood(res.x)
print('total fit time: %f s'%(time.time() - fit_start_time))

plot_traces(trace_sim.traces_to_fit, 'clipped real traces')
plot_traces(trace_sim.aligned_sim_traces, 'simulated traces')
plot_residuals()
plot_residuals_3d()
#best fit from differential evolution: E=6.034191 MeV, (x,y,z)=(26.532488, 16.171300, 39.979532) mm, theta = 149.786768deg, phi=79.963981 deg, cs=5.403833 mm, LL=6.807297e+06
#typical for direct: E=6.408290 MeV, (x,y,z)=(-19.338566, 13.018618, 100.294812) mm, theta = 85.538278 deg, phi=-24.128679 deg, cs=6.449649 mm, cpe=5.947572e+04 LL=4.876868e+07

show_simulated_3d_data(mode='aligned', threshold=100)
h5file.plot_3d_traces(event_num, threshold=100)


#do MCMC
import emcee

def log_priors(params):
    
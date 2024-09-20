import emcee
import matplotlib.pylab as plt
import corner
import numpy as np
import sklearn.cluster as cluster
from track_fitting import ParticleAndPointDeposition
from raw_viewer import raw_h5_file

'''
olds
'''
if False:
    filename = '../run368_event5_samples.h5'
    labels = ['E', 'x','y','z','theta', 'phi', 'charge_spread', 'shaping_width', 'P', 'adc_scale']
    tau = [630, 1050]
if False:
    event=331
    filename = '../run368_event%d_samples_E_x_y_theta_phi.h5'%event
    labels = ['E', 'x','y','theta', 'phi']
    tau = [20,6]#reader.get_autocorr_time(

if False:
    filename = '../run368_event5_samples_E_x_y_theta_phi_15walker.h5'
    labels = ['E', 'x','y','theta', 'phi']
    tau = 200

if False:
    filename = '../run368_event5_samples_E_x_y_theta_phi_15walker.h5'
    labels = ['E', 'x','y','theta', 'phi']
    tau = 200

if False:
    run_number, event_number = 124, 4 #108
    filename = '../run%d_event%d.h5'%(run_number, event_number)
    labels = ['E', 'x','y','z','theta', 'phi', 'charge_spread',  'P', 'sigma']
    tau = [95,256]
    
if False:
    run_number, event_number = 124, 4
    filename = '../run%d_event%d_init_by_priors.h5'%(run_number, event_number)
    labels = ['E', 'x','y','z','theta', 'phi']
    tau = [2]
if False:
    run_number, event_number, beta = 124, 34, 1
    filename = '../run%d_mcmc/event%d/beta%f.h5'%(run_number, event_number, beta)
    labels = ['E', 'x','y','z','theta', 'phi']
    tau = [400,100]
if False:
    run_number, event_number = 124, 29
    filename = '../run%d_mcmc/event%d/after_clustering.h5'%(run_number, event_number)
    labels = ['E', 'x','y','z','theta', 'phi']
    tau = [100,400]
if True:
    run_number, event_number, beta = 124, 68192, 1
    #filename = '../run%d_palpha_mcmc/event%d/initial_run_beta%f.h5'%(run_number, event_number, beta)
    filename = './run%d_palpha_mcmc/event%d/cluster0.h5'%(run_number, event_number)
    labels = ['E', 'Ea_frac', 'x','y','z','theta', 'phi']
    tau = [100,400]

reader = emcee.backends.HDFBackend(filename=filename, read_only=True)


samples = reader.get_chain()
log_prob = reader.get_log_prob()

if False:
    #show time series
    fig, axes = plt.subplots(len(labels), figsize=(10, 7), sharex=True)#len(labels)
    for i in range(len(labels)):
        ax = axes[i]
        to_plot = samples[:, :, i]
        if labels[i] == 'theta' or labels[i] == 'phi':
            to_plot = np.degrees(to_plot)
        ax.plot(to_plot, "k", alpha=0.3)
        ax.set_xlim(0, len(samples))
        ax.set_ylabel(labels[i])
        ax.yaxis.set_label_coords(-0.1, 0.5)
    axes[-1].set_xlabel("step number")

    #show plot of ll vs phi in last step
    thetas = samples[-1][:, -2]
    phis = samples[-1][:, -1]
    plt.figure()
    plt.title("before clustering")
    plt.scatter(np.degrees(thetas), np.degrees(phis), c=log_prob[-1])
    plt.colorbar(label="log prob")
    plt.xlabel('theta (deg)')
    plt.ylabel('phi (deg)')

#make plot with proton and alpha energies, instead of total and Ea_frac
Ea_Ep_samples = np.copy(samples)
Ea_Ep_samples[:,:,0] = samples[:,:,0]*samples[:,:,1]
Ea_Ep_samples[:,:,1] = samples[:,:,0]*(1-samples[:,:,1])
Ea_Ep_labels = ['Ea', 'Ep', 'x','y','z','theta', 'phi']
if False:
    fig, axes = plt.subplots(len(Ea_Ep_labels), figsize=(10, 7), sharex=True)#len(labels)
    for i in range(len(labels)):
        ax = axes[i]
        to_plot = Ea_Ep_samples[:, :, i]
        if labels[i] == 'theta' or labels[i] == 'phi':
            to_plot = np.degrees(to_plot)
        ax.plot(to_plot, "k", alpha=0.3)
        ax.set_xlim(0, len(Ea_Ep_samples))
        ax.set_ylabel(Ea_Ep_labels[i])
        ax.yaxis.set_label_coords(-0.1, 0.5)
    axes[-1].set_xlabel("step number")

    #scatter plot of Ea and Ep color coded by posterior
    Ea = Ea_Ep_samples[-1][:, 0]
    Ep = Ea_Ep_samples[-1][:, 1]
    plt.figure()
    plt.scatter(Ea, Ep, c=log_prob[-1])
    plt.colorbar(label="log prob")
    plt.xlabel('Ea')
    plt.ylabel('Ep')

    plt.show()

tau=reader.get_autocorr_time(tol=0)
print('autocorrelation times:', tau)

burnin = int(2 * np.max(tau))
thin = int(0.5 * np.min(tau))

#flat_samples = reader.get_chain(discard=0, thin=1, flat=True)
#corner.corner(flat_samples, labels=labels)
flat_samples = reader.get_chain(discard=burnin, thin=thin, flat=True)

ndim = len(labels)
for i in range(ndim):
    mcmc = np.percentile(flat_samples[:, i], [16, 50, 84])
    if labels[i] == 'theta' or labels[i] == 'phi':
        mcmc = np.degrees(mcmc)
    q = np.diff(mcmc)
    txt = "\mathrm{{{3}}} = {0:.3f}_{{-{1:.3f}}}^{{{2:.3f}}}"
    txt = txt.format(mcmc[1], q[0], q[1], labels[i])
    print(txt)

EaEp_flat = reader.get_chain(discard=burnin, thin=thin, flat=True)
EaEp_flat[:,0] = flat_samples[:,0]*flat_samples[:,1]
EaEp_flat[:,1] = flat_samples[:,0]*(1-flat_samples[:,1])
if False: #make corner plots
    corner.corner(flat_samples, labels=labels)
    plt.savefig('corner_plot.png')
    corner.corner(EaEp_flat, labels=Ea_Ep_labels)
    plt.savefig('corner_plot_EaEp.png')

ndim = len(labels)
for i in range(ndim):
    mcmc = np.percentile(EaEp_flat[:, i], [16, 50, 84])
    if Ea_Ep_labels[i] == 'theta' or Ea_Ep_labels[i] == 'phi':
        mcmc = np.degrees(mcmc)
    q = np.diff(mcmc)
    txt = "\mathrm{{{3}}} = {0:.3f}_{{-{1:.3f}}}^{{{2:.3f}}}"
    txt = txt.format(mcmc[1], q[0], q[1], Ea_Ep_labels[i])
    print(txt)


h5_folder = '../../shared/Run_Data/'
run_number = 124
run_h5_path = h5_folder +'run_%04d.h5'%run_number

clock_freq = 50e6 #Hz, from e21062 config file on mac minis
drift_speed = 54.4*1e6 #mm/s, from ruchi's paper

h5file = raw_h5_file.raw_h5_file(file_path=run_h5_path,
                                zscale=drift_speed/clock_freq,
                                flat_lookup_csv='raw_viewer/channel_mappings/flatlookup4cobos.csv')
h5file.background_subtract_mode='fixed window'
h5file.data_select_mode='near peak'
h5file.remove_outliers=True
h5file.near_peak_window_width = 50
h5file.require_peak_within= (-np.inf, np.inf)
h5file.num_background_bins=(160, 250)
h5file.ic_counts_threshold = 25

def get_sim(params, grid_size=0.5, pad_gain_match_uncertainty=0.381959476, other_systematics=16.86638095):
    rho0 = 1.5256 #mg/cm^3, P10 at 300K and 760 torr
    T = 20+273.15 #K
    get_gas_density = lambda P: rho0*(P/760)*(300./T)
    sim=ParticleAndPointDeposition.ParticleAndPointDeposition(get_gas_density(800), 'proton')
    sim.initial_energy = params[1]
    sim.point_energy_deposition = params[0]
    sim.initial_point = params[2:5]
    sim.theta = params[5]
    sim.phi = params[6]
    sim.charge_spreading_sigma = 2
    sim.pad_gain_match_uncertainty = pad_gain_match_uncertainty
    sim.other_systematics = other_systematics
    sim.grid_resolution = grid_size
    #use theoretical zscale
    
    sim.zscale =  drift_speed/clock_freq
    shaping_time = 70e-9 #s, from e21062 config file on mac minis
    sim.shaping_width = shaping_time*clock_freq*2.355
    sim.counts_per_MeV = 86431./0.757
    #
    pads, traces = h5file.get_pad_traces(event_number, False)
    sim.set_real_data(pads, traces, 50, int(sim.shaping_width))
    sim.simulate_event()
    sim.align_pad_traces()
    return sim
   

#sim.plot_simulated_3d_data(mode='aligned', threshold=25)
#sim.plot_residuals_3d(energy_threshold=25)
#sim.plot_residuals()
#print(sim.log_likelihood())
#plt.show(block=False)
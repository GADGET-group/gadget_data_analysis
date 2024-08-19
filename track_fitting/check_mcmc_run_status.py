import emcee
import matplotlib.pylab as plt
import corner
import numpy as np

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

if True:
    run_number, event_number = 124, 4 #108
    filename = '../run%d_event%d.h5'%(run_number, event_number)
    labels = ['E', 'x','y','z','theta', 'phi', 'charge_spread',  'P', 'sigma']
    tau = [2, 2]
    

reader = emcee.backends.HDFBackend(filename=filename, read_only=True)


samples = reader.get_chain()

fig, axes = plt.subplots(len(labels), figsize=(10, 7), sharex=True)#len(labels)
for i in range(len(labels)):
    ax = axes[i]
    ax.plot(samples[:, :, i], "k", alpha=0.3)
    ax.set_xlim(0, len(samples))
    ax.set_ylabel(labels[i])
    ax.yaxis.set_label_coords(-0.1, 0.5)

axes[-1].set_xlabel("step number")

plt.show()

#tau=reader.get_autocorr_time()

burnin = int(2 * np.max(tau))
thin = int(0.5 * np.min(tau))

flat_samples = reader.get_chain(discard=0, thin=1, flat=True)
corner.corner(flat_samples, labels=labels)
flat_samples = reader.get_chain(discard=burnin, thin=thin, flat=True)
corner.corner(flat_samples, labels=labels)

ndim = len(labels)
for i in range(ndim):
    mcmc = np.percentile(flat_samples[:, i], [16, 50, 84])
    q = np.diff(mcmc)
    txt = "\mathrm{{{3}}} = {0:.3f}_{{-{1:.3f}}}^{{{2:.3f}}}"
    txt = txt.format(mcmc[1], q[0], q[1], labels[i])
    print(txt)

plt.show()


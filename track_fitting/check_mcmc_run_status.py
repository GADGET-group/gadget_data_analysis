import emcee
import matplotlib.pylab as plt
import corner
import numpy as np

if True:
    filename = '../run368_event5_samples.h5'
    labels = ['E', 'x','y','z','theta', 'phi', 'charge_spread', 'shaping_width', 'P', 'adc_scale']
else:
    filename = '../run368_event5_samples_E_x_y_theta_phi.h5'
    labels = ['E', 'x','y','theta', 'phi']

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


tau = 150#reader.get_autocorr_time()
burnin = int(2 * np.max(tau))
thin = int(0.5 * np.min(tau))

flat_samples = reader.get_chain(discard=burnin, thin=thin, flat=True)
corner.corner(flat_samples, labels=labels)

plt.show()
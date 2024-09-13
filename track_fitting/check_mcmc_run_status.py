import emcee
import matplotlib.pylab as plt
import corner
import numpy as np
import sklearn.cluster as cluster

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
    run_number, event_number = 124, 68129
    #filename = '../run%d_palpha_mcmc/event%d/beta%f_no_tempering.h5'%(run_number, event_number, beta)
    filename = '../run%d_palpha_mcmc/event%d/initial_run.h5'%(run_number, event_number)
    labels = ['E', 'Ea_frac', 'x','y','z','theta', 'phi']
    tau = [2]#[35,200]

reader = emcee.backends.HDFBackend(filename=filename, read_only=True)


samples = reader.get_chain()
log_prob = reader.get_log_prob()

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

#cluster by direction vector, to avoid issues at phi=0/pi
#keep all clusters of size >10
zhat = np.cos(thetas)
xhat = np.sin(thetas)*np.cos(phis)
yhat = np.sin(thetas)*np.sin(phis)
cluster_obj = cluster.DBSCAN(0.1).fit(np.vstack((xhat, yhat, zhat)).T)
cluster_label, cluster_counts = np.unique(cluster_obj.labels_, return_counts=True)
clusters_to_keep = cluster_label[(cluster_label>=0) & (cluster_counts>10)]
to_keep = np.in1d(cluster_obj.labels_, clusters_to_keep)
plt.figure()
plt.title("clusters to fit")
plt.scatter(np.degrees(thetas)[to_keep], np.degrees(phis)[to_keep], c=cluster_obj.labels_[to_keep])
plt.colorbar(label="cluster id")
plt.xlabel('theta (deg)')
plt.ylabel('phi (deg)')

plt.show()

#tau=reader.get_autocorr_time()

burnin = int(2 * np.max(tau))
thin = int(0.5 * np.min(tau))

#flat_samples = reader.get_chain(discard=0, thin=1, flat=True)
#corner.corner(flat_samples, labels=labels)
flat_samples = reader.get_chain(discard=burnin, thin=thin, flat=True)
corner.corner(flat_samples, labels=labels)

ndim = len(labels)
for i in range(ndim):
    mcmc = np.percentile(flat_samples[:, i], [16, 50, 84])
    if labels[i] == 'theta' or labels[i] == 'phi':
        mcmc = np.degrees(mcmc)
    q = np.diff(mcmc)
    txt = "\mathrm{{{3}}} = {0:.3f}_{{-{1:.3f}}}^{{{2:.3f}}}"
    txt = txt.format(mcmc[1], q[0], q[1], labels[i])
    print(txt)

plt.savefig('test.png')


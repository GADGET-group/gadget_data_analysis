import os

import emcee
import matplotlib.pylab as plt
import corner
import numpy as np
import sklearn.cluster as cluster

#from track_fitting import ParticleAndPointDeposition
#from raw_viewer import raw_h5_file
def process_h5(filepath, labels, Ea_Ep_labels=None, summary_file=None):
    base_fname = os.path.splitext(filepath)[0]
    reader = emcee.backends.HDFBackend(filename=filepath, read_only=True)
    with open(base_fname+'.txt', 'w') as output_text_file:
        samples = reader.get_chain()
        log_prob = reader.get_log_prob()

        show_time_series_plots = True

        if show_time_series_plots:
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
            plt.savefig(base_fname+'_chain.png')

            #show plot of ll vs phi in last step
            thetas = samples[-1][:, theta_index]
            phis = samples[-1][:, phi_index]
            plt.figure()
            plt.title("before clustering")
            plt.scatter(np.degrees(thetas), np.degrees(phis), c=log_prob[-1])
            plt.colorbar(label="log prob")
            plt.xlabel('theta (deg)')
            plt.ylabel('phi (deg)')
            plt.savefig(base_fname+'_theta_phi_ll.png')
            


        #make plot with proton and alpha energies, instead of total and Ea_frac
        if show_time_series_plots and Ea_Ep_labels != None:
            Ea_Ep_samples = np.copy(samples)
            Ea_Ep_samples[:,:,0] = samples[:,:,0]*samples[:,:,1]
            Ea_Ep_samples[:,:,1] = samples[:,:,0]*(1-samples[:,:,1])
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
            plt.savefig(base_fname+'Ea_Ep_ll.png')

        #plt.show()

        tau_auto=reader.get_autocorr_time(tol=0)
        output_text_file.write('autocorrelation times: '+str(tau_auto)+'\n')

        if True not in np.isnan(tau_auto):
            tau = tau_auto
        else:
            tau = [2]
        burnin = int(2 * np.max(tau))
        thin = int(0.5 * np.min(tau))
        output_text_file.write('burnin: %f\n'%burnin)
        output_text_file.write('thin: %f\n'%thin)

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
            output_text_file.write('%s\n'%txt)
        if True: #make corner plots
            corner.corner(flat_samples, labels=labels)
            plt.savefig(base_fname+'_corner_plot.png')
            if Ea_Ep_labels != None:
                EaEp_flat = reader.get_chain(discard=burnin, thin=thin, flat=True)
                EaEp_flat[:,0] = flat_samples[:,0]*flat_samples[:,1]
                EaEp_flat[:,1] = flat_samples[:,0]*(1-flat_samples[:,1])
                corner.corner(EaEp_flat, labels=Ea_Ep_labels)
                plt.savefig(base_fname+'corner_plot_EaEp.png')


                ndim = len(labels)
                for i in range(ndim):
                    mcmc = np.percentile(EaEp_flat[:, i], [16, 50, 84])
                    if Ea_Ep_labels[i] == 'theta' or Ea_Ep_labels[i] == 'phi':
                        mcmc = np.degrees(mcmc)
                    q = np.diff(mcmc)
                    txt = "\mathrm{{{3}}} = {0:.3f}_{{-{1:.3f}}}^{{{2:.3f}}}"
                    txt = txt.format(mcmc[1], q[0], q[1], Ea_Ep_labels[i])
                    output_text_file.write('%s\n'%txt)
                    if summary_file != None:
                        summary_file.write('%f + %f - %f, '%(mcmc[1],mcmc[0], mcmc[2]))
                if summary_file != None:
                    summary_file.write('\n')

        plt.close('all') 



if False:
    run_number= 124
    steps = 5
    filenames = []
    for event in [17,29,34,43,91, 108]:
        for step in range(steps):
            filenames.append('../run%d_mcmc/event%d/clustering_run%d.h5'%(run_number, event, step))
        #filenames.append('../run%d_mcmc/event%d/final_run.h5'%(run_number, event))
    labels = ['E', 'x','y','z','theta', 'phi', 'sigma_xy', 'sigma_z']
    theta_index, phi_index = 4,5
    tau = [2]
    Ea_Ep_labels = None
else:
    run_number= 124
    steps = 1
    filenames = []
    for event in [126]:#[74443, 25304, 38909, 104723, 43833, 52010, 95644, 98220,87480, 19699, 51777, 68192, 68087, 10356, 21640, 96369, 21662, 26303, 50543, 27067]:
        for step in range(steps):
            filenames.append('../run%d_palpha_mcmc/event%d/clustering_run%d.h5'%(run_number, event, step))
        #filenames.append('../run%d_palpha_mcmc/event%d/final_run.h5'%(run_number, event))
    labels = ['E', 'Ea_frac', 'x','y','z','theta', 'phi', 'sigma_xy', 'sigma_z']
    theta_index, phi_index = 5,6
    tau = [2]
    Ea_Ep_labels = ['Ea', 'Ep', 'x','y','z','theta', 'phi', 'sigma_xy', 'sigma_z']
    summary_file_path = '../run%d_palpha_mcmc/summary.txt'%run_number

for filepath in filenames:
    with open(summary_file_path, 'w') as summary_file:
        summary_file.write(str(Ea_Ep_labels) + '\n')
        process_h5(filepath, labels, Ea_Ep_labels, summary_file)


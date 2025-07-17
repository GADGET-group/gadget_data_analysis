import os

import emcee

import matplotlib.pylab as plt
import corner
import numpy as np
import sklearn.cluster as cluster

from track_fitting import build_sim

def process_h5(mcmc_filepath, run, event, labels, Ea_Ep_labels=None, summary_file=None):
    base_fname = os.path.splitext(mcmc_filepath)[0]
    reader = emcee.backends.HDFBackend(filename=mcmc_filepath, read_only=True)
    with open(base_fname+'.txt', 'w') as output_text_file:
        energy_from_ic = build_sim.get_energy_from_ic('e24joe', run, event)
        energy_from_ic_uncertainty = build_sim.get_detector_E_sigma('e24joe', run, energy_from_ic)
        output_text_file.write('Energy from integrated charge = %f +/- %f MeV\n'%(energy_from_ic, energy_from_ic_uncertainty))
        summary_file.write('%f +/- %f,'%(energy_from_ic, energy_from_ic_uncertainty))

        samples = reader.get_chain()
        log_prob = reader.get_log_prob()

        show_time_series_plots = True

        if show_time_series_plots:
            #show time series
            fig, axes = plt.subplots(len(labels), figsize=(20, 20), sharex=True)#len(labels)
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
            thetas_2 = samples[-1][:, theta_index_2]
            phis_2 = samples[-1][:, phi_index_2]
            plt.figure()
            plt.title("before clustering")
            plt.scatter(np.degrees(thetas_2), np.degrees(phis_2), c=log_prob[-1])
            plt.colorbar(label="log prob")
            plt.xlabel('theta (deg)')
            plt.ylabel('phi (deg)')
            plt.savefig(base_fname+'_theta_phi_ll_2.png')
            


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
        plt.close('all') 


        tau_auto=reader.get_autocorr_time(tol=0)
        output_text_file.write('autocorrelation times: '+str(tau_auto)+'\n')

        if True not in np.isnan(tau_auto):
            tau = tau_auto
        else:
            tau = [2]
        burnin = int(3 * np.max(tau))
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
            txt = "{{{3}}} = {0:.3f}_{{-{1:.3f}}}^{{{2:.3f}}}"
            txt = txt.format(mcmc[1], q[0], q[1], labels[i])
            output_text_file.write('%s\n'%txt)
            if summary_file != None:
                summary_file.write('%f +%f/-%f, '%(mcmc[1], q[0], q[1]))
        if summary_file != None:
                summary_file.write('\n')
        
        # true_params = [12,0.5,0,0,20,0.785398,0.785398,2.35619,3.92699,2,3]
        # true_params = [6,0,0,20,0.785398,0.785398,2,3]   
        corner.corner(flat_samples, labels=labels)# , truths=true_params)
        plt.savefig(base_fname+'_corner_plot.png')
        plt.close('all') 
        if Ea_Ep_labels != None:
            EaEp_flat = reader.get_chain(discard=burnin, thin=thin, flat=True)
            EaEp_flat[:,0] = flat_samples[:,0]*flat_samples[:,1]
            EaEp_flat[:,1] = flat_samples[:,0]*(1-flat_samples[:,1])
            corner.corner(EaEp_flat, labels=Ea_Ep_labels)# , truths=true_params)
            plt.savefig(base_fname+'corner_plot_EaEp.png')


            ndim = len(labels)
            for i in range(ndim):
                mcmc = np.percentile(EaEp_flat[:, i], [16, 50, 84])
                if Ea_Ep_labels[i] == 'theta' or Ea_Ep_labels[i] == 'phi':
                    mcmc = np.degrees(mcmc)
                q = np.diff(mcmc)
                txt = "$\mathrm{{{3}}} = {0:.3f}_{{-{1:.3f}}}^{{{2:.3f}}}$"
                txt = txt.format(mcmc[1], q[0], q[1], Ea_Ep_labels[i])
                output_text_file.write('%s\n'%txt)
            

        plt.close('all') 



run_number= 124
steps = ['forward', 'backward']
steps = ['forward']
filenames = []
events = [1]
# labels = ['E', 'Ea_frac', 'x', 'y', 'z', 'x_1', 'y_1', 'z_1', 'theta_p', 'phi_p', 'theta_a', 'phi_a', 'sigma_p_xy', 'sigma_p_z', 'c']
labels = ['E', 'Ea_frac', 'x', 'y', 'z', 'theta_p', 'phi_p', 'theta_a', 'phi_a', 'sigma_p_xy', 'sigma_p_z']
theta_index, phi_index = 5,6
theta_index_2, phi_index_2 = 7,8
tau = [2]
# Ea_Ep_labels = ['Ea', 'Ep', 'x', 'y', 'z', 'x_1', 'y_1', 'z_1', 'theta_p', 'phi_p', 'theta_a', 'phi_a', 'sigma_p_xy', 'sigma_p_z', 'c']
Ea_Ep_labels = ['Ea', 'Ep', 'x', 'y', 'z', 'theta_p', 'phi_p', 'theta_a', 'phi_a', 'sigma_p_xy', 'sigma_p_z']
summary_file_path = './run%d_dalpha_sim_mcmc_init_walker_at_true_values/summary.txt'%run_number
filepath_template = './run%d_dalpha_sim_mcmc_init_walker_at_true_values/event%d/%s.h5'

if False: #change this to True for single particle fits
    run_number= 124
    steps = ['forward', 'backward']
    filenames = []
    events = [3]#[4, 15 ,17 , 19, 20, 29, 31, 34, 43, 45, 55, 65, 71, 91, 108]
        #filenames.append('../run%d_mcmc/event%d/final_run.h5'%(run_number, event))
    labels = ['E', 'x','y','z','theta', 'phi', 'sigma_xy', 'sigma_z']
    theta_index, phi_index = 4,5
    tau = [2]
    Ea_Ep_labels = None
    summary_file_path = './run%d_single_alpha_sim_mcmc/summary.txt'%run_number
    filepath_template = './run%d_single_alpha_sim_mcmc/event%d/%s.h5'
else:
    run_number= 124
    steps = ['forward', 'backward']
    steps = ['forward']
    filenames = []
    events = [90,1762,2061,7175,11400,14822,21693,22081,35094]
    labels = ['E', 'Ea_frac', 'x','y','z', 'xa','ya','za','theta_p', 'phi_p', 'theta_a', 'phi_a', 'sigma_p_xy', 'sigma_p_z', 'sigma_a_xy', 'sigma_a_z']
    # labels = ['E', 'Ea_frac', 'x','y','z','theta_p', 'phi_p', 'theta_a', 'phi_a', 'sigma_xy', 'sigma_z']
    # labels = ['E', 'x','y','z','theta_p', 'phi_p', 'sigma_xy', 'sigma_z']
    theta_index, phi_index = 6,7
    theta_index_2, phi_index_2 = 8,9
    tau = [2]
    Ea_Ep_labels = ['Ea', 'Ep', 'x','y','z', 'xa','ya','za','theta_p', 'phi_p', 'theta_a', 'phi_a', 'sigma_p_xy', 'sigma_p_z', 'sigma_a_xy', 'sigma_a_z']
    # Ea_Ep_labels = ['Ea', 'Ep', 'x','y','z','theta_p', 'phi_p', 'theta_a', 'phi_a', 'sigma_xy', 'sigma_z']
    # Ea_Ep_labels = ['Ea', 'x','y','z','theta_p', 'phi_p', 'sigma_xy', 'sigma_z']
    # summary_file_path = './run%d_dalpha_mcmc/summary.txt'%run_number
    # filepath_template = './run%d_dalpha_mcmc/event%d/%s.h5'
    summary_file_path = './run%d_dalpha_good_starting_values_mcmc/summary.txt'%run_number
    filepath_template = './run%d_dalpha_good_starting_values_mcmc/event%d/%s.h5'


with open(summary_file_path, 'w') as summary_file:
    summary_file.write('event, energy from IC, ')
    for label in labels:
        summary_file.write('%s, '%label)
    summary_file.write('\n')
    for event in events:
        for step in steps:
            filepath = filepath_template%(run_number, event, step)
            print('processing: %s'%filepath)
            summary_file.write('%s, '%filepath)
            process_h5(filepath, run_number, event, labels, Ea_Ep_labels, summary_file)


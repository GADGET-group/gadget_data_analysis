import os

import emcee
import matplotlib.pylab as plt
import corner
import numpy as np
import sklearn.cluster as cluster

from track_fitting import build_sim
experiment = 'e25058'

def transform_to_spherical(raw_samples):
    """
    Converts the 14-parameter Cartesian MCMC chain back to the 12-parameter spherical chain
    so plots and percentiles are easily readable. Works for both 3D (chain) and 2D (flat) arrays.
    """
    new_shape = list(raw_samples.shape)
    new_shape[-1] = 12
    samples = np.zeros(new_shape)
    
    # Copy E, Ea_frac, x, y, z
    samples[..., 0:5] = raw_samples[..., 0:5]
    
    # Convert Proton Cartesian (indices 5, 6, 7) back to theta, phi
    p_x, p_y, p_z = raw_samples[..., 5], raw_samples[..., 6], raw_samples[..., 7]
    r_p = np.sqrt(p_x**2 + p_y**2 + p_z**2)
    samples[..., 5] = np.arccos(p_z / r_p)      # theta_p
    samples[..., 6] = np.arctan2(p_y, p_x)      # phi_p
    
    # Convert Alpha Cartesian (indices 8, 9, 10) back to theta, phi
    a_x, a_y, a_z = raw_samples[..., 8], raw_samples[..., 9], raw_samples[..., 10]
    r_a = np.sqrt(a_x**2 + a_y**2 + a_z**2)
    samples[..., 7] = np.arccos(a_z / r_a)      # theta_a
    samples[..., 8] = np.arctan2(a_y, a_x)      # phi_a
    
    # Copy sigma_p_xy, sigma_p_z, rho/k (indices 11, 12, 13)
    samples[..., 9:12] = raw_samples[..., 11:14]
    
    return samples

def process_h5(mcmc_filepath, run, event, labels, Ea_Ep_labels=None, summary_file=None):
    base_fname = os.path.splitext(mcmc_filepath)[0]
    reader = emcee.backends.HDFBackend(filename=mcmc_filepath, read_only=True)
    with open(base_fname+'.txt', 'w') as output_text_file:
        energy_from_ic = build_sim.get_energy_from_ic(experiment, run, event)
        energy_from_ic_uncertainty = build_sim.get_detector_E_sigma(experiment, run, energy_from_ic)
        output_text_file.write('Energy from integrated charge = %f +/- %f MeV\n'%(energy_from_ic, energy_from_ic_uncertainty))
        summary_file.write('%f +/- %f,'%(energy_from_ic, energy_from_ic_uncertainty))

        raw_samples = reader.get_chain()
        # Transform 14D chain back to 12D if necessary (keeps it backwards compatible with old files)
        if raw_samples.shape[-1] == 14:
            samples = transform_to_spherical(raw_samples)
        else:
            samples = raw_samples

        log_prob = reader.get_log_prob()

        show_time_series_plots = True

        if show_time_series_plots:
            #show time series
            fig, axes = plt.subplots(len(labels), figsize=(20, 20), sharex=True)#len(labels)
            for i in range(len(labels)):
                ax = axes[i]
                to_plot = samples[:, :, i]
                # Fixed: Use 'in' so it catches 'theta_p' and 'theta_a'
                if 'theta' in labels[i] or 'phi' in labels[i]:
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
                # Fixed: Use 'in' so it catches 'theta_p' and 'theta_a'
                if 'theta' in labels[i] or 'phi' in labels[i]:
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

        plt.close('all') 

        # Note: autocorr time is calculated on the raw 14D chain, which is mathematically correct
        tau_auto = reader.get_autocorr_time(tol=0)
        output_text_file.write('autocorrelation times: '+str(tau_auto)+'\n')

        if True not in np.isnan(tau_auto):
            tau = tau_auto
        else:
            tau = [2]
        burnin = int(5 * np.max(tau))
        thin = int(np.max(tau))
        output_text_file.write('burnin: %f\n'%burnin)
        output_text_file.write('thin: %f\n'%thin)

        raw_flat_samples = reader.get_chain(discard=burnin, thin=thin, flat=True)
        # Transform flat chain as well
        if raw_flat_samples.shape[-1] == 14:
            flat_samples = transform_to_spherical(raw_flat_samples)
        else:
            flat_samples = raw_flat_samples

        ndim = len(labels)
        for i in range(ndim):
            mcmc = np.percentile(flat_samples[:, i], [16, 50, 84])
            if 'theta' in labels[i] or 'phi' in labels[i]:
                mcmc = np.degrees(mcmc)
            q = np.diff(mcmc)
            txt = "{{{3}}} = {0:.3f}_{{-{1:.3f}}}^{{{2:.3f}}}"
            txt = txt.format(mcmc[1], q[0], q[1], labels[i])
            output_text_file.write('%s\n'%txt)
            if summary_file != None:
                summary_file.write('%f +%f/-%f, '%(mcmc[1], q[0], q[1]))
        if summary_file != None:
                summary_file.write('\n')
            
        corner.corner(flat_samples, labels=labels)
        plt.savefig(base_fname+'_corner_plot.png')
        plt.close('all') 
        
        if Ea_Ep_labels != None:
            EaEp_flat = np.copy(flat_samples)
            EaEp_flat[:,0] = flat_samples[:,0]*flat_samples[:,1]
            EaEp_flat[:,1] = flat_samples[:,0]*(1-flat_samples[:,1])
            corner.corner(EaEp_flat, labels=Ea_Ep_labels)
            plt.savefig(base_fname+'corner_plot_EaEp.png')

            corner.corner(EaEp_flat[:, np.r_[:2,-1]], labels=['Ea', 'Ep', 'k'])
            plt.savefig(base_fname+'corner_plot_onlyEaEp.png')

            ndim = len(labels)
            for i in range(ndim):
                mcmc = np.percentile(EaEp_flat[:, i], [16, 50, 84])
                if 'theta' in Ea_Ep_labels[i] or 'phi' in Ea_Ep_labels[i]:
                    mcmc = np.degrees(mcmc)
                q = np.diff(mcmc)
                txt = r"\mathrm{{{3}}} = {0:.3f}_{{-{1:.3f}}}^{{{2:.3f}}}"
                txt = txt.format(mcmc[1], q[0], q[1], Ea_Ep_labels[i])
                output_text_file.write('%s\n'%txt)
            
        plt.close('all') 

if False: #change this to True for single particle fits
    run_number= 124
    steps = ['backward']#['forward', 'backward']
    filenames = []
    events = [1070]#[4, 15 ,17 , 19, 20, 29, 31, 34, 43, 45, 55, 65, 71, 91, 108]
    labels = ['E', 'x','y','z','theta', 'phi', 'sigma_xy', 'sigma_z', 'rho']
    theta_index, phi_index = 4,5
    tau = [2]
    Ea_Ep_labels = None
    summary_file_path = './run%d_mcmc/summary.txt'%run_number
    filepath_template = './run%d_mcmc/event%d/%s.h5'
else:
    run_number= 71
    steps = ['forward', 'backward']
    filenames = []
    events = [4007, 7074, 9174, 11379, 15302, 21224, 22222, 28950, 33414, 4434, 5866, 314,  993, 1723, 166, 563 ]
    
    # Note: Keep the labels exactly as they were (12 dimensions). The transform_to_spherical function 
    # handles the conversion from 14 back to 12 behind the scenes.
    labels = ['E', 'Ea_frac', 'x','y','z','theta_p', 'phi_p', 'theta_a', 'phi_a', 'sigma_p_xy', 'sigma_p_z', 'k']
    theta_index, phi_index = 5,6
    tau = [2]
    Ea_Ep_labels = ['Ea', 'Ep', 'x','y','z','theta_p', 'phi_p', 'theta_a', 'phi_a', 'sigma_p_xy', 'sigma_p_z', 'k']
    summary_file_path = '%s_mcmc/run%d_palpha_mcmc/summary.txt'%(experiment, run_number)
    filepath_template = '%s_mcmc/run%d_palpha_mcmc/event%d/%s.h5'

with open(summary_file_path, 'w') as summary_file:
    summary_file.write('event, energy from IC, ')
    for label in labels:
        summary_file.write('%s, '%label)
    summary_file.write('\n')
    for event in events:
        for step in steps:
            filepath = filepath_template%(experiment, run_number, event, step)
            print('processing: %s'%filepath)
            summary_file.write('%s, '%filepath)
            process_h5(filepath, run_number, event, labels, Ea_Ep_labels, summary_file)
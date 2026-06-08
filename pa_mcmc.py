import time
import os
os.environ["OMP_NUM_THREADS"] = "1"
import sys

import numpy as np
import matplotlib.pylab as plt
import scipy.optimize as opt
import sklearn.cluster as cluster
import multiprocessing

import emcee
import corner

from track_fitting import ParticleAndPointDeposition, build_sim
from raw_viewer import raw_h5_file

if __name__ == '__main__':
    run_number, event_num = sys.argv[1:]
    particle_type = 'proton'
    run_number = int(run_number)
    event_num = int(event_num)

    experiment = 'e25058'

    #MCMC priors
    class GaussianVar:
        def __init__(self, mu, sigma):
            self.mu, self.sigma  = mu, sigma

        def log_likelihood(self, val):
            return -np.log(np.sqrt(2*np.pi*self.sigma**2)) - (val - self.mu)**2/2/self.sigma**2

    E_from_ic = build_sim.get_energy_from_ic(experiment, run_number, event_num)
    E_from_ic_sigma = build_sim.get_detector_E_sigma(experiment, run_number, E_from_ic)
    E_prior = GaussianVar(E_from_ic, E_from_ic_sigma)
    rho0 = build_sim.get_gas_density(experiment, run_number)
    density_scale_prior = GaussianVar(1, 0.05)#TODO: decid on density range

    h5file = build_sim.get_rawh5_object(experiment, run_number)
    #set zmax to length of trimmed traces
    temp_sim = build_sim.create_multi_particle_decay(experiment, run_number, event_num, ['1H', '4He'], [1.,4.], '16O', 16. )
    zmax = temp_sim.num_trace_bins*temp_sim.zscale

    x_real, y_real, z_real, e_real = temp_sim.get_xyze(threshold=h5file.length_counts_threshold, traces=temp_sim.traces_to_fit)
    xmin, xmax = np.min(x_real), np.max(x_real)
    ymin, ymax = np.min(y_real), np.max(y_real)
    zmin = 0

    track_center, track_direction_vec = h5file.get_track_axis(event_num)
    track_direction_vec = track_direction_vec[0]

    def get_sim(params):
        # Unpack 14 parameters (swapped 4 angles for 6 Cartesian coords)
        E, Ea_frac, x, y, z, p_x, p_y, p_z, a_x, a_y, a_z, sigma_xy, sigma_z, rho_scale = params
        
        Ep = E*(1-Ea_frac)
        Ea = E*Ea_frac
        
        # Convert Cartesian to Spherical for the simulation
        r_p = np.sqrt(p_x**2 + p_y**2 + p_z**2)
        theta_p = np.arccos(p_z / r_p)
        phi_p = np.arctan2(p_y, p_x)
        
        r_a = np.sqrt(a_x**2 + a_y**2 + a_z**2)
        theta_a = np.arccos(a_z / r_a)
        phi_a = np.arctan2(a_y, a_x)

        trace_sim = build_sim.create_multi_particle_decay(experiment, run_number, event_num, ['1H', '4He'], [1.,4.], '16O', 16. )
        trace_sim.sims[0].initial_energy = Ep
        trace_sim.sims[1].initial_energy = Ea
        trace_sim.initial_point = (x,y,z)
        trace_sim.sigma_xy = sigma_xy
        trace_sim.sigma_z = sigma_z
        trace_sim.sims[0].theta = theta_p
        trace_sim.sims[0].phi = phi_p
        trace_sim.sims[1].theta = theta_a
        trace_sim.sims[1].phi = phi_a
        
        for sim in trace_sim.sims:
            sim.load_srim_table(sim.particle, 'P10', rho0*rho_scale)
        trace_sim.simulate_event()
        return trace_sim

    def log_likelihood(params, print_out=False):
        trace_sim = get_sim(params)
        to_return = trace_sim.log_likelihood()
        if print_out:
            print(params, to_return)
        return to_return

    def log_priors(params, direction):
        E, Ea_frac, x, y, z, p_x, p_y, p_z, a_x, a_y, a_z, sigma_p_xy, sigma_p_z, rho_scale = params
        
        if E > 1.5*E_prior.mu:
            return -np.inf
        if Ea_frac < 0 or Ea_frac > 1:
            return -np.inf
        if x**2 + y**2 > 40**2:
            return -np.inf
        if z < zmin or z > zmax:
            return -np.inf
        if sigma_p_xy < 0 or sigma_p_xy > 30:
            return -np.inf
        if sigma_p_z < 0 or sigma_p_z > 30:
            return -np.inf

        # Normalize proton direction to evaluate bimodal constraint
        r_p = np.sqrt(p_x**2 + p_y**2 + p_z**2)
        vhat_p = np.array([p_x, p_y, p_z]) / r_p
        
        # Keep bimodal sampling restricted to the correct hemisphere
        if np.dot(vhat_p, direction*track_direction_vec) < 0:
            return -np.inf
            
        # Standard normal prior for 3D Cartesian vectors ensures uniform spherical sampling
        cartesian_prior = -0.5 * (p_x**2 + p_y**2 + p_z**2 + a_x**2 + a_y**2 + a_z**2)

        return E_prior.log_likelihood(E) + density_scale_prior.log_likelihood(rho_scale) + cartesian_prior

    def log_posterior(params, direction, print_out=False):
        to_return = log_priors(params, direction)
        if to_return != -np.inf:
            to_return +=  log_likelihood(params)
        if np.isnan(to_return):
            to_return = -np.inf
        if print_out:
            print('log posterior: %e'%to_return, params)
        return to_return

    # --- Top-level NLL and Wrapper for Pickling, only used for initial optimization of posterior---
    def nll(params, direction):
        lp = log_posterior(params, direction)
        if not np.isfinite(lp) or params[-1]<0.9 or params[-1]>1.1:
            return 1e15  # Massive penalty for stepping out of bounds
        return -lp

    def nll_pool_wrapper(args_tuple):
        # pool.map only passes one argument, so we unpack the tuple here
        params, dir_arg = args_tuple
        return nll(params, dir_arg)

    def nll_pool_wrapper_scaled(args_tuple):
        scaled_params, dir_arg, p_scales = args_tuple
        physical_params = scaled_params * p_scales
        return nll(physical_params, dir_arg)
    # ---------------------------------------------------
    # ---------------------------------------------------

    fit_start_time = time.time()

    fit_start_time = time.time()
    nwalkers = 256
    steps = 200
    ndim = 14 # Increased from 12 to 14

    def get_init_walker_pos(direction):
        # Direction vector for proton initialization
        vhat = track_direction_vec * direction

        # --- HEURISTIC INITIAL GUESS ---
        coords = np.column_stack((x_real, y_real, z_real))
        dists_along_track = np.dot(coords - track_center, vhat)
        extreme_end_point = coords[np.argmin(dists_along_track)]
        vertex_guess = extreme_end_point + vhat * 2.0 

        sigma_guess = 2.5
        max_veto_pad_counts, dxy, dz, measured_counts, angle, pads_railed = h5file.process_event(event_num)
        track_length = np.sqrt(dxy**2 + dz**2)
        Ep_guess = temp_sim.sims[0].srim_table.get_energy_w_stopping_distance(track_length - sigma_guess) 
        Ea_frac_guess = 1 - Ep_guess / E_prior.mu
        if Ea_frac_guess < 0:
            Ea_frac_guess = 0.001

        # Package the heuristic guess into a single 1D array (p0) for the optimizer
        # Alpha vector is guessed as a random unit vector to avoid div-by-zero
        a_guess = np.random.randn(3)
        a_guess /= np.linalg.norm(a_guess)

        p0 = np.array([
            E_prior.mu,                 # E
            Ea_frac_guess,              # Ea_frac
            vertex_guess[0],            # x
            vertex_guess[1],            # y
            vertex_guess[2],            # z
            vhat[0],                    # p_x
            vhat[1],                    # p_y
            vhat[2],                    # p_z
            a_guess[0],                 # a_x
            a_guess[1],                 # a_y
            a_guess[2],                 # a_z
            sigma_guess,                # sigma_p_xy
            15.0,                       # sigma_p_z
            1.0                         # rho_scale
        ])

        print(f"\n--- Optimizing Initial Position for Direction {direction} ---")

        # --- PRECONDITIONING FOR BFGS ---
        # Generate scale factors based on the initial guess. 
        # Using np.maximum prevents division by values that are exactly zero.
        param_scales = np.array([
            1,                 # E
            0.5,              # Ea_frac
            20,            # x
            20,            # y
            20,            # z
            1,                    # p_x
            1,                    # p_y
            1,                    # p_z
            1,                 # a_x
            1,                 # a_y
            1,                 # a_z
            3,                # sigma_p_xy
            3,                       # sigma_p_z
            0.1                         # rho_scale
        ])
        # Force the Energy scale strictly to its prior mean to handle the extreme magnitude
        param_scales[0] = E_prior.mu 
        
        # Initial guess in scaled space O(1)
        p0_scaled = p0 / param_scales

        def nll_scaled(scaled_params, direction_arg):
            physical_params = scaled_params * param_scales
            return nll(physical_params, direction_arg)

        # --- PARALLEL NUMERICAL GRADIENT (SCALED, CENTRAL DIFFERENCE) ---
        def parallel_jac_scaled(scaled_params, direction_arg):
            epsilon = 1e-5  # Increased epsilon to prevent floating-point truncation
            perturbed_params = []
            
            # Create (x + h) and (x - h) for all 14 parameters
            for i in range(ndim):
                # Forward step
                p_plus = np.copy(scaled_params)
                p_plus[i] += epsilon
                perturbed_params.append((p_plus, direction_arg, param_scales))
                
                # Backward step
                p_minus = np.copy(scaled_params)
                p_minus[i] -= epsilon
                perturbed_params.append((p_minus, direction_arg, param_scales))
                
            # Evaluate all 28 perturbed states simultaneously (28 < 256 cores)
            results = np.array(pool.map(nll_pool_wrapper_scaled, perturbed_params))
            
            # Split the results back into plus and minus arrays
            f_x_plus_h = results[0::2]   # Evens: 0, 2, 4...
            f_x_minus_h = results[1::2]  # Odds:  1, 3, 5...
            
            # Calculate Central Difference Gradient
            return (f_x_plus_h - f_x_minus_h) / (2 * epsilon)

        # --- CALLBACK FUNCTION WITH UN-SCALING ---
        iteration_counter = [0]
        last_lp = [-np.inf] 

        def print_callback_scaled(intermediate_result):
            iteration_counter[0] += 1
            current_lp = -intermediate_result.fun 
            
            # Print the PHYSICAL parameters, not the scaled ones
            xk_phys = intermediate_result.x * param_scales
            
            delta_lp = current_lp - last_lp[0]
            last_lp[0] = current_lp
            
            params_str = np.array2string(xk_phys, precision=3, suppress_small=True, max_line_width=120)
            print(f"  Step {iteration_counter[0]:02d} | Log-Posterior: {current_lp:+.4e} | Delta: {delta_lp:+.4e} | Params: {params_str}")
            
            if iteration_counter[0] > 10 and abs(delta_lp) < 0.1:
                print("  -> Stopping early: Log-posterior change is below 0.1 tolerance.")
                raise StopIteration
        # ------------------------------------------------

        # Run BFGS optimization in the SCALED space
        opt_start = time.time()
        res = opt.minimize(nll_scaled, p0_scaled, args=(direction,), method='BFGS', 
                        jac=parallel_jac_scaled, callback=print_callback_scaled,
                        options={'disp': True})

        stopped_by_us = getattr(res, 'status', None) == 99
            

        if res.success or stopped_by_us:
            print(f"  -> Optimization successful. (Reason: {res.message})")
            # Convert the result back to physical units
            best_p = res.x * param_scales
        else:
            print(f"  -> Optimization failed. (Reason: {res.message}). Trying to continue with Nelder-Mead")
            
            # FIXED CALLBACK: Properly passing 'direction' to nll_scaled
            def callback_nm(x):
                # Calculate the current NLL using both required arguments
                current_lp = -nll_scaled(x, direction)
                delta_lp = current_lp - last_lp[0]
                last_lp[0] = current_lp
                
                # Un-scale the parameters so the printout makes physical sense
                xk_phys = x * param_scales
                params_str = np.array2string(xk_phys, precision=3, suppress_small=True, max_line_width=120)
                print(f"  NM Step | Log-Posterior: {current_lp:+.4e} | delta lp {delta} | Params: {params_str}")

            res2 = opt.minimize(
                nll_scaled, 
                res.x,                   # Start exactly where BFGS left off
                args=(direction,), 
                callback=callback_nm,
                method='Nelder-Mead', 
                options={
                    'disp': True,
                    'fatol': 0.1,        # Exit criteria: Function (NLL) absolute tolerance is ~0.1
                    'xatol': 1e-6        # Exit criteria: Parameter step size is small
                }
            )
            
            print(f"  -> Nelder-Mead finished. (Reason: {res2.message})")
            best_p = res2.x * param_scales

        initial_positions = []
        
        # --- Hessian-based Initialization (WITH UN-SCALING) ---
        if (res.success or stopped_by_us) and hasattr(res, 'hess_inv'): #(res.success or stopped_by_us) and 
            try:
                cov_matrix_scaled = res.hess_inv
                
                # Transform inverse Hessian from scaled space to physical space:
                # Cov_phys = diag(Scales) * Cov_scaled * diag(Scales)
                cov_matrix = cov_matrix_scaled * np.outer(param_scales, param_scales)
                
                if np.all(np.diag(cov_matrix) > 0):
                    print("  -> Using un-scaled BFGS inverse Hessian to shape the initial walker ball.")
                    tight_cov = cov_matrix * 0.01 
                    initial_positions = np.random.multivariate_normal(best_p, tight_cov, size=nwalkers)
                    initial_positions = list(initial_positions) 
                else:
                    print("  -> Warning: Hessian diagonal has negative values. Falling back to heuristic.")
            except Exception as e:
                print(f"  -> Warning: Hessian initialization failed ({e}). Falling back to heuristic.")
        # -----------------------------------------

        # --- FALLBACK: Heuristic Scales ---
        if len(initial_positions) == 0:
            vp = np.sum(best_p[5:8]**2)**0.5
            va = np.sum(best_p[8:11]**2)**0.5
            print("  -> Using heuristic scales for walker initialization.")
            scales = np.array([
                E_prior.sigma * 0.05,  # E
                0.01,                 # Ea_frac
                0.1, 0.1, 0.1,        # x, y, z
                0.01*vp, 0.01*vp, 0.01*vp,     # p_x, p_y, p_z
                0.01*va, 0.01*va, 0.01*va,        # a_x, a_y, a_z
                0.1,                  # sigma_p_xy
                0.1,                  # sigma_p_z
                0.01                  # rho_scale
            ])
            initial_positions = [best_p + np.random.randn(ndim) * scales for w in range(nwalkers)]
        # ----------------------------------

        return initial_positions

    directory = '%s_mcmc/run%d_palpha_mcmc/event%d'%(experiment, run_number, event_num)
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    # --- NEW: Redirect all print statements to a log file ---
    log_file = os.path.join(directory, 'mcmc_output.log')
    # buffering=1 ensures the file writes line-by-line so you can view it live
    log_file_obj = open(log_file, 'w', buffering=1) 
    sys.stdout = log_file_obj
    sys.stderr = log_file_obj

    with multiprocessing.Pool() as pool:
        for direction in [1, -1]:
            if direction == 1:
                backend_fname = 'forward.h5'
            elif direction == -1:
                backend_fname = 'backward.h5'
            else:
                assert False

            init_walker_pos = get_init_walker_pos(direction)

            backend_file = os.path.join(directory, backend_fname)
            backend = emcee.backends.HDFBackend(backend_file)
            backend.reset(nwalkers, ndim)

            de_moves =[
                (emcee.moves.DEMove(), 0.8),
                (emcee.moves.DESnookerMove(), 0.2)
            ]
            stretch_kde = [
                (emcee.moves.StretchMove(), 0.75),
                (emcee.moves.KDEMove(), 0.25)
            ]

            kde_only = [
                (emcee.moves.KDEMove(), 1)
            ]

            stretch_move_only = [
                (emcee.moves.StretchMove(), 1)
            ]

            sampler = emcee.EnsembleSampler(nwalkers, ndim, log_posterior, args=[direction],backend=backend, 
                                            moves=stretch_move_only,
                                            pool=pool)

            for step_idx, sample in enumerate(sampler.sample(init_walker_pos, iterations=steps, progress=True)):
                print('step: ', step_idx)
                tau = sampler.get_autocorr_time(tol=0)
                xs = sampler.get_chain()[-1]
                Ea = xs[:, 0]*xs[:, 1]
                Ep = xs[:, 0]*(1-xs[:, 1])
                print('Ea = ', np.percentile(Ea, [0,16, 50, 84,100]))
                print('Ep = ', np.percentile(Ep, [0,16, 50, 84,100]))
                print(backend_fname, ', tau=', tau, ', accept fraction=', np.average(sampler.acceptance_fraction))
                lls = sampler.get_log_prob()[-1]
                
                print(np.percentile(xs, [50], axis=0))
                print('log prob:',np.percentile(lls, [0,16, 50, 84,100]))
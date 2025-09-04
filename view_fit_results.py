import pickle
import matplotlib.pyplot as plt
import numpy as np

from track_fitting.SimulatedEvent import SimulatedEvent
from track_fitting. MultiParticleEvent import MultiParticleEvent
import raw_viewer.raw_h5_file as raw_h5_file
from track_fitting import SingleParticleEvent, build_sim

evts, theta0,theta1, phi0,phi1, x0,y0,z0, x1,y1,z1, lls, cats, E0,E1, Erecs, nfev, sigma_xy, sigma_z = [], [],[], [],[], [],[],[], [],[],[], [], [], [],[], [], [], [],[]
trace_sims = []

file_path = '/egr/research-tpc/dopferjo/gadget_analysis/two_particle_decays_in_e24joe_test.dat' #TODO: double-check this is the right file name
file_path = '/egr/research-tpc/dopferjo/gadget_analysis/two_particle_decays_in_e24joe_best_direction_20_events.dat' 
with open(file_path, 'rb') as file:
    fit_results_dict = pickle.load(file)
    for key in fit_results_dict:
        print(key)
        res, sim = fit_results_dict[key]
        if res.fun == np.inf:
            print("Event %d exited with inf ll!"%key)
        # if res.success == True:
        print(res.message)
        # elif res.success == True:
        theta0.append(res.x[0] * np.pi)
        theta1.append(res.x[1] * np.pi)
        phi0.append(res.x[2] * 2 * np.pi)
        phi1.append(res.x[3] * 2 * np.pi)
        x0.append(res.x[4]* 40)
        y0.append(res.x[5] * 40)
        z0.append(res.x[6] * 400)
        x1.append(res.x[7]* 40)
        y1.append(res.x[8] * 40)
        z1.append(res.x[9] * 400)
        E0.append(res.x[10] * 10)
        E1.append(res.x[11] * 10)
        sigma_xy.append(res.x[12] * 10)
        sigma_z.append(res.x[13] * 10)
        lls.append(res.fun)
        evts.append(key)
        nfev.append(res.nfev)
    # thetas.append(res.x[0])
    # phis.append(res.x[1])
    # xs.append(res.x[2])
    # ys.append(res.x[3])
    # zs.append(res.x[4])
    # Es.append(res.x[5])
    # sigma_xys.append(res.x[6])
    # sigma_zs.append(res.x[7])
    # lls.append(res.fun)
    # cats.append(cat)
    # evts.append(evt)
    # nfev.append(res.nfev)
        # print(key)
        # print(res.x)
        # print(theta0)
        # print(theta1)
        # print(phi0)
        # print(phi1)
        # print(x0)
        # print(y0)
        # print(z0)
        # print(x1)
        # print(y1)
        # print(z1)
        # print(E0)
        # print(E1)
        
        trace_sim = build_sim.create_multi_particle_event('e24joe', 124, key, ['4He','4He'])
        trace_sim.per_particle_params = ['initial_energy', 'theta', 'phi', 'sigma_xy', 'sigma_z', 'num_stopping_power_points','initial_point'] 
        trace_sim.shared_params = ['gas_density']
        trace_sim.sims[0].adaptive_stopping_power = False
        trace_sim.sims[1].adaptive_stopping_power = False
        trace_sim.sims[0].initial_energy = E0[-1]
        trace_sim.sims[1].initial_energy = E1[-1]
        trace_sim.sims[0].theta, trace_sim.sims[0].phi, trace_sim.sims[1].theta, trace_sim.sims[1].phi = theta0[-1], phi0[-1], theta1[-1], phi1[-1]

        trace_sim.sims[0].num_stopping_power_points = trace_sim.sims[0].get_num_stopping_points_for_energy(E0[-1])
        trace_sim.sims[1].num_stopping_power_points = trace_sim.sims[1].get_num_stopping_points_for_energy(E1[-1])
        trace_sim.sims[0].initial_point = (x0[-1],y0[-1],z0[-1])
        trace_sim.sims[1].initial_point = (x1[-1],y1[-1],z1[-1])
        trace_sim.sims[0].sigma_xy, trace_sim.sims[1].sigma_xy = sigma_xy[-1], sigma_xy[-1]
        trace_sim.sims[0].sigma_z, trace_sim.sims[1].sigma_z = sigma_z[-1], sigma_z[-1]
        print(trace_sim.sims[0].sigma_xy)
        print(trace_sim.sims[1].sigma_xy) 
        # trace_sim.plot_residuals()
        # trace_sim.plot_residuals_3d(threshold=25)
        # trace_sim.plot_simulated_3d_data(threshold=25)
        plt.show(block=True)
    

plt.plot(z1, color = 'red')
plt.plot(z0, color = 'blue')
plt.show()
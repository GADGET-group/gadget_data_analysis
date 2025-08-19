import numpy as np
import scipy
import random
import tkinter as tk
from tkinter import ttk
import time
import multiprocessing
import pickle

import matplotlib.pylab as plt
import scipy.optimize as opt
from matplotlib.colors import ListedColormap
import mpl_toolkits.mplot3d as m3d

from track_fitting.SimulatedEvent import SimulatedEvent
from track_fitting. MultiParticleEvent import MultiParticleEvent
import raw_viewer.raw_h5_file as raw_h5_file 
import h5py

import pandas as pd
import hough3d
from hough3d import hough3D
import pyransac3d as pyrsc # not used atm
from track_fitting import SingleParticleEvent, build_sim

def cluster_and_fit(data,points):
    '''
    Categorizes data into two clusters based on how close they are to the lines created by 2 sets of 2 chosen points.
    Creates a new line based on the clustering.
    Calculates the sum of square distances from each point in the cluster to the line.
    
    '''
    
    # Start all points in cluster 0
    cluster = np.zeros(len(data), dtype=int)
    
    # points_index = np.random.randint(len(data), size=4)
    # points = data[points_index]
    sum_of_squares = 0
    # Assign points to the closest line
    for i in range(len(data)):
        # Distance to line 0
        d0 = np.linalg.norm(np.cross(points[0] - points[1],points[1] - data[i])) / np.linalg.norm(points[1] - points[0])
        # d0 = np.cross(points[0] - points[1],points[1] - data[i]) / np.linalg.norm(points[1] - points[0])
        # Distance to line 1
        d1 = np.linalg.norm(np.cross(points[2] - points[3],points[3] - data[i])) / np.linalg.norm(points[3] - points[2])
        # d1 = np.cross(points[2] - points[3],points[3] - data[i]) / np.linalg.norm(points[3] - points[2])
        cluster[i] = 0 if d0 < d1 else 1
    # Fit each cluster to a new line via SVD
    for clust_id in [0,1]:
        pts = data[cluster == clust_id]
        if len(pts) < 2:
            continue

        mean = pts.mean(axis=0)
        uu, dd, vv = np.linalg.svd(pts - mean)
        direction = vv[0]
        # direction[clust_id] = vv[0]

        # Line for plotting
        t = np.linspace(-10, 10, 20)[:, np.newaxis]
        linepts = mean + t * direction

        if clust_id == 0:
            linepts0 = linepts
            points[0], points[1] = linepts[0], linepts[-1]
        else:
            linepts1 = linepts
            points[2], points[3] = linepts[0], linepts[-1]

    for i in range(len(data)):
        if cluster[i] == 0:
            # Distance to line 0
            d = np.linalg.norm(np.cross(points[0] - points[1],points[1] - data[i])) / np.linalg.norm(points[1] - points[0])
        elif cluster[i] == 1:
            # Distance to line 1
            d = np.linalg.norm(np.cross(points[2] - points[3],points[3] - data[i])) / np.linalg.norm(points[3] - points[2])
        else:
            print("Clustering not working as expected! Point is not assigned to either cluster.")
        # calculate the sum of squares of the distance of each point in each cluster to the line of best fit
        sum_of_squares += d*d

    lines_of_best_fit = np.array([linepts0,linepts1])

    return cluster, lines_of_best_fit, sum_of_squares

use_likelihood = False #if false, uses least squares
fit_adc_count_per_MeV = False #use known energy rather than fitting it as a free parameter, and instead fit adc_counts_per_MeV
fix_energy = False
load_previous_fit = False
processes = []
pickle_fname = "two_particle_decays_in_e24joe_energy_free.dat"

def fit_event(event, best_point, best_point_end, Eknown = 6.288, particle_type = ['4He','4He'], direction = [0,0],
              return_key=None, return_dict=None, debug_plots=False): # currently only works with 2 particles
    trace_sim = build_sim.create_multi_particle_event('e24joe', 124, event, particle_type)
    particle0 = trace_sim.sims[0]
    particle1 = trace_sim.sims[1]
    trace_sim.per_particle_params = ['initial_energy', 'theta', 'phi', 'sigma_xy', 'sigma_z', 'num_stopping_power_points','initial_point'] 
    trace_sim.shared_params = ['gas_density']
    # print(trace_sim.__dict__)
    # in order for the initial guesses fed from the clustering script to match the 
    #trace_sim.counts_per_MeV *= 1.058
    m_guess, c_guess = 0.0, 22.5
    trace_sim.pad_gain_match_uncertainty = m_guess
    trace_sim.other_systematics = c_guess
    zmin = 0
    #set zmax to length of trimmed traces
    zmax = trace_sim.num_trace_bins*trace_sim.zscale
    
    track_direction_vec = np.array([])
    theta_guess = np.array([])
    phi_guess = np.array([])
    sigma_guess = np.array([])
    
    for i in range(len(particle_type)):
        #start theta, phi in a small ball around track direction from svd
        dx, dy, dz = best_point_end[i] - best_point[i]
        mag = np.sqrt(dx**2 + dy**2 + dz**2)
        track_direction_vec = np.append(track_direction_vec, (np.array([dx/mag, dy/mag, dz/mag])))
        theta_guess = np.append(theta_guess, (np.arccos(dz/mag)))
        phi_guess = np.append(phi_guess, (np.arctan2(dy,dx)))
        #start sigma_xy, sigma_z, and c in a small ball around an initial guess
        sigma_guess = np.append(sigma_guess, 2.5)
    
    init_guess = np.array((theta_guess[0], theta_guess[1], phi_guess[0], phi_guess[1], best_point[0][0], best_point[0][1], best_point[0][2], best_point[1][0], best_point[1][1], best_point[1][2], Eknown, Eknown, sigma_guess[0], sigma_guess[1]))
    print("Initial Guess: ", init_guess)
    
    def to_minimize(params, least_squares):
        theta0, theta1, phi0, phi1, x0, y0, z0, x1, y1, z1, E_or_m0, E_or_m1, sigma_xy0, sigma_z0 = params # note that each param is an array with length of the number of particles in the fit
        sigma_xy1, sigma_z1 = sigma_xy0, sigma_z0
        if fit_adc_count_per_MeV:
            trace_sim.counts_per_MeV = E_or_m0
            particle0.initial_energy = Eknown
            particle1.initial_energy = Eknown
        else:
            if fix_energy:
                particle0.initial_energy = Eknown
                particle1.initial_energy = Eknown
            else:
                particle0.initial_energy = E_or_m0
                particle1.initial_energy = E_or_m1
            trace_sim.counts_per_MeV = 129600. #using mean value fit when this was a free parameter
        

        #enforce particle direction
        # vhat0 = np.array([np.sin(theta0)*np.cos(phi0), np.sin(theta0)*np.sin(phi0), np.cos(theta0)])
        # vhat1 = np.array([np.sin(theta1)*np.cos(phi1), np.sin(theta1)*np.sin(phi1), np.cos(theta1)])
        # if np.dot(vhat0, direction[0]*np.array([track_direction_vec[0], track_direction_vec[1], track_direction_vec[2]])) < 0 \
        #     or np.dot(vhat1, direction[1]*np.array([track_direction_vec[3], track_direction_vec[4], track_direction_vec[5]])) < 0:
        #     print("returned inf because of particle direction: ", vhat0, vhat1)
        #     print("track_direction_vec: ", track_direction_vec)
        #     print("Dot product for clusters 0 & 1: ",  np.dot(vhat0, direction[0]*np.array([track_direction_vec[0], track_direction_vec[1], track_direction_vec[2]])), np.dot(vhat1, direction[1]*np.array([track_direction_vec[3], track_direction_vec[4], track_direction_vec[5]])))
        #     return np.inf

        if z0 > trace_sim.num_trace_bins*trace_sim.zscale or z0<-1 or z1 > trace_sim.num_trace_bins*trace_sim.zscale or z1<-1:
            print("returned inf because of z scale: ", z0, z1)
            print("Boundary z-values calculated via the trimmed trace: ", trace_sim.num_trace_bins*trace_sim.zscale)
            return np.inf
        if x0**2 + y0**2 > 40**2 or x1**2 + y1**2 > 40**2:
            print("returned inf because of xy plane boundary")
            return np.inf
        if particle0.initial_energy < 0 or particle0.initial_energy  > 10 or particle1.initial_energy < 0 or particle1.initial_energy  > 10: #stopping power tables currently only go to 10 MeV
            print("returned inf because of particle energy")
            return np.inf
        if theta0 < 0 or theta0 >= np.pi or phi0 < 0 or phi0 > 2*np.pi:
            return -np.inf 
        if theta1 < 0 or theta1 >= np.pi or phi1 < 0 or phi1 > 2*np.pi:
            return -np.inf        

        particle0.theta, particle0.phi, particle1.theta, particle1.phi = theta0, phi0, theta1, phi1
        # TODO:   File "/egr/research-tpc/dopferjo/gadget_analysis/da_ll_fit.py", line 166, in to_minimize
                # trace_sim[0].initial_point = (x0,y0,z0)
                # ~~~~~~~~~^^^
                # TypeError: 'MultiParticleEvent' object is not subscriptable
        trace_sim.sims[0].initial_point = (x0,y0,z0)
        trace_sim.sims[1].initial_point = (x1,y1,z1)
        trace_sim.sims[0].sigma_xy, trace_sim.sims[1].sigma_xy = sigma_xy0, sigma_xy1
        trace_sim.sims[0].sigma_z, trace_sim.sims[1].sigma_z = sigma_z0, sigma_z1
        
        trace_sim.simulate_event()
        if least_squares:
            residuals_dict = trace_sim.get_residuals()
            for pad in residuals_dict:
                #don't penalize for traces which don't go above threshold if pad didnt fire
                if pad not in trace_sim.traces_to_fit and np.max(trace_sim.sim_traces[pad] < trace_sim.pad_threshold): 
                    residuals_dict[pad] = residuals_dict[pad] - trace_sim.pad_threshold
                    residuals_dict[pad][residuals_dict[pad] < 0] = 0
            residuals = np.array([residuals_dict[p] for p in residuals_dict])
            to_return  = np.sum(residuals*residuals)
        else:
            to_return = -trace_sim.log_likelihood()
        #to_return = -trace_sim.log_likelihood()
        if debug_plots:
            print('%e'%to_return, params)
        if np.isnan(to_return):
            to_return = np.inf
        print('%e'%to_return, params)
        return to_return
    if debug_plots:
        print('guess:', init_guess)
        to_minimize(init_guess, True)
        trace_sim.plot_residuals()
        trace_sim.plot_residuals_3d(threshold=25)
        trace_sim.plot_simulated_3d_data(threshold=25)
        plt.show(block=True)
    print('starting optimization of event %d in run %d with particles %s in directions %d, %d'%(event, 124, particle_type, direction[0], direction[1]))
    res = opt.minimize(fun=to_minimize, x0=init_guess, args=(True,))
    if use_likelihood:
        res = opt.minimize(fun=to_minimize, x0=res.x, args=(False,))

    if return_dict != None:
        to_minimize(res.x, use_likelihood) #make sure sim is updated with best params
        return_dict[return_key] = (res, trace_sim)
        print(return_key, res)
        print('total completed in direction [%d,%d]:'%(direction[0],direction[1]), len(return_dict.keys()))
    if debug_plots:
        print(res)
        trace_sim.plot_residuals_3d(title=[str(return_key),particle_type], threshold=20)
        trace_sim.plot_simulated_3d_data(title=[str(return_key),particle_type], threshold=20)
        trace_sim.plot_residuals()
        plt.show()
    return res

np.random.seed(7) # Comment this out when you are finished debugging
random.seed(7)

event_type = 'RnPo Chain'  # Change to 'Accidental Coin' to look at random events
fit_type = 'kmeans'
zscale = 0.65

categorized_events_of_interest = pd.read_csv('./complete_categorized_events_of_interest.csv',\
    encoding='utf-8-sig', skip_blank_lines = False, nrows = 36164, header=None)

array_of_categorized_events_of_interest = categorized_events_of_interest[0].to_numpy()

h5file = raw_h5_file.raw_h5_file('/egr/research-tpc/dopferjo/interesting_events_without_run_number_in_event_name_without_event_447.h5', 
                                zscale = zscale, 
                                flat_lookup_csv = "/egr/research-tpc/dopferjo/gadget_analysis/raw_viewer/channel_mappings/flatlookup2cobos.csv")

f = h5py.File('/egr/research-tpc/dopferjo/interesting_events_without_run_number_in_event_name_without_event_447.h5', 'r')
first_event, last_event = int(f['meta']['meta'][0]), int(f['meta']['meta'][2])
print("First event: %d, Last event: %d"%(first_event, last_event))

# process settings
h5file.length_ic_threshold = 100
h5file.ic_counts_threshold = 9
h5file.view_threshold = 100
h5file.include_cobos = all
h5file.include_asads = all
h5file.include_pads = all
h5file.veto_threshold = 300
h5file.range_min = 1
h5file.range_max = np.inf
h5file.min_ic = 1
h5file.max_ic = np.inf
h5file.angle_min = 0
h5file.angle_max = 90
h5file.background_bin_start = 160
h5file.background_bin_stop = 250
h5file.zscale = zscale
h5file.background_start_entry = 160
h5file.background_stop_entry = 250
h5file.exclude_width_entry = 20
h5file.include_width_entry = 40
h5file.near_peak_window_entry = 50
h5file.near_peak_window_width = 50
h5file.peak_first_allowed_bin_entry = -np.inf
h5file.peak_last_allowed_bin_entry = np.inf
h5file.peak_first_allowed_bin = -np.inf
h5file.peak_last_allowed_bin = np.inf
h5file.peak_mode = 'all data'
h5file.background_subtract_mode = 'fixed window'
h5file.data_select_mode = 'all data'
h5file.remove_outliers = 1
h5file.num_background_bins = (450,500)

for event_number in range(len(array_of_categorized_events_of_interest)):
    if (array_of_categorized_events_of_interest[event_number] == 'RnPo Chain' or \
        array_of_categorized_events_of_interest[event_number] == 'Accidental Coin') and \
            event_number == 22:
    # if array_of_categorized_events_of_interest[event_number] == "Large Energy Single Event":
        x,y,z,e = h5file.get_xyze(event_number, threshold=1000, include_veto_pads=False) # a threshold of 140 is pretty good
        
        print("Number of Points in Event %d, (%s): "%(event_number,array_of_categorized_events_of_interest[event_number]),len(x))
        
        data = np.stack((x,y,z),axis=1)
        if fit_type == 'hough':
            # Determines how many tesselations of the icosahedron
            # to perform; larger number means more fine direction
            # discretization, but larger computation time.
            # See genIcosahedron docs for exact numbers.
            directionGranularity = 4
            directionVectors = hough3d.utils.genIcosahedron(directionGranularity)
            # use 3dhough to find the track axis for the 2 particles

            # hist_data = []
            # nDs = np.linspace(0,1,21)
            # mPPL = np.linspace(1,1300,21)
            # for i in range(len(nDs)):
            #     print(i)
            #     for j in range(len(mPPL)):
            #         linePoints = hough3D(data, directionVectors, latticeSize=128,
            #                             neighborDistance = nDs[i], minPointsPerLine = mPPL[j])
            #         for k in range(len(linePoints)):
            #             hist_data.append([nDs[i],mPPL[j]])
            # hist_data = np.array(hist_data)
            # plt.hist2d(hist_data.T[0], hist_data.T[1], bins = 20)
            # plt.show()
            bounds = np.zeros((2, 3))
            for i in range(3):
                bounds[:,i] = [np.min(data[:,i]), np.max(data[:,i])]
            
            systemLengthScale = np.sqrt(np.sum((bounds[1] - bounds[0])**2))
            print("systemLengthScale = ", systemLengthScale)
            
            linePoints = hough3D(data, directionVectors, latticeSize=128,
                                neighborDistance = 4*2.2/systemLengthScale, minPointsPerLine = 360)       
            print(np.shape(linePoints))
            fig = plt.figure()
            plt.clf()
            ax = plt.axes(projection='3d')
            for i in range(int(len(linePoints[0::]))):
                ax.plot(linePoints[i,:,0],linePoints[i,:,1],linePoints[i,:,2], label = 'Line %d'%i)
            ax.scatter(x,y,z, label='TPC Data for Event %d (%s)'%(event_number,event_type), alpha = 0.07, color = 'gray')
            ax.set_xlim3d(-200, 200)
            ax.set_ylim3d(-200, 200)
            ax.set_zlim3d(0, 400)
            ax.legend()
            plt.show()
        
        if fit_type == 'ransac': # not used as it has the possibility of getting stuck in both the bragg peaks of two alphas at 90 degrees from one another
            line = pyrsc.Line()
            A,B,inliers = line.fit(data,thresh=9)
        
        if fit_type =='kmeans':
            start_time = time.time()
                
            # Iterate the clusters and lines
            best_sos = np.inf
            # Use a combination of k-means++ and varying the random point to get the best clustering
            for j in range(20):

                # Pick a random initial point, then the point farthest from that, 
                # then the point farthest from those two, then the point farthest from those three
                ip0 = np.random.randint(len(data))
                points = [data[ip0]]
                dists = np.linalg.norm(data - points[0], axis=1)
                ip1 = np.argmax(dists)
                points.append(data[ip1])
                dists_to_set = np.min(
                    np.vstack([np.linalg.norm(data - p, axis=1) for p in points]),
                    axis=0
                )
                ip2 = np.argmax(dists_to_set)
                points.append(data[ip2])
                dists_to_set = np.min(
                    np.vstack([np.linalg.norm(data - p, axis=1) for p in points]),
                    axis=0
                )
                ip3 = np.argmax(dists_to_set)
                points.append(data[ip3])
                points = [data[ip0],data[ip3],data[ip1],data[ip2]]
                points = sorted(points, key=lambda points: points[2])

                # points_index = random.sample(range(0,len(data)), 4)
                # points = data[points_index]
                # print(points_index)
                
                cluster, lobf, sos = cluster_and_fit(data,points)
                if sos < best_sos:
                    best_sos = sos
                    best_cluster = cluster
                    best_lobf = lobf
                    print("Iteration: ",j)
                    print("Lowest sum of squares so far: ",sos)

                    # Plot results
                    fig = plt.figure()
                    ax = fig.add_subplot(projection='3d')
                    ax.scatter(*data[best_cluster == 0].T, color='teal', alpha=0.3, label="Cluster 0")
                    ax.scatter(*data[best_cluster == 1].T, color='pink', alpha=0.6, label="Cluster 1")
                    # ax.scatter(*best_lobf[0][0].T, s=100)
                    # ax.scatter(*best_lobf[0][-1].T, s=100)
                    # ax.scatter(*best_lobf[1][0].T, s=100)
                    # ax.scatter(*best_lobf[1][-1].T, s=100)
                    ax.plot(*best_lobf[0].T, color='blue', linewidth=2)
                    ax.plot(*best_lobf[1].T, color='red', linewidth=2)
                    ax.set_xlim3d(-200, 200)
                    ax.set_ylim3d(-200, 200)
                    ax.set_zlim3d(0, 400)
                    ax.legend()
                    plt.show()
                    
            print("Time to cluster with k-means: ", time.time() - start_time)

            # Once we extract the line along which each track travels, we use it to get the starting values for the fitter
            
            # scale the z value of our initial guess for use in the fitter with trimmed traces
            best_lobf[:,:,2] -= np.min(best_lobf[:,:,2])
            
            # Let's fit all combinations of forward and backward for the two clusters
            directions = [[1,1],[1,-1],[-1,1],[-1,-1]]
            for direction in directions:
                # fit_event(event_number, 
                #           [best_lobf[0][0], best_lobf[1][0]], 
                #           [best_lobf[0][-1], best_lobf[1][-1]],
                #           direction=direction)
            
                if not load_previous_fit:
                    # n = h5file.get_event_num_bounds()[0]
                    manager = multiprocessing.Manager()
                    forward_fit_results_dict = manager.dict()
                    backward_fit_results_dict = manager.dict()
                    # while np.min([len(x) for x in events_in_catagory]) < events_per_catagory and event_number < h5file.get_event_num_bounds()[1]: # collect a number of events from each category
                    max_veto_counts, dxy, dz, counts, angle, pads_railed = h5file.process_event(event_number) # process event to extract length and ADC counts
                    l = np.sqrt(dxy**2 + dz**2)
                    # event_catagory = classify(l, counts, experiment) # classify event based on processed event
                    # print('event %d classified as %d'%(n, event_catagory))
                    
                    # now fit the event forwards and backwards if it was not vetoed, was categorized properly, and we still need an event from that category
                    # if max_veto_counts < veto_threshold and event_catagory >= 0 and len(events_in_catagory[event_catagory]) < events_per_catagory: 
                    # particle_type, include_recoil, E = ptype_and_recoil_dict[event_catagory]
                    processes.append(multiprocessing.Process(target=fit_event, 
                                                                args=(event_number, 
                                                                [best_lobf[0][0], best_lobf[1][0]], 
                                                                [best_lobf[0][-1], best_lobf[1][-1]],
                                                                6.288,
                                                                ['4He','4He'],
                                                                direction,
                                                                event_number,
                                                                forward_fit_results_dict
                                                                )
                                                            )
                                     )
                    processes[-1].start()
                #wait for all processes to end
                for p in processes:
                    p.join()
                
                print('fitting took %f s'%(time.time() - start_time))

                #pick the best of each direction, and save it
                fit_results_dict = {k:forward_fit_results_dict[k] for k in forward_fit_results_dict}
                for k in backward_fit_results_dict:
                    if (k in fit_results_dict and backward_fit_results_dict[k][0].fun < fit_results_dict[k][0].fun) or k not in fit_results_dict: 
                        fit_results_dict[k] = backward_fit_results_dict[k]
                with open(pickle_fname, 'wb') as f:
                    pickle.dump(fit_results_dict, f)
          
            # initial_point_0 = 
            # initial_point_1 = 
            # energy_0 = 6.3
            # energy_1 = 6.3
            # theta_0 = 
            # theta_1 = 
            # phi_0 = 
            # phi_1 = 
            # sigma_xy = 2.0
            # sigma_z = 3.0
            
            
            # Hic Sunt Insectum
            # is_peak = np.zeros(len(data), dtype=int)
            # counter = 0
            # print(points)
            # for i in [0,1]:
            #     print(points[2*(i)])
            #     print(points[2*i+1])
            #     midpoint = (points[2*i] - points[2*i+1]) / 2
            #     for j in range(len(data[cluster == i])):
            #         if cluster[j] == i and np.sum(direction[i] * (data[cluster == i][j] - midpoint)) > 0: # assign the point 'peak'
            #             is_peak[counter] = 1
            #         elif cluster[j] == i: # assign it 'tail'
            #             is_peak[counter] = 0
            #         counter += 1
            #     print(np.shape(is_peak))
            #     print(np.shape(cluster))
                # now we determine which side is the true tail based on the summed charge in each region
            # Plot results
            # fig = plt.figure()
            # ax = fig.add_subplot(projection='3d')
            # ax.scatter(*data[np.logical_and.reduce((cluster == i, is_peak))].T, color='blue', alpha=0.6, label="Half 0")
            # ax.scatter(*data[np.logical_and.reduce((cluster == i, ~is_peak))].T, color='red', alpha=0.6, label="Half 1")
            # ax.plot(*linepts0.T, color='blue', linewidth=2)
            # ax.plot(*linepts1.T, color='red', linewidth=2)
            # ax.set_xlim3d(-200, 200)
            # ax.set_ylim3d(-200, 200)
            # ax.set_zlim3d(0, 400)
            # ax.legend()
            # plt.show() 
            
        # for x,y,z,e in zip(x,y,z,e):
        #     data[cluster == 0]    
        # initial_point_0 = 
        # initial_point_1 = 
        # energy_0 = 6.3
        # energy_1 = 6.3
        # theta_0 = 
        # theta_1 = 
        # phi_0 = 
        # phi_1 = 
        # sigma_xy = 2.0
        # sigma_z = 3.0
        # break

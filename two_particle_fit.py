import os
import re

load_previous_fit = False
if not load_previous_fit:
    os.environ["OMP_NUM_THREADS"] = "1"

import multiprocessing
# multiprocessing.set_start_method('spawn', force=True)

import numpy as np
import scipy
import random
import tkinter as tk
from tkinter import ttk
import time
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
# import pyransac3d as pyrsc # not used atm
from track_fitting import SingleParticleEvent, build_sim

gpus_to_use = [0,1,3]
np.seterr(over='raise')

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
        t = np.linspace(-15, 15, 20)[:, np.newaxis]
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
            assert False, "Clustering not working as expected! Point is not assigned to either cluster."
        # calculate the sum of squares of the distance of each point in each cluster to the line of best fit
        sum_of_squares += d*d

    lines_of_best_fit = np.array([linepts0,linepts1])

    return cluster, lines_of_best_fit, sum_of_squares

use_likelihood = True #if false, uses least squares
fit_adc_count_per_MeV = False #use known energy rather than fitting it as a free parameter, and instead fit adc_counts_per_MeV
fix_energy = False
processes = []
flast = 0
start_time = time.time()

def fit_event(event, best_point, best_point_end, Eknown = 6.288, particle_type = ['4He','4He'], direction = [1,1],
              return_key=None, return_dict=None, debug_plots=False, fit = True): # currently only works with 2 particles
    trace_sim = build_sim.create_multi_particle_event('e24joe', 124, event, particle_type)
    trace_sim.gpu_device_id = gpus_to_use[event%len(gpus_to_use)]
    # if event % 3 == 0:
    #     trace_sim.gpu_device_id = 0
    # else:
    #     trace_sim.gpu_device_id = 1
    trace_sim.per_particle_params = ['initial_energy', 'theta', 'phi', 'sigma_xy', 'sigma_z', 'num_stopping_power_points','initial_point'] 
    trace_sim.shared_params = ['gas_density']
    # in order for the initial guesses fed from the clustering script to match the 
    #trace_sim.counts_per_MeV *= 1.058
    m_guess, c_guess = 0.0, 22.5
    trace_sim.pad_gain_match_uncertainty = m_guess
    trace_sim.other_systematics = c_guess
    zmin = 0
    #set zmax to length of trimmed traces
    zmax = trace_sim.num_trace_bins*trace_sim.zscale
    
    trace_sim.sims[0].adaptive_stopping_power = False
    trace_sim.sims[1].adaptive_stopping_power = False
    trace_sim.sims[0].num_stopping_power_points = trace_sim.sims[0].get_num_stopping_points_for_energy(Eknown)
    trace_sim.sims[1].num_stopping_power_points = trace_sim.sims[1].get_num_stopping_points_for_energy(Eknown)
    
    track_direction_vec = np.array([])
    theta_guess = np.array([])
    phi_guess = np.array([])
    sigma_guess = np.array([])
    
    for i in range(len(particle_type)):
        dx, dy, dz = (best_point_end[i] - best_point[i])
        mag = np.sqrt(dx**2 + dy**2 + dz**2)
        track_direction_vec = np.append(track_direction_vec, (np.array([dx/mag, dy/mag, dz/mag])))
        # theta_guess = np.append(theta_guess, (np.arccos(dz/mag)))
        # phi_guess = np.append(phi_guess, (np.arctan2(dy,dx)))
        theta_guess = np.append(theta_guess,np.arctan2(np.sqrt(dx**2 + dy**2), dz))
        phi = np.arctan2(dy, dx)
        if phi < 0:
            phi += np.pi * 2 
        phi_guess = np.append(phi_guess,phi)
        #start sigma_xy, sigma_z, and c in a small ball around an initial guess
        sigma_guess = np.append(sigma_guess, 2.5)
    
    init_guess = np.array((theta_guess[0], theta_guess[1], phi_guess[0], phi_guess[1], best_point[0][0], best_point[0][1], best_point[0][2], best_point[1][0], best_point[1][1], best_point[1][2], Eknown, Eknown, sigma_guess[0], sigma_guess[1]))
    scaled_init_guess = np.zeros_like(init_guess)

    # condition the parameters so the fitter can move in the parameter space and converge faster (each parameter should approx go from -1 to 1)
    scaled_init_guess[0] = init_guess[0]  / np.pi
    scaled_init_guess[1] = init_guess[1]  / np.pi
    scaled_init_guess[2] = init_guess[2]  / (2* np.pi)
    scaled_init_guess[3] = init_guess[3]  / (2* np.pi)
    scaled_init_guess[4] = init_guess[4] / 40
    scaled_init_guess[5] = init_guess[5] / 40
    scaled_init_guess[6] = init_guess[6] / 400
    scaled_init_guess[7] = init_guess[7] / 40
    scaled_init_guess[8] = init_guess[8] / 40
    scaled_init_guess[9] = init_guess[9] / 400
    scaled_init_guess[10] = init_guess[10] / 10
    scaled_init_guess[11] = init_guess[11] / 10
    scaled_init_guess[12] = init_guess[12] / 10
    scaled_init_guess[13] = init_guess[13] / 10
    
    def to_minimize(params, least_squares):
        theta0, theta1, phi0, phi1, x0, y0, z0, x1, y1, z1, E_or_m0, E_or_m1, sigma_xy0, sigma_z0 = params # note that each param is an array with length of the number of particles in the fit
        # unscale the parameters
        theta0 = theta0 * np.pi
        theta1 = theta1 * np.pi
        phi0 = phi0 * 2 * np.pi
        phi1 = phi1 * 2 * np.pi
        x0 = x0 *40
        y0 = y0 *40
        z0 = z0 *400
        x1 = x1 *40
        y1 = y1 *40
        z1 = z1 *400
        E_or_m0 = E_or_m0 * 10
        E_or_m1 = E_or_m1 * 10
        sigma_xy0, sigma_z0 = sigma_xy0 * 10, sigma_z0 * 10
        # comment the above block out if you use the original parameters instead of the scaled parameters
        sigma_xy1, sigma_z1 = sigma_xy0, sigma_z0
        if fit_adc_count_per_MeV:
            trace_sim.counts_per_MeV = E_or_m0
            trace_sim.sims[0].initial_energy = Eknown
            trace_sim.sims[1].initial_energy = Eknown
        else:
            if fix_energy:
                trace_sim.sims[0].initial_energy = Eknown
                trace_sim.sims[1].initial_energy = Eknown
            else:
                trace_sim.sims[0].initial_energy = E_or_m0
                trace_sim.sims[1].initial_energy = E_or_m1
            trace_sim.counts_per_MeV = 77000. #using rough value from run 193 in e24joe      

        trace_sim.sims[0].theta, trace_sim.sims[0].phi, trace_sim.sims[1].theta, trace_sim.sims[1].phi = theta0, phi0, theta1, phi1
        
        trace_sim.sims[0].initial_point = (x0,y0,z0)
        trace_sim.sims[1].initial_point = (x1,y1,z1)
        trace_sim.sims[0].sigma_xy, trace_sim.sims[1].sigma_xy = sigma_xy0, sigma_xy1
        trace_sim.sims[0].sigma_z, trace_sim.sims[1].sigma_z = sigma_z0, sigma_z1
        trace_sim.enable_print_statements = False
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
            # trace_sim.log_likelihood_old()
        #to_return = -trace_sim.log_likelihood()
        if debug_plots:
            print('%e'%to_return, params)
        if np.isnan(to_return):
            to_return = np.inf
        return to_return
    
    # print("f(x0) =", to_minimize(scaled_init_guess,True))
    # for i in range(len(scaled_init_guess)):
    #     x_perturbed = scaled_init_guess.copy()
    #     x_perturbed[i] += 1e-3
    #     print(f"f(x0 + perturb in {i}) =", to_minimize(x_perturbed,True))
    
    if debug_plots:
        print('guess:', init_guess)
        print(to_minimize(scaled_init_guess, True))
        # build_sim.open_gui(trace_sim.sims[0])
        # build_sim.open_gui(trace_sim.sims[1])
        trace_sim.plot_residuals()
        trace_sim.plot_residuals_3d(threshold=25)
        trace_sim.plot_simulated_3d_data(threshold=25)
        plt.show(block=True)

    ftol = 0.01
    global flast
    flast = np.inf
    def callback(intermediate_result: opt.OptimizeResult):
        global flast
        print(intermediate_result.x, intermediate_result.fun)
        if np.abs(flast - intermediate_result.fun) < ftol:
            raise StopIteration
        flast = intermediate_result.fun
    if not use_likelihood and fit:
        print('Fitting event: direction, guess, least_squares:', direction, init_guess, to_minimize(scaled_init_guess, True))
        bnds = ((0,1),(0,1),(0,1),(0,1),(-1,1),(-1,1),(0,1),(-1,1),(-1,1),(0,1),(0.1,1),(0.1,1),(0,0.5),(0,0.5))
        res = opt.minimize(fun=to_minimize, x0=scaled_init_guess, args=(True,), callback=callback, bounds = bnds, options={'gtol': 1e-5,'ftol':1e-5})
    if use_likelihood and fit:
        print('Fitting event: direction, guess, likelihood:', direction, init_guess, to_minimize(scaled_init_guess, False))
        bnds = ((0,1),(0,1),(0,1),(0,1),(-1,1),(-1,1),(0,1),(-1,1),(-1,1),(0,1),(0.1,1),(0.1,1),(0,0.5),(0,0.5))
        res = opt.minimize(fun=to_minimize, x0=scaled_init_guess, args=(False,), callback=callback, bounds = bnds)#options={'gtol': 1e-5,'ftol':1e-5})
        print(res)
    if not fit:
        fit_results_dict[event] = init_guess
        # just return the residuals of the least squares fit for the initial guess params
        return to_minimize(scaled_init_guess,True)
    if return_dict != None:
        to_minimize(res.x, use_likelihood) #make sure sim is updated with best params
        # print("Direction: ",direction)
        return_dict[return_key] = (res, trace_sim)
        # Dump contents of fit into a pickle file to be retrieved later
        pickle_fname = "./fit_results/event_%05d_ll_fit_two_particle_decays_in_e24joe.dat"%return_key
        with open(pickle_fname, 'wb') as f:
            pickle.dump(return_dict[return_key], f)
        # print(return_key, res)
        # print('total completed in direction [%d,%d]:'%(direction[0],direction[1]), len(return_dict.keys()))
    if debug_plots:
        print(res)
        trace_sim.plot_residuals_3d(title=[str(return_key),particle_type], threshold=20)
        trace_sim.plot_simulated_3d_data(title=[str(return_key),particle_type], threshold=20)
        trace_sim.plot_residuals()
        plt.show()
    return res

event_type = 'RnPo Chain'  # Change to 'Accidental Coin' to look at random events
fit_type = 'kmeans'
zscale = 0.65

def process_two_particle_event(event_number):
    print("Processing Event %d"%event_number)
    # instead of using the raw xyze values, extract those values from the simulation so that the trimmed trace is consistent from the clustering to the fitter
    temp_sim = build_sim.create_multi_particle_event('e24joe', 124, event_number, ['4He','4He'])
    # zmax = temp_sim.num_trace_bins*temp_sim.zscale
    x, y, z, e = temp_sim.get_xyze(threshold=1500, traces=temp_sim.traces_to_fit)
    
    # print("Number of Points in Event %d, (%s): "%(event_number,array_of_categorized_events_of_interest[event_number]),len(x))
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

            
            cluster, lobf, sos = cluster_and_fit(data,points)
            if sos < best_sos:
                best_sos = sos
                best_cluster = cluster
                best_lobf = lobf

                # Plot results
                if False:
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
                
        # print("Time to cluster with k-means: ", time.time() - start_time)

        # Once we extract the line along which each track travels, we use it to get the starting values for the fitter
        
        # scale the z value of our initial guess for use in the fitter with trimmed traces (this is no longer needed if the new x,y,z,e values are consistent)
        # best_lobf[:,:,2] -= np.min(best_lobf[:,:,2])
        
        # Let's fit all combinations of forward and backward for the two clusters
        directions = [[1,1],[1,-1],[-1,1],[-1,-1]]
        # directions = [[1,-1]]
        cluster0_start = best_lobf[0][0]
        cluster0_end = best_lobf[0][-1]
        cluster1_start = best_lobf[1][0]
        cluster1_end = best_lobf[1][-1]
        for direction in directions:
            if not load_previous_fit:
                if direction == [1,1]:
                    direction1_residuals = fit_event(event_number, 
                                                     [cluster0_start, cluster1_start], 
                                                     [cluster0_end, cluster1_end],
                                                     6.288,
                                                     ['4He','4He'],
                                                     direction,
                                                     event_number,
                                                     None,
                                                     False,
                                                     fit=False)
                elif direction == [1,-1]:
                    direction2_residuals = fit_event(event_number, 
                                                     [cluster0_start, cluster1_end], 
                                                     [cluster0_end, cluster1_start],
                                                     6.288,
                                                     ['4He','4He'],
                                                     direction,
                                                     event_number,
                                                     None,
                                                     False,
                                                     fit=False)
                elif direction == [-1,1]:
                    direction3_residuals = fit_event(event_number, 
                                                     [cluster0_end, cluster1_start], 
                                                     [cluster0_start, cluster1_end],
                                                     6.288,
                                                     ['4He','4He'],
                                                     direction,
                                                     event_number,
                                                     None,
                                                     False,
                                                     fit=False)
                elif direction == [-1,-1]:
                    direction4_residuals = fit_event(event_number, 
                                                     [cluster0_end, cluster1_end], 
                                                     [cluster0_start, cluster1_start],
                                                     6.288,
                                                     ['4He','4He'],
                                                     direction,
                                                     event_number,
                                                     None,
                                                     False,
                                                     fit=False)
        print("Results of each direction's least squares residuals for the initial guess based on clustering: ",direction1_residuals, direction2_residuals, direction3_residuals, direction4_residuals)
        if direction1_residuals < direction2_residuals and direction1_residuals < direction3_residuals and direction1_residuals < direction4_residuals:
            print("Direction 1 is best!")
            fit_event(event_number, 
                    [cluster0_start, cluster1_start], 
                    [cluster0_end, cluster1_end],
                    6.288,
                    ['4He','4He'],
                    [1,1],
                    event_number,
                    ff_fit_results_dict,
                    False,
                    fit=True) 
            # fit_results_dict[event_number] = ff_fit_results_dict[event_number]
            return "Event %d finished fitting in Direction 1"%event_number
            
        if direction2_residuals < direction1_residuals and direction2_residuals < direction3_residuals and direction2_residuals < direction4_residuals:
            print("Direction 2 is best!")            
            fit_event(event_number,
                    [cluster0_start, cluster1_end], 
                    [cluster0_end, cluster1_start],
                    6.288,
                    ['4He','4He'],
                    [1,-1],
                    event_number,
                    fb_fit_results_dict,
                    False,
                    fit=True) 
            # fit_results_dict[event_number] = fb_fit_results_dict[event_number]
            return "Event %d finished fitting in Direction 2"%event_number

        if direction3_residuals < direction1_residuals and direction3_residuals < direction2_residuals and direction3_residuals < direction4_residuals:
            print("Direction 3 is best!")
            fit_event(event_number,
                    [cluster0_end, cluster1_start], 
                    [cluster0_start, cluster1_end],
                    6.288,
                    ['4He','4He'],
                    [-1,1],
                    event_number,
                    bf_fit_results_dict,
                    False,
                    fit=True) 
            # fit_results_dict[event_number] = bf_fit_results_dict[event_number]
            return "Event %d finished fitting in Direction 3"%event_number
            
        if direction4_residuals < direction1_residuals and direction4_residuals < direction2_residuals and direction4_residuals < direction3_residuals:
            print("Direction 4 is best!")
            fit_event(event_number, 
                        [cluster0_end, cluster1_end], 
                        [cluster0_start, cluster1_start],
                        6.288,
                        ['4He','4He'],
                        [-1,-1],
                        event_number,
                        bb_fit_results_dict,
                        False,
                        fit=True) 
            # fit_results_dict[event_number] = bb_fit_results_dict[event_number]
            return "Event %d finished fitting in Direction 4"%event_number
        return "Something went wrong"


categorized_events_of_interest = pd.read_csv('./complete_categorized_events_of_interest.csv',\
    encoding='utf-8-sig', skip_blank_lines = False, nrows = 36164, header=None)

array_of_categorized_events_of_interest = categorized_events_of_interest[0].to_numpy()

h5file = raw_h5_file.raw_h5_file('/egr/research-tpc/dopferjo/interesting_events_without_run_number_in_event_name_without_event_447.h5', 
                                zscale = zscale, 
                                flat_lookup_csv = "/egr/research-tpc/dopferjo/gadget_analysis/raw_viewer/channel_mappings/flatlookup2cobos.csv")

f = h5py.File('/egr/research-tpc/dopferjo/interesting_events_without_run_number_in_event_name_without_event_447.h5', 'r')
first_event, last_event = int(f['meta']['meta'][0]), int(f['meta']['meta'][2])

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

n_workers = 100
mask = np.isin(array_of_categorized_events_of_interest, ['RnPo Chain', 'Accidental Coin', 'Double Alpha Candidate'])
events = np.where(mask)[0]

# Get a list of events already fitted and remove them from the array of events to be fit
results_directory = os.fsencode("/egr/research-tpc/dopferjo/gadget_analysis/fit_results")
pattern = re.compile(r"event_(\d+)_ll_fit_two_particle_decays_in_e24joe.dat$")
completed_fit_events = []
for filename in os.listdir(results_directory):
    match = pattern.match(filename.decode('utf-8'))
    if match:
        event_num = int(match.group(1))
        completed_fit_events.append(event_num)
events = [item for item in events if item not in completed_fit_events]
# for i 
# events = [97]
# fit_results_dict = {}  # shared dictionary
# ff_fit_results_dict = {}
# fb_fit_results_dict = {}
# bf_fit_results_dict = {}
# bb_fit_results_dict = {}  
# process_two_particle_event(4)
if __name__ == "__main__":
    manager = multiprocessing.Manager()
    fit_results_dict = manager.dict()  # shared dictionary
    ff_fit_results_dict = manager.dict()
    fb_fit_results_dict = manager.dict()
    bf_fit_results_dict = manager.dict()
    bb_fit_results_dict = manager.dict()
    with multiprocessing.Pool(processes=n_workers) as pool:
        for result in pool.imap_unordered(process_two_particle_event,events):
            print("Done Fitting")

#wait for all processes to end
print('fitting took %f s'%(time.time() - start_time))
#pick the best of each direction, and save it
# fit_results_dict = {k:ff_fit_results_dict[k] for k in ff_fit_results_dict}
# yac = 0
# for k in events:
#     if (k in fit_results_dict and bb_fit_results_dict[k][0].fun < fit_results_dict[k][0].fun) or k not in fit_results_dict: 
#         fit_results_dict[k] = bb_fit_results_dict[k]
#     if (k in fit_results_dict and fb_fit_results_dict[k][0].fun < fit_results_dict[k][0].fun) or k not in fit_results_dict: 
#         fit_results_dict[k] = fb_fit_results_dict[k]
#     if (k in fit_results_dict and bf_fit_results_dict[k][0].fun < fit_results_dict[k][0].fun) or k not in fit_results_dict: 
#         fit_results_dict[k] = bf_fit_results_dict[k]
#     elif k not in fit_results_dict:
#         fit_results_dict[k] = 'Event not fitted'
#     yac += 1
# pickle_fname = "two_particle_decays_in_e24joe_energy_free_%d.dat"%process_counter
pickle_fname = "ll_fit_two_particle_decays_in_e24joe.dat"
# pickle_fname = "two_particle_decays_in_e24joe_no_fit.dat"
fit_results_dict = dict(fit_results_dict)
with open(pickle_fname, 'wb') as f:
    pickle.dump(fit_results_dict, f)
import numpy as np
import tkinter as tk
from tkinter import ttk
import time

import  numpy as np
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
import pyransac3d as pyrsc

start_time = time.time()

np.random.seed(7) # Comment this out when you are finished debugging

event_type = 'RnPo Chain'  # Change to 'Accidental Coin' to look at random events
fit_type = 'kmeans'
zscale = 0.65

categorized_events_of_interest = pd.read_csv('./complete_categorized_events_of_interest.csv', encoding='utf-8-sig', skip_blank_lines = False, nrows = 36164, header=None)
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

# Determines how many tesselations of the icosahedron
# to perform; larger number means more fine direction
# discretization, but larger computation time.
# See genIcosahedron docs for exact numbers.
directionGranularity = 4
directionVectors = hough3d.utils.genIcosahedron(directionGranularity)

for event_number in range(len(array_of_categorized_events_of_interest)):
    if (array_of_categorized_events_of_interest[event_number] == 'RnPo Chain' or array_of_categorized_events_of_interest[event_number] == 'Accidental Coin'):
    # if array_of_categorized_events_of_interest[event_number] == "Large Energy Single Event":
        x,y,z,e = h5file.get_xyze(event_number, threshold=140, include_veto_pads=False)
        
        print("Number of Points in Event %d, (%s): "%(event_number,array_of_categorized_events_of_interest[event_number]),len(x))
        
        data = np.stack((x,y,z),axis=1)
        if fit_type == 'hough':
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
        
        if fit_type == 'ransac':
            line = pyrsc.Line()
            A,B,inliers = line.fit(data,thresh=9)
        
        if fit_type =='kmeans':
            start_time = time.time()
            # Start all points in cluster 0
            cluster = np.zeros(len(data), dtype=int)

            # Pick 4 random initial points (2 for each line)
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
            # points_index = np.random.randint(len(data), size=4)
            # points_index = np.argmax(z)
            # points_index = np.append(np.argmax(z),points_index)
            # points_index = np.append(points_index,np.argmin(z))
            # points_index = np.append(points_index,np.argmin(z))
            # print(points_index)
            print(points)
            
            # Iterate the clusters and lines
            best_sos = np.inf
            for j in range(50):
                # points_index = np.random.randint(len(data), size=4)
                # points = data[points_index]
                # sum_of_squares = 0
                # Assign points to the closest line
                for i in range(len(data)):
                    # Distance to line 0
                    d0 = np.linalg.norm(np.cross(points[0] - points[1],points[1] - data[i])) / np.linalg.norm(points[1] - points[0])
                    # Distance to line 1
                    d1 = np.linalg.norm(np.cross(points[2] - points[3],points[3] - data[i])) / np.linalg.norm(points[3] - points[2])

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
                    t = np.linspace(-20, 20, 20)[:, np.newaxis]
                    linepts = mean + t * direction

                    if clust_id == 0:
                        linepts0 = linepts
                        points[0], points[1] = linepts[0], linepts[-1]
                    else:
                        linepts1 = linepts
                        points[2], points[3] = linepts[0], linepts[-1]
                
                # calculate the sum of squares of the distance of each point in each cluster to the line of best fit,
                # if it is the minimum value, keep that clustering and those lines of best fit
                # sum_of_squares += np.linalg.norm(np.cross(points[0] - points[1],points[1] - data[i])) / np.linalg.norm(points[1] - points[0])**2
                # sum_of_squares += np.linalg.norm(np.cross(points[0] - points[1],points[1] - data[i])) / np.linalg.norm(points[1] - points[0])**2

                # if sum_of_squares < best_sos:
                #     best_sos = sum_of_squares
                #     best_lines = np.array([linepts0,linepts1])
                #     best_cluster = cluster
                #     print("New Best: ", sum_of_squares)


                if j>0:
                    if np.array_equal(cluster,cluster_previous) == 1:
                        print("Clusters remain static at iter", j)
                        break
                cluster_previous = cluster
                    
            print("Time to fit with k-means: ", time.time() - start_time)
            # if points[3][2] - points[0][2] < 30:
            # Plot results
            fig = plt.figure()
            ax = fig.add_subplot(projection='3d')
            ax.scatter(*data[cluster == 0].T, color='teal', alpha=0.6, label="Cluster 0")
            ax.scatter(*data[cluster == 1].T, color='pink', alpha=0.6, label="Cluster 1")
            ax.plot(*linepts0.T, color='blue', linewidth=2)
            ax.plot(*linepts1.T, color='red', linewidth=2)
            ax.set_xlim3d(-200, 200)
            ax.set_ylim3d(-200, 200)
            ax.set_zlim3d(0, 400)
            ax.legend()
            plt.show()
            break
    
            # Once we extract the line along which each track travels, we use it to get the starting values for the fitter
            # Let's fit all combinations of forward and backward for the two clusters
            # point0f = 
            # point1f = 
            # point0b = 
            # point1b = 
            
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
            
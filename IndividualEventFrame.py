import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
import numpy as np
from raw_viewer import raw_trace_viewer
import scipy.stats

from sklearn.neighbors import NearestNeighbors

import gadget_widgets
from fit_gui.FitFrame import FitFrame

from skspatial.objects import Line

class IndividualEventFrame(ttk.Frame):
    def __init__(self, parent, run_data):
        super().__init__(parent)
        self.run_data = run_data
        #show background image
        self.background_image = gadget_widgets.get_background_image()
        self.background = ttk.Label(self, image=self.background_image)
        self.background.place(relx=0.5, rely=0.5, anchor='center')

        self.event_select_frame = ttk.LabelFrame(self, text='Event Selection')
        self.event_select_frame.pack()
        ttk.Label(self.event_select_frame, text='Event #').grid(row=0, column=0)
        self.event_num_entry = gadget_widgets.GEntry(self.event_select_frame,
                                                      text='Enter Event #')
        self.event_num_entry.grid(row=0, column=1)

        self.threeD_frame = ttk.LabelFrame(self, text='Point Cloud Viewer')
        track_w_trace_button = ttk.Button(self.threeD_frame,
                                          text='Show Track w/ Trace',
                                          command = self.track_w_trace)
        track_w_trace_button.grid(row=0, column=0, columnspan=2)
        ttk.Button(self.threeD_frame, text='3D Point Cloud', 
                   command=self.show_point_cloud).grid(row=1, column=0)
        ttk.Button(self.threeD_frame, text='3D Dense Point Cloud',
                   command=self.plot_dense_3d_track).grid(row=1, column=1)
        self.threeD_frame.pack()

        fitting_frame = ttk.LabelFrame(self, text='Point Cloud Fitting Tools')
        fitting_frame.pack()
        ttk.Label(fitting_frame, text='Bandwidth Factor:').grid(row=0, column=0)
        self.bandwidth_entry = ttk.Entry(fitting_frame)
        self.bandwidth_entry.grid(row=0, column=1)
        ttk.Button(fitting_frame, text='Project onto Principle Axis',
                   command=self.project_trace).grid(row=1, columnspan=2)
        
        trace_frame = ttk.LabelFrame(self, text='Original Trace Data')
        ttk.Label(trace_frame, text='threshold').grid(row=0, column=0)
        self.threshold_entry = ttk.Entry(trace_frame)
        self.threshold_entry.grid(row=0, column=1)
        ttk.Button(trace_frame, text='show raw traces', 
                   command=self.plot_traces).grid(row=1, column=0)
        ttk.Button(trace_frame, text='3D trace plot', 
                   command=self.plot_3d_traces).grid(row=1, column=1)
        trace_frame.pack()

    def plot_traces(self):
        event_num = int(self.event_num_entry.get())
        raw_trace_viewer.plot_traces(self.run_data.h5_file, event_num, False)

    def plot_3d_traces(self):
        event_num = int(self.event_num_entry.get())
        threshold = self.threshold_entry.get()
        if len(threshold) == 0:
            threshold = 0
        else:
            threshold = float(threshold)
        raw_trace_viewer.plot_3d_traces(self.run_data.h5_file, event_num, threshold, False)

    def project_trace(self):
        debug = True

        index = self.run_data.get_index(int(self.event_num_entry.get()))
        bandwidth = float(self.bandwidth_entry.get())

        xHit = self.run_data.xHit_list[index]
        yHit = self.run_data.yHit_list[index]
        zHit = self.run_data.zHit_list[index]
        eHit = self.run_data.eHit_list[index]

        extend_bins = 10 #TODO: this should probably be relaed to the bandwidth
        if debug:
            self.show_plot(xHit, yHit, zHit, eHit)
        
        #TODO: do we want to weight the fit by energy deposition?
        points = np.vstack((xHit, yHit, zHit)).transpose()
        line = Line.best_fit(points)
        pstart, direction = line.point, line.vector
        xProj, yProj, zProj = [], [], []
        for x,y,z in zip(xHit, yHit, zHit):
            v = line.project_vector([x, y, z])
            xProj.append(v[0])
            yProj.append(v[1])
            zProj.append(v[2])
        xProj = np.array(xProj)
        yProj = np.array(yProj)
        zProj = np.array(zProj)
        if debug:
            self.show_plot(xProj, yProj,zProj, eHit)

        dist = np.sqrt((xProj - pstart[0])**2 + (yProj - pstart[1])**2 + (zProj - pstart[2])**2)

        kde = scipy.stats.gaussian_kde(dist, weights=eHit)
        # division_factor = 3
        kde.set_bandwidth(kde.factor / bandwidth)  # You can adjust the bandwidth to control the smoothness

        # Create a dense array of x values for the histogram
        x_dense = np.linspace(np.min(dist) - extend_bins, np.max(dist) + extend_bins, 100)

        # Evaluate the KDE for the dense x values
        y_smooth = kde.evaluate(x_dense)

        new_window = tk.Toplevel()
        FitFrame(new_window, x_dense, y_smooth).pack()



    def show_plot(self, xHit, yHit, zHit, eHit):
        event_num = int(self.event_num_entry.get())
        index = self.run_data.get_index(event_num)
        fig = plt.figure(figsize=(6,6))
        ax = plt.axes(projection='3d')
        ax.set_xlim3d(-35, 35)
        ax.set_ylim3d(-35, 35)
        ax.set_zlim3d(0, 35)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        ax.set_title(f"3D Point-cloud of Event {event_num}\nLength: {self.run_data.len_list[index]:.2f}\nAngle: {self.run_data.angle_list[event_num]:.2f}", fontdict = {'fontsize' : 10})
        ax.scatter(xHit, yHit, zHit-np.min(zHit), c=eHit, cmap='RdBu_r')
        cbar = fig.colorbar(ax.get_children()[0])
        plt.show(block=False) 

    def track_w_trace(self):
        event_num = int(self.event_num_entry.get())
        index = self.run_data.get_index(event_num)
        plt.figure()
        self.run_data.make_image(index, show=True)

    def show_point_cloud(self):
        event_num = int(self.event_num_entry.get())
        xHit, yHit, zHit, eHit = self.run_data.get_hit_lists(event_num)
        self.show_plot(xHit, yHit, zHit, eHit)

    def plot_dense_3d_track(self):
        radius = 5

        event_num = int(self.event_num_entry.get())
        xHit, yHit, zHit, eHit = self.run_data.get_hit_lists(event_num)
        
        nbrs = NearestNeighbors(radius=radius).fit(np.vstack((xHit, yHit, zHit)).T)

        # Find the points within the specified radius
        points_within_radius = nbrs.radius_neighbors(np.vstack((xHit, yHit, zHit)).T, return_distance=False)

        # Interpolate between points within the radius
        xHit_dense, yHit_dense, zHit_dense, eHit_dense = [], [], [], []
        for i, neighbors in enumerate(points_within_radius):
            for j in neighbors:
                t = np.random.rand()
                xHit_dense.append(xHit[i] + t * (xHit[j] - xHit[i]))
                yHit_dense.append(yHit[i] + t * (yHit[j] - yHit[i]))
                zHit_dense.append(zHit[i] + t * (zHit[j] - zHit[i]))
                eHit_dense.append(eHit[i] + t * (eHit[j] - eHit[i]))

        # Convert lists to arrays
        xHit_dense = np.array(xHit_dense)
        yHit_dense = np.array(yHit_dense)
        zHit_dense = np.array(zHit_dense)
        eHit_dense = np.array(eHit_dense)

        self.show_plot(xHit_dense, yHit_dense, zHit_dense, eHit_dense)
import os
import configparser
import csv

import tkinter as tk
from tkinter import ttk
import tkinter.filedialog
import tkinter.messagebox

import matplotlib.pyplot as plt
import numpy as np
from skspatial.objects import Line, Point

from tqdm import tqdm

from fit_gui.HistogramFitFrame import HistogramFitFrame
from raw_viewer import raw_h5_file
from raw_viewer import heritage_h5_file

class RawEventViewerFrame(ttk.Frame):
    def __init__(self, parent, file_path=None, flat_lookup_path=None, heritage_file=False):
        super().__init__(parent)
        self.heritage_file = 10
        if not heritage_file:
            if file_path == None:
                file_path = tk.filedialog.askopenfilename(initialdir='/mnt/analysis/e21072/', title='Select H5 File', filetypes=[('H5', ".h5")])
            if flat_lookup_path == None:
                flat_lookup_path = tk.filedialog.askopenfilename(initialdir='./raw_viewer/channel_mappings', title='Select Channel Mapping FIle', filetypes=[('CSV', ".csv")])
            self.data = raw_h5_file.raw_h5_file(file_path, flat_lookup_csv=flat_lookup_path, zscale=1.45)
        else:
            if file_path == None:
                file_path = tk.filedialog.askopenfilename(initialdir='.', title='Select H5 File', filetypes=[('H5', ".h5")])
            self.data = heritage_h5_file.heritage_h5_file(file_path)
        self.winfo_toplevel().title(file_path)
        
        #objects whose values should be saved in settings file, indexed by settings file variable
        self.settings_entry_map={}
        self.settings_checkbutton_map={}
        self.settings_optionmenu_map={}

        settings_frame = ttk.LabelFrame(self, text='settings files and data export')
        ttk.Label(settings_frame, text='settings file:').grid(row=0, column=0)
        self.settings_file_entry = ttk.Entry(settings_frame)
        self.settings_file_entry.grid(row=0, column=1)
        ttk.Button(settings_frame, text='Browse', command=self.browse_for_settings_file).grid(row=0, column=2)
        ttk.Button(settings_frame, text='Load', command=self.load_settings_file).grid(row=0, column=3)
        ttk.Button(settings_frame, text='Save Config File', command=self.save_settings_file).grid(row=1, column=0, columnspan=3)

        ttk.Label(settings_frame, text='settings file status:').grid(row=2, column=0)
        self.settings_status_label = ttk.Label(settings_frame,text='no settings file loaded')
        self.settings_status_label.grid(row=2, column = 1, columnspan=2)
        settings_frame.grid()

        ttk.Button(settings_frame, text='process run', command=self.process_run).grid(row=3, column=0, columnspan=3)


        #widget setup in individual_event_Frame
        individual_event_frame = ttk.LabelFrame(self, text='Individual Events')

        ttk.Label(individual_event_frame, text='event #:').grid(row=0, column=0)
        self.event_number_entry = ttk.Entry(individual_event_frame)
        self.event_number_entry.insert(0, self.data.get_event_num_bounds()[0])
        self.event_number_entry.grid(row=0, column=1)

        ttk.Label(individual_event_frame, text='length ic threshold:').grid(row=1, column=0)
        self.length_threshold_entry = ttk.Entry(individual_event_frame)
        self.length_threshold_entry.insert(0, 100)
        self.length_threshold_entry.grid(row=1, column=1)
        self.length_threshold_entry.bind('<FocusOut>', self.entry_changed)
        self.settings_entry_map['length_ic_threshold']=self.length_threshold_entry

        ttk.Label(individual_event_frame, text='energy ic threshold:').grid(row=1, column=2)
        self.energy_threshold_entry = ttk.Entry(individual_event_frame)
        self.energy_threshold_entry.insert(0, 100)
        self.energy_threshold_entry.grid(row=1, column=3)
        self.energy_threshold_entry.bind('<FocusOut>', self.entry_changed)
        self.settings_entry_map['energy_ic_threshold']=self.energy_threshold_entry

        show_3d_button = ttk.Button(individual_event_frame, text='show', command = self.show_3d_cloud)
        show_3d_button.grid(row=2, column=0)

        next_button = ttk.Button(individual_event_frame, text='next', command=self.next)
        next_button.grid(row=2, column=1)

        show_2D_button = ttk.Button(individual_event_frame, text='x-y proj', command=self.show_xy_proj)
        show_2D_button.grid(row=3, column=0)
        show_traces_button = ttk.Button(individual_event_frame, text='pad traces', command=self.show_raw_traces)
        show_traces_button.grid(row=3, column=1)

        ttk.Button(individual_event_frame, text='1D proj on trac axis', command = self.project_to_principle_axis).grid(row=3, column=2)
        

        ttk.Label(individual_event_frame, text='view threshold:').grid(row=4, column=0)
        self.view_threshold_entry = ttk.Entry(individual_event_frame)
        self.view_threshold_entry.insert(0, '100')
        self.view_threshold_entry.grid(row=4, column=1)
        self.settings_entry_map['view_threshold']=self.view_threshold_entry

        ttk.Label(individual_event_frame, text='use data from CoBos:').grid(row=5, column=0)
        self.cobos_entry = ttk.Entry(individual_event_frame)
        self.cobos_entry.insert(0, 'all')
        self.cobos_entry.grid(row=5, column=1)
        self.cobos_entry.bind('<FocusOut>', self.entry_changed)
        self.settings_entry_map['include_cobos']=self.cobos_entry
        ttk.Label(individual_event_frame, text='use data from ASADs:').grid(row=5, column=2)
        self.asads_entry = ttk.Entry(individual_event_frame)
        self.asads_entry.insert(0, 'all')
        self.asads_entry.grid(row=5, column=3)
        self.asads_entry.bind('<FocusOut>', self.entry_changed)
        self.settings_entry_map['include_asads']=self.asads_entry
        ttk.Label(individual_event_frame, text='use data from pads:').grid(row=6, column=0)
        self.pads_entry = ttk.Entry(individual_event_frame)
        self.pads_entry.insert(0, 'all')
        self.pads_entry.grid(row=6, column=1)
        self.pads_entry.bind('<FocusOut>', self.entry_changed)
        self.settings_entry_map['include_pads']=self.pads_entry

        individual_event_frame.grid()
        
        count_hist_frame = ttk.LabelFrame(self, text='Counts Histogram')
        ttk.Label(count_hist_frame, text='# bins:').grid(row=0, column=0)
        self.bins_entry = ttk.Entry(count_hist_frame)
        self.bins_entry.insert(0, '100')
        self.bins_entry.grid(row=0, column=1)
        ttk.Label(count_hist_frame, text='veto pad threshold').grid(row=1, column=0)
        self.veto_threshold_entry = ttk.Entry(count_hist_frame)
        self.veto_threshold_entry.insert(0,'100')
        self.veto_threshold_entry.grid(row=1, column=1)
        self.veto_threshold_entry.bind('<FocusOut>', self.entry_changed)
        self.settings_entry_map['veto_threshold']=self.veto_threshold_entry
        ttk.Label(count_hist_frame,text='range min/max (mm):').grid(row=2, column=0)
        self.range_min_entry, self.range_max_entry = ttk.Entry(count_hist_frame), ttk.Entry(count_hist_frame)
        self.range_min_entry.grid(row=2, column=1)
        self.range_min_entry.insert(0, '0')
        self.range_min_entry.bind('<FocusOut>', self.entry_changed)
        self.settings_entry_map['range_min']=self.range_min_entry
        self.range_max_entry.grid(row=2, column=2)
        self.range_max_entry.insert(0, '100')
        self.range_max_entry.bind('<FocusOut>', self.entry_changed)
        self.settings_entry_map['range_max']=self.range_max_entry
        
        ttk.Label(count_hist_frame,text='ic min/max:').grid(row=3, column=0)
        self.ic_min_entry, self.ic_max_entry = ttk.Entry(count_hist_frame), ttk.Entry(count_hist_frame)
        self.ic_min_entry.grid(row=3, column=1)
        self.ic_min_entry.insert(0, '0')
        self.ic_min_entry.bind('<FocusOut>', self.entry_changed)
        self.ic_max_entry.grid(row=3, column=2)
        self.ic_max_entry.insert(0, '1e9')
        self.ic_max_entry.bind('<FocusOut>', self.entry_changed)
        self.settings_entry_map['min_ic']=self.ic_min_entry
        self.settings_entry_map['max_ic']=self.ic_max_entry

        ttk.Label(count_hist_frame,text='angle min/max:').grid(row=4, column=0)
        self.angle_min_entry, self.angle_max_entry = ttk.Entry(count_hist_frame), ttk.Entry(count_hist_frame)
        self.angle_min_entry.grid(row=4, column=1)
        self.angle_min_entry.insert(0, '0')
        self.angle_min_entry.bind('<FocusOut>', self.entry_changed)
        self.angle_max_entry.grid(row=4, column=2)
        self.angle_max_entry.insert(0, '90')
        self.angle_max_entry.bind('<FocusOut>', self.entry_changed)
        self.settings_entry_map['angle_min']=self.angle_min_entry
        self.settings_entry_map['angle_max']=self.angle_max_entry

        count_hist_button = ttk.Button(count_hist_frame, text='count histogram', command=self.show_count_hist)
        count_hist_button.grid(row=5, column=0)
        ttk.Button(count_hist_frame, text='RvE Histogram', command=self.show_rve_plot).grid(row=5, column=1)
        count_hist_frame.grid()

        settings_frame = ttk.LabelFrame(self, text='Processing Settings')
        ttk.Label(settings_frame, text='background time bins:').grid(row=0, column=0)
        self.background_start_entry = ttk.Entry(settings_frame)
        self.background_start_entry.grid(row=0, column=1)
        self.background_start_entry.bind('<FocusOut>', self.entry_changed)
        self.background_start_entry.insert(0, '0')
        self.background_stop_entry = ttk.Entry(settings_frame)
        self.background_stop_entry.grid(row=0, column=2)
        self.background_stop_entry.bind('<FocusOut>', self.entry_changed)
        self.background_stop_entry.insert(0, '200')
        self.settings_entry_map['background_bin_start']=self.background_start_entry
        self.settings_entry_map['background_bin_stop']=self.background_stop_entry

        get_backgrounds_button = ttk.Button(settings_frame, text='get pad backgrounds', command=self.get_backgrounds)
        get_backgrounds_button.grid(row = 1, column=0)
        show_backgrounds_button = ttk.Button(settings_frame, text='show pad backgrounds', command=self.show_backgrounds)
        show_backgrounds_button.grid(row=1, column=1)
        self.background_subtract_enable_var = tk.IntVar()
        background_subtract_check = ttk.Checkbutton(settings_frame, text='background subtraction', variable=self.background_subtract_enable_var, 
                                                         command=self.check_state_changed)
        self.settings_checkbutton_map['background_subtraction']=self.background_subtract_enable_var
        background_subtract_check.grid(row=2, column=0)
        
        self.remove_outlier_var = tk.IntVar()
        remove_outliers_check = ttk.Checkbutton(settings_frame, text='remove outlier pads', variable=self.remove_outlier_var, 
                                                         command=self.check_state_changed)
        self.settings_checkbutton_map['remove_outliers']=self.remove_outlier_var
        remove_outliers_check.grid(row=2, column=1)

        ttk.Label(settings_frame, text='zscale (mm/time bin):').grid(row=3,column=0)
        self.zscale_entry = ttk.Entry(settings_frame)
        self.zscale_entry.insert(0, '1.45')
        self.zscale_entry.grid(row=3,column=1)
        self.zscale_entry.bind('<FocusOut>', self.entry_changed)
        self.settings_entry_map['zscale']=self.zscale_entry

        self.mode_var = tk.StringVar() #traced added later, after all entries are created
        ttk.OptionMenu(settings_frame, self.mode_var, 'all data', 'all data', 'peak only', 'near peak').grid(row=4, column=0)
        self.settings_optionmenu_map['peak_mode']=self.mode_var
        ttk.Label(settings_frame, text='near peak window size:').grid(row=5, column=0)
        self.near_peak_window_entry = ttk.Entry(settings_frame)
        self.near_peak_window_entry.insert(0,'10')
        self.near_peak_window_entry.bind('<FocusOut>', self.entry_changed)
        self.near_peak_window_entry.grid(row=5, column=1)
        self.settings_entry_map['near_peak_window_width']=self.near_peak_window_entry
        ttk.Label(settings_frame, text='require peak between:').grid(row=6, column=0)
        self.peak_first_allowed_bin_entry, self.peak_last_allowed_bin_entry = ttk.Entry(settings_frame),ttk.Entry(settings_frame)
        self.peak_first_allowed_bin_entry.grid(row=6, column=1)
        self.peak_last_allowed_bin_entry.grid(row=6, column=2)
        self.peak_first_allowed_bin_entry.insert(0,'-inf')
        self.peak_last_allowed_bin_entry.insert(0,'inf')
        self.peak_first_allowed_bin_entry.bind('<FocusOut>', self.entry_changed)
        self.peak_last_allowed_bin_entry.bind('<FocusOut>', self.entry_changed)
        self.settings_entry_map['peak_first_allowed_bin']=self.peak_first_allowed_bin_entry
        self.settings_entry_map['peak_last_allowed_bin']=self.peak_last_allowed_bin_entry
        settings_frame.grid()
        self.mode_var.trace_add('write', lambda x,y,z: self.entry_changed(None))

        #sync setting with GUI
        self.entry_changed(None) 
        self.check_state_changed()
    
    def browse_for_settings_file(self):
        file_path = tk.filedialog.askopenfilename(initialdir='./raw_viewer/gui_configs', title='select GUI settings file', filetypes=([("gui config", ".gui_ini")]))
        self.settings_file_entry.delete(0, tk.END)
        self.settings_file_entry.insert(0, file_path)

    def load_settings_file(self):
        config = configparser.ConfigParser()
        file_path = self.settings_file_entry.get()
        config.read(file_path)

        for entry_name in config['ttk.Entry']:
            entry = self.settings_entry_map[entry_name]
            entry.delete(0, tk.END)
            entry.insert(0, config['ttk.Entry'][entry_name])

        for menu_name in config['ttk.OptionMenu']:
            var_to_update = self.settings_optionmenu_map[menu_name]
            var_to_update.set(config['ttk.OptionMenu'][menu_name])

        
        for checkbox_name in config['ttk.CheckButton']:
            var_to_update = self.settings_checkbutton_map[checkbox_name]
            var_to_update.set(config['ttk.CheckButton'][checkbox_name])

        #apply settings to raw data object
        self.entry_changed(None)
        self.check_state_changed()

    def save_settings_file(self,file_path=None):
        if file_path == None:
            file_path = tk.filedialog.asksaveasfilename(initialdir='.', title='GUI settings file save path', filetypes=([("gui config", ".gui_ini")]), defaultextension='.gui_ini')
        config = configparser.ConfigParser()
        
        entries_to_save = {}
        for entry_name in self.settings_entry_map:
            entries_to_save[entry_name] = self.settings_entry_map[entry_name].get()
        config['ttk.Entry']=entries_to_save

        option_menus_to_save = {}
        for menu_name in self.settings_optionmenu_map:
            option_menus_to_save[menu_name] = self.settings_optionmenu_map[menu_name].get()
        config['ttk.OptionMenu']=option_menus_to_save

        check_buttons_to_save = {}
        for check_name in self.settings_checkbutton_map:
            check_buttons_to_save[check_name] = self.settings_checkbutton_map[check_name].get()
        config['ttk.CheckButton']=check_buttons_to_save

        with open(file_path, 'w') as configfile:
            config.write(configfile)

    def process_run(self):
        directory_path, h5_fname = os.path.split(self.data.file_path)
        directory_path = os.path.join(directory_path, os.path.splitext(h5_fname)[0]+'_raw_data_export')
        assert not os.path.isdir(directory_path) #TODO: make this run # + config file name, or pop up for non-up to date config
        os.mkdir(directory_path)

        self.save_settings_file(os.path.join(directory_path, 'config.gui_ini'))

        ranges, counts, angles, pads_railed_list = self.data.get_histogram_arrays()
        np.save(os.path.join(directory_path, 'counts.npy'), counts)
        np.save(os.path.join(directory_path, 'ranges.npy'), ranges)
        np.save(os.path.join(directory_path, 'angles.npy'), angles)
        with open(os.path.join(directory_path, 'pads_railed.csv'), 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(pads_railed_list)

        #TODO: save git hash and if up to date


    def show_3d_cloud(self):
        event_number = int(self.event_number_entry.get())
        self.data.plot_3d_traces(event_number, threshold=float(self.view_threshold_entry.get()),block=False)
    
    def next(self):
        plt.close()
        event_number = int(self.event_number_entry.get())+1
        veto, length, energy, angle, pads_railed = self.data.process_event(event_number)
        while veto:
            event_number += 1
            veto, length, energy, angle, pads_railed = self.data.process_event(event_number)
        self.event_number_entry.delete(0, tk.END)
        self.event_number_entry.insert(0, event_number)
        self.show_3d_cloud()

    def show_raw_traces(self):
        event_number = int(self.event_number_entry.get())
        self.data.plot_traces(event_number, block=False)

    def show_xy_proj(self):
        event_number = int(self.event_number_entry.get())
        self.data.show_2d_projection(event_number, False)
    
    def show_count_hist(self):
        bins = int(self.bins_entry.get())
        self.data.show_counts_histogram(num_bins=bins, block=False)

    def get_backgrounds(self):
        #TODO: update to use range
        background_bins = int(self.background_start_entry.get())
        self.data.determine_pad_backgrounds(background_bins)

    def show_backgrounds(self):
        self.data.show_pad_backgrounds()

    def check_state_changed(self):
        self.data.apply_background_subtraction = (self.background_subtract_enable_var.get() == 1)
        self.data.remove_outliers = (self.remove_outlier_var.get() == 1)

    def show_rve_plot(self):
        bins = int(self.bins_entry.get())
        self.data.show_rve_histogram(num_e_bins=bins, num_range_bins=bins, block=False)

    def entry_changed(self, event):
        self.data.zscale = float(self.zscale_entry.get())
        self.data.num_background_bins = (int(self.background_start_entry.get()), int(self.background_stop_entry.get()))
        self.data.ic_bounds = (float(self.ic_min_entry.get()), float(self.ic_max_entry.get()))
        self.data.range_bounds = (float(self.range_min_entry.get()), float(self.range_max_entry.get()))
        self.data.angle_bounds = (np.radians(float(self.angle_min_entry.get())), np.radians(float(self.angle_max_entry.get())))
        self.data.length_counts_threshold = float(self.length_threshold_entry.get())
        self.data.ic_counts_threshold = float(self.energy_threshold_entry.get())
        self.data.veto_threshold = float(self.veto_threshold_entry.get())
        self.data.mode = self.mode_var.get()
        self.data.near_peak_window_width = int(self.near_peak_window_entry.get())
        self.data.require_peak_within=(float(self.peak_first_allowed_bin_entry.get()), float(self.peak_last_allowed_bin_entry.get()))
        asads = self.asads_entry.get()
        if asads.lower() == 'all':
            self.data.asads = 'all'
        else:
            self.data.asads = np.fromstring(asads, sep=',')
        cobos = self.cobos_entry.get()
        if cobos.lower() == 'all':
            self.data.cobos = 'all'
        else:
            self.data.cobos = np.fromstring(cobos, sep=',')
        pads = self.pads_entry.get()
        if pads.lower() == 'all':
            self.data.pads = 'all'
        else:
            self.data.pads = np.fromstring(pads, sep=',')


    def project_to_principle_axis(self):
        '''
        Do the following for the current event:
            Get all data above length thresholdand find best fit line for all points above length threshold
            Get all data above ic threshold and project these points to the line found in previous step
            Generate histogram and export to histogram fit frame
        '''
        event_number = int(self.event_number_entry.get())
        length_threshold = float(self.length_threshold_entry.get())
        ic_threshold = float(self.energy_threshold_entry.get())
        #get fit line for all data above the length threshold
        xs, ys, zs, es = self.data.get_xyze(event_number, include_veto_pads=False)
        if length_threshold != -np.inf:
            xs = xs[es>length_threshold]
            ys = ys[es>length_threshold]
            zs = zs[es>length_threshold]
            es = es[es>length_threshold]
            points = np.vstack((xs, ys, zs)).transpose()
        line = Line.best_fit(points)
        DEBUG=False
        if DEBUG:
            fig = plt.figure(figsize=(6,6))
            plt.clf()
            ax = plt.axes(projection='3d')
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_zlabel("z")
            ax.set_xlim3d(-200, 200)
            ax.set_ylim3d(-200, 200)
            ax.set_zlim3d(0, 400)
            xs, ys, zs, es = self.data.get_xyze(event_number, threshold=length_threshold)
            ax.scatter(xs, ys, zs, c=es)
        #bin the charge, assume charge is uniformly distributed in each voxel.
        #approximate this by breaking each point into N^3 points, each with 1/N^3
        #the charge of the original point
        N = 10
        xs, ys, zs, es= [],[],[],[]
        dxys = np.arange(0, 2.2, 2.2/N)
        dzs = np.arange(0,self.data.zscale, self.data.zscale/N)
        if self.heritage_file: 
            #smear z even more because of timing jitter
            #TODO: be rigorous about this
            dzs *= 10
        x_old, y_old, z_old, e_old = self.data.get_xyze(event_number, include_veto_pads=False)
        x_old = x_old[e_old>ic_threshold]
        y_old = y_old[e_old>ic_threshold]
        z_old = z_old[e_old>ic_threshold]
        e_old = e_old[e_old>ic_threshold]
        for x,y,z,e in tqdm(zip(x_old, y_old, z_old, e_old)):
            for dx in dxys:
                for dy in dxys:
                    for dz in dzs:
                        xs.append(x+dx)
                        ys.append(y+dy)
                        zs.append(z+dz)
                        es.append(e/N**3)
        xs, ys, zs, es = np.array(xs), np.array(ys), np.array(zs), np.array(es)

        
        #calculate distance along projected axis for each point
        pstart, direction_vect = line.point, line.vector
        direction_vect = direction_vect.unit()
        dist = []
        xproj, yproj, zproj = [],[],[]
        for x,y,z in tqdm(zip(xs, ys, zs)):
            point = Point([x,y,z])
            #point = line.project_point(point)
            dist.append(np.dot(point - pstart, direction_vect))

        if DEBUG:
            ax.scatter(xproj[::30], yproj[::30], zproj[::30], c=es[::30])
            plt.show(block=False)

        #dist = np.sqrt((xProj - pstart[0])**2 + (yProj - pstart[1])**2 + (zProj - pstart[2])**2)
        #export to histogram fit frame
        np.save('dist', dist)
        np.save('e', es)
        print('opening histogram fit frame)')
        new_window = tk.Toplevel()
        HistogramFitFrame(new_window, data=dist, weights=es).pack()


    def get_track_info(self):
        event_number = int(self.event_number_entry.get())
        return self.data.process_event(event_num=event_number)
        
    def show_track_info(self):
        veto, length, energy = self.get_track_info()
        tk.messagebox.showinfo(message='length = %f mm, energy=%E, veto=%d'%(length, energy, veto))




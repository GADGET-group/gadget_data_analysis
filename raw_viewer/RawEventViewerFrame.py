import tkinter as tk
from tkinter import ttk
import tkinter.filedialog
import tkinter.messagebox

import matplotlib.pyplot as plt
import numpy as np
import raw_h5_file

class IndividualEventFrame(ttk.Frame):
    def __init__(self, parent):
        super().__init__(parent)
        #get file path and load file
        file_path = tk.filedialog.askopenfilename(initialdir='/mnt/analysis/e21072/', title='Select a Directory')
        self.data = raw_h5_file.raw_h5_file(file_path, zscale=1.45)
        self.winfo_toplevel().title(file_path)
        
        #widget setup in individual_event_Frame
        individual_event_frame = ttk.LabelFrame(self, text='Individual Events')

        ttk.Label(individual_event_frame, text='event #:').grid(row=0, column=0)
        self.event_number_entry = ttk.Entry(individual_event_frame)
        self.event_number_entry.insert(0, self.data.get_event_num_bounds()[0])
        self.event_number_entry.grid(row=0, column=1)

        ttk.Label(individual_event_frame, text='threshold:').grid(row=1, column=0)
        self.threshold_entry = ttk.Entry(individual_event_frame)
        self.threshold_entry.insert(0, 500)
        self.threshold_entry.grid(row=1, column=1)

        show_3d_button = ttk.Button(individual_event_frame, text='show', command = self.show_3d_cloud)
        show_3d_button.grid(row=2, column=0)

        next_button = ttk.Button(individual_event_frame, text='next', command=self.next)
        next_button.grid(row=2, column=1)

        show_2D_button = ttk.Button(individual_event_frame, text='x-y proj', command=self.show_xy_proj)
        show_2D_button.grid(row=3, column=0)
        show_traces_button = ttk.Button(individual_event_frame, text='pad traces', command=self.show_raw_traces)
        show_traces_button.grid(row=3, column=1)

        ttk.Label(individual_event_frame, text='# pad threshold:').grid(row=4, column=0)
        self.num_pad_threshold_entry = ttk.Entry(individual_event_frame)
        self.num_pad_threshold_entry.insert(0, 10)
        self.num_pad_threshold_entry.grid(row=4, column=2)

        ttk.Button(individual_event_frame, text='show track info', command=self.show_track_info).grid()

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
        ttk.Label(count_hist_frame,text='range min/max (mm):').grid(row=2, column=0)
        self.range_min_entry, self.range_max_entry = ttk.Entry(count_hist_frame), ttk.Entry(count_hist_frame)
        self.range_min_entry.grid(row=2, column=1)
        self.range_min_entry.insert(0, '0')
        self.range_max_entry.grid(row=2, column=2)
        self.range_max_entry.insert(0, '1000')
        count_hist_button = ttk.Button(count_hist_frame, text='count histogram', command=self.show_count_hist)
        count_hist_button.grid(row=3, column=0)
        rve_hist_button = ttk.Button(count_hist_frame, text='RvE Histogram', command=self.show_rve_plot).grid(row=3, column=1)
        count_hist_frame.grid()

        settings_frame = ttk.LabelFrame(self, text='Processing Settings')
        ttk.Label(settings_frame, text='# background time bins:').grid(row=0, column=0)
        self.background_bins_entry = ttk.Entry(settings_frame)
        self.background_bins_entry.grid(row=0, column=1)
        self.background_bins_entry.bind('<FocusOut>', self.entry_changed)
        get_backgrounds_button = ttk.Button(settings_frame, text='get pad backgrounds', command=self.get_backgrounds)
        get_backgrounds_button.grid(row = 1, column=0)
        show_backgrounds_button = ttk.Button(settings_frame, text='show pad backgrounds', command=self.show_backgrounds)
        show_backgrounds_button.grid(row=1, column=1)
        self.background_subtract_enable_var = tk.IntVar()
        background_subtract_check = ttk.Checkbutton(settings_frame, text='background subtraction', variable=self.background_subtract_enable_var, 
                                                         command=self.check_state_changed)
        background_subtract_check.grid(row=2, column=0)
        self.remove_outlier_var = tk.IntVar()
        remove_outliers_check = ttk.Checkbutton(settings_frame, text='remove outlier pads', variable=self.remove_outlier_var, 
                                                         command=self.check_state_changed)
        ttk.Label(settings_frame, text='zscale (mm/time bin):').grid(row=3,column=0)
        self.zscale_entry = ttk.Entry(settings_frame)
        self.zscale_entry.insert(0, '1')
        self.zscale_entry.grid(row=3,column=1)
        self.zscale_entry.bind('<FocusOut>', self.entry_changed)

        remove_outliers_check.grid(row=2, column=1)
        settings_frame.grid()
    
    def show_3d_cloud(self):
        event_number = int(self.event_number_entry.get())
        threshold = int(self.threshold_entry.get())
        self.data.plot_3d_traces(event_number, threshold=threshold, block=False)
    
    def next(self):
        plt.close()
        event_number = int(self.event_number_entry.get())+1
        pads_threshold = int(self.num_pad_threshold_entry.get())
        while self.data.get_num_pads_fired(event_number) < pads_threshold:
            event_number += 1
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
        threshold = float(self.threshold_entry.get())
        veto_threshold = float(self.veto_threshold_entry.get())
        bins = int(self.bins_entry.get())
        min_range = float(self.range_min_entry.get())
        max_range = float(self.range_max_entry.get())
        self.data.show_counts_histogram(num_bins=bins, threshold=threshold, 
                                        veto_threshold=veto_threshold,
                                        range_bounds=(min_range, max_range),block=False)

    def get_backgrounds(self):
        background_bins = int(self.background_bins_entry.get())
        self.data.determine_pad_backgrounds(background_bins)

    def show_backgrounds(self):
        self.data.show_pad_backgrounds()

    def check_state_changed(self):
        self.data.apply_background_subtraction = (self.background_subtract_enable_var.get() == 1)
        self.data.remove_outliers = (self.remove_outlier_var.get() == 1)
        
    def show_track_info(self):
        event_number = int(self.event_number_entry.get())
        threshold = float(self.threshold_entry.get())
        length = self.data.get_track_length(event_number, threshold)
        tk.messagebox.showinfo(message='track length = %f mm'%length)

    def show_rve_plot(self):
        threshold = float(self.threshold_entry.get())
        veto_threshold = float(self.veto_threshold_entry.get())
        bins = int(self.bins_entry.get())
        min_range = float(self.range_min_entry.get())
        max_range = float(self.range_max_entry.get())
        self.data.show_rve_histogram(num_e_bins=bins, num_range_bins=bins, threshold=threshold, 
                                     veto_threshold=threshold, block=False, range_bounds=(min_range, max_range))

    def entry_changed(self, event):
        self.data.zscale = float(self.zscale_entry.get())
        self.data.num_background_bins = int(self.background_bins_entry.get())

if __name__ == '__main__':
    root = tk.Tk()
    IndividualEventFrame(root).grid()
    root.mainloop()
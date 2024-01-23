import tkinter as tk
from tkinter import ttk
import tkinter.filedialog
import matplotlib.pyplot as plt
import numpy as np
import raw_trace_viewer
import h5py

class IndividualEventFrame(ttk.Frame):
    def __init__(self, parent):
        super().__init__(parent)
        #get file path and load file
        file_path = tk.filedialog.askopenfilename(initialdir='/mnt/analysis/e21072/', title='Select a Directory')
        self.h5_file = h5py.File(file_path, 'r')
        self.winfo_toplevel().title(file_path)
        
        #widget setup in individual_event_Frame
        individual_event_frame = ttk.LabelFrame(self, text='Individual Events')

        ttk.Label(individual_event_frame, text='event #:').grid(row=0, column=0)
        self.event_number_entry = ttk.Entry(individual_event_frame)
        self.event_number_entry.insert(0, raw_trace_viewer.get_first_good_event_number(self.h5_file))
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

        individual_event_frame.grid()

        #widgets in run_frame
        run_frame = ttk.LabelFrame(self, text='Entire Run')
        ttk.Label(run_frame, text='# bins:').grid(row=0, column=0)
        self.bins_entry = ttk.Entry(run_frame)
        self.bins_entry.grid(row=0, column=1)
        count_hist_button = ttk.Button(run_frame, text='count histogram', command=self.show_count_hist)
        count_hist_button.grid()
        run_frame.grid()
    
    def show_3d_cloud(self):
        event_number = int(self.event_number_entry.get())
        threshold = int(self.threshold_entry.get())
        raw_trace_viewer.plot_3d_traces(self.h5_file, event_number, threshold=threshold, block=False)
    
    def next(self):
        plt.close()
        event_number = int(self.event_number_entry.get())+1
        pads_threshold = int(self.num_pad_threshold_entry.get())
        while raw_trace_viewer.get_pads_fired(self.h5_file, event_number) < pads_threshold:
            event_number += 1
        self.event_number_entry.delete(0, tk.END)
        self.event_number_entry.insert(0, event_number)
        self.show_3d_cloud()

    def show_raw_traces(self):
        event_number = int(self.event_number_entry.get())
        raw_trace_viewer.plot_traces(self.h5_file, event_number, block=False)

    def show_xy_proj(self):
        event_number = int(self.event_number_entry.get())
        raw_trace_viewer.show_2d_projection(self.h5_file, event_number, block=False)

    def show_count_hist(self):
        hist = raw_trace_viewer.get_counts_array(self.h5_file)
        bins=int(self.bins_entry.get())
        plt.figure()
        plt.hist(hist, bins)
        plt.yscale('log')
        plt.show(block=False)

if __name__ == '__main__':
    root = tk.Tk()
    IndividualEventFrame(root).grid()
    root.mainloop()
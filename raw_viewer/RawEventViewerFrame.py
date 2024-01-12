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
        
        file_path = tk.filedialog.askopenfilename(initialdir='/mnt/analysis/e21072/', title='Select a Directory')
        self.h5_file = h5py.File(file_path, 'r')

        ttk.Label(self, text='event #:').grid(row=0, column=0)
        self.event_number_entry = ttk.Entry(self)
        self.event_number_entry.insert(0, raw_trace_viewer.get_first_good_event_number(self.h5_file))
        self.event_number_entry.grid(row=0, column=1)

        ttk.Label(self, text='threshold:').grid(row=1, column=0)
        self.threshold_entry = ttk.Entry(self)
        self.threshold_entry.insert(0, 500)
        self.threshold_entry.grid(row=1, column=1)

        show_3d_button = ttk.Button(self, text='show', command = self.show_3d_cloud)
        show_3d_button.grid(row=2, column=0)

        next_button = ttk.Button(self, text='next', command=self.next)
        next_button.grid(row=2, column=1)

        show_2D_button = ttk.Button(self, text='x-y proj', command=self.show_xy_proj)
        show_2D_button.grid(row=3)

        ttk.Label(self, text='# pad threshold:').grid(row=4, column=0)
        self.num_pad_threshold_entry = ttk.Entry(self)
        self.num_pad_threshold_entry.insert(0, 10)
        self.num_pad_threshold_entry.grid(row=4, column=2)
    
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

    def show_xy_proj(self):
        event_number = int(self.event_number_entry.get())
        raw_trace_viewer.show_2d_projection(self.h5_file, event_number, block=False)

if __name__ == '__main__':
    root = tk.Tk()
    IndividualEventFrame(root).grid()
    root.mainloop()
import datetime
import random
import os

import tkinter.filedialog
import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.patches as patches
from matplotlib.path import Path
from GadgetRunH5 import GadgetRunH5
import numpy as np
from tqdm import tqdm

import gadget_widgets
from prev_cut_select_window import PrevCutSelectWindow

class RvE_Frame(ttk.Frame):
    def __init__(self, parent, run_data:GadgetRunH5):
        super().__init__(parent)
        self.run_data = run_data
        #show background image
        self.background_image = gadget_widgets.get_background_image()
        self.background = ttk.Label(self, image=self.background_image)
        self.background.place(relx=0.5, rely=0.5, anchor='center')
        #plot settings
        self.plot_settings_frame = ttk.LabelFrame(self,text='plot settings')
        self.plot_settings_frame.grid(row=0)
        self.range_bins_label = ttk.Label(self.plot_settings_frame, text='# range bins:')
        self.range_bins_label.grid(row=0, column=0)
        self.range_bins_entry = gadget_widgets.GEntry(self.plot_settings_frame)
        self.range_bins_entry.grid(row=0, column=1)
        self.range_bins_entry.insert(0,'200')
        self.energy_bins_label = ttk.Label(self.plot_settings_frame, text='# energy bins:')
        self.energy_bins_label.grid(row=0, column=2)
        self.energy_bins_entry = gadget_widgets.GEntry(self.plot_settings_frame)
        self.energy_bins_entry.grid(row=0, column=3)
        self.energy_bins_entry.insert(0,'200')
        self.log_scale_var = tk.BooleanVar(value=True)
        self.scale_label = ttk.Label(self.plot_settings_frame, text='energy scale:')
        self.scale_label.grid(row=1, column=1, sticky=tk.E)
        self.log_scale_radio = ttk.Radiobutton(self.plot_settings_frame,
                                               text='log', variable=self.log_scale_var,
                                               value=True)
        self.lin_scale_radio = ttk.Radiobutton(self.plot_settings_frame,
                                               text='linear', variable=self.log_scale_var,
                                               value=False)
        self.lin_scale_radio.grid(row=1,column=2)
        self.log_scale_radio.grid(row=1,column=3)
        #basic viewing tools
        self.view_frame = ttk.LabelFrame(self, text='viewing tools')
        self.view_frame.grid(row=1)
        self.show_rve_plot_button = ttk.Button(self.view_frame, text='Plot Range vs Energy',
                                               command=self.plot_spectrum)
        self.show_rve_plot_button.grid(row=0, column=0, columnspan=2)
        self.event_num_entry = gadget_widgets.GEntry(self.view_frame, default_text='Event #')
        self.event_num_entry.grid(row=1, column=0)
        self.show_event_button = ttk.Button(self.view_frame, text='show event on RvE plot',
                                            command=self.show_event)
        self.show_event_button.grid(row=1, column=1)
        self.cut_tools_frame = ttk.LabelFrame(self, text='cut tools')
        self.cut_tools_frame.grid(row=2)
        #TODO: implement manual cut, and project to axis
        self.manual_cut_button = ttk.Button(self.cut_tools_frame, text='Manual Cut Selection')
        self.manual_cut_button.grid(row=0, column=0)
        self.from_file_cut_button = ttk.Button(self.cut_tools_frame,
                                               text='Polygon from File',
                                               command=self.cut_from_file)
        self.from_file_cut_button.grid(row=0, column=1)
        self.prev_cut_button = ttk.Button(self.cut_tools_frame, 
                                          text='Previous Cuts',
                                          command=self.prev_cut)
        self.prev_cut_button.grid(row=1, column=0, columnspan=2)
        self.project_cut_x_ax_button = ttk.Button(self.cut_tools_frame, text='Project Cut to X-axis')
        self.project_cut_x_ax_button.grid(row=2, column=0)
        self.project_cut_y_ax_button = ttk.Button(self.cut_tools_frame, text='Project Cut to Y-axis')
        self.project_cut_y_ax_button.grid(row=2, column=1)

    def plot_spectrum(self, fig_name='RvE',clear=True, show=True):
        num_range_bins = int(self.range_bins_entry.get())
        num_energy_bins = int(self.energy_bins_entry.get())
        
        plt.figure(fig_name, clear=clear)
        plt.xlabel('Energy (MeV)', fontdict={'fontsize': 20})
        plt.ylabel('Range (mm)', fontdict={'fontsize': 20})
        plt.title(f'Range vs Energy \n Energy Bins = {num_energy_bins} | Range Bins = {num_range_bins}', fontdict={'fontsize': 20})
        tot_energy_temp = np.concatenate(([0], self.run_data.total_energy_MeV))
        len_list_temp = np.concatenate(([0], self.run_data.len_list))
        if self.log_scale_var.get():
            norm = colors.LogNorm()
        else:
            norm = colors.Normalize()
        plt.hist2d(tot_energy_temp, len_list_temp, (num_energy_bins, num_range_bins), 
                   cmap=plt.cm.jet, norm=norm)
        plt.colorbar()
        plt.gca().set_facecolor('darkblue')
        if show:
            plt.show(block=False)

    def show_event(self): #TODO: add "show annotation" checkbox
        #only draw plot if it's not already open
        if not plt.fignum_exists('RvE'):
            self.plot_spectrum()
        else:
            plt.figure('RvE')  # switch focus back to RvE plot
        event_num = int(self.event_num_entry.get())
        event_index = self.run_data.get_index(event_num)
        plt.plot(self.run_data.total_energy_MeV[event_index], self.run_data.len_list[event_index], 'ro', picker=5) 
        plt.annotate(f"Evt#: {event_num}", (self.run_data.total_energy_MeV[event_index], 
                    self.run_data.len_list[event_index]), textcoords="offset points", xytext=(-15,7),
                    ha='center', fontsize=10, color='black',
                    bbox=dict(boxstyle="round,pad=0.5", facecolor="yellow", edgecolor="black"))
        plt.show(block=False)
        
    def prev_cut(self):
        PrevCutSelectWindow(self, self.run_data)

    def save_cut_files(self, points):
        '''
        points: verticies specifying the cut region
        '''
        now = datetime.datetime.now()
        rand_num = str(random.randrange(0,1000000,1))
        cut_name = rand_num+now.strftime("CUT_Date_%m_%d_%Y")
        event_images_path = os.path.join(self.run_data.folder_path, cut_name)

        #save an image for future cut selection
        self.plot_spectrum(fig_name=cut_name)
        ax = plt.gca()
        #add once point and use this to close the path
        #actual value of final vertex is ignored for CLOSEPOLY code
        points_list = list(points)
        points_list.append([0,0])
        codes = [Path.LINETO]*len(points_list)
        codes[0] = Path.MOVETO
        codes[-1] = Path.CLOSEPOLY
        path = Path(points_list, codes)
        
        to_draw = patches.PathPatch(path, fill=False, color='red')
        ax.add_patch(to_draw)
        plt.savefig(os.path.join(self.run_data.folder_path, cut_name+'.jpg'))
        plt.close()

        #save images of the selected events
        os.makedirs(event_images_path)
        selected_indices = self.run_data.get_RvE_cut_indexes(points)
        for index in tqdm(selected_indices):
            image_name = f"run{self.run_data.run_num}_image_{index}.jpg"
            image_path = os.path.join(event_images_path, image_name)
            self.run_data.save_image(index, save_path=image_path)
        #save the cut parameters used
        np.savetxt(os.path.join(event_images_path, 'cut_used.txt'), points)


    def cut_from_file(self):
        '''
        Input files should have an energy (MeV) followed by range (mm) on each line,
        with the values seperated by a space. 
        '''
        fname = tkinter.filedialog.askopenfile(initialdir=os.getcwd())
        points = np.loadtxt(fname)
        self.save_cut_files(points)
import os
import glob
import pickle

from tqdm import tqdm
import tkinter as tk
from tkinter import ttk
from PIL import ImageTk, Image
import h5py
from tkinter import messagebox

from gadget_analysis.main_gui.view_cut_images_window import ViewCutImagesWindow

class PrevCutSelectWindow(tk.Toplevel):
    def __init__(self, parent, run_data, **kwargs):
        super().__init__(parent, **kwargs)
        self.run_data = run_data
        #get previous cut information
        prev_cut_path = os.path.join(self.run_data.folder_path, "*.jpg")
        self.image_path_list = sorted(glob.glob(prev_cut_path))
        if not self.image_path_list:
            messagebox.showinfo('No saved cuts', f'No cuts have been saved for this run yet ({self.run_data.folder_path}).', parent=parent)
            self.destroy()
            return
        self.image_list = []
        for i in range(len(self.image_path_list)):
            self.image_list.append(ImageTk.PhotoImage(Image.open(self.image_path_list[i])))
        self.current_image_index = 0
        #setup GUI
        self.back_button = ttk.Button(self, text="<<", command=self.back, state=tk.DISABLED)
        self.exit_button = ttk.Button(self, text="Exit Program", command=self.destroy)
        self.forward_button = ttk.Button(self, text=">>", command=self.forward)
        self.select_button = ttk.Button(self, text="Select Cut", command=self.select_cut)
        self.create_h5_button = ttk.Button(self, text="Create H5 File", command=self.create_h5)
        self.image_label = ttk.Label(self, image=self.image_list[self.current_image_index])        

        self.back_button.grid(row=0, column=0)
        self.select_button.grid(row=0, column=1)
        self.forward_button.grid(row=0, column=2)
        self.exit_button.grid(row=1, column=1)
        self.image_label.grid(row=2, column=0, columnspan=3)
        self.create_h5_button.grid(row=3, column=1)

        self.change_image_index_to(0) #enables/disables next button as needed

    def update_title(self, image_index, init_image_list):
        dir_name = init_image_list[image_index][:-4]
        self.title(f'Image Viewer - {dir_name}')
    
    def change_image_index_to(self, new_index):
        self.current_image_index = new_index
        if self.current_image_index == 0:
            self.back_button.state(["disabled"])
        else:
            self.back_button.state(["!disabled"])
        if self.current_image_index == (len(self.image_list) - 1):
            self.forward_button.state(["disabled"])
        else:
            self.forward_button.state(["!disabled"])
        self.image_label.configure(image=self.image_list[self.current_image_index])

    def forward(self):
         self.change_image_index_to(self.current_image_index + 1)
    
    def back(self):
         self.change_image_index_to(self.current_image_index - 1)
    
    def select_cut(self):
         dir = self.image_path_list[self.current_image_index][:-4] #-4 takes off the .jpg
         ViewCutImagesWindow(self.master, self.run_data, dir)
         self.destroy()
         

    
    def create_h5(self):
        dir_select = self.image_path_list[self.current_image_index][:-4]
        print('CUT DIRECTORY \n',dir_select)

        cut_indicies_str = "cut_indices_H5list.pkl"
        cut_indicies_path = os.path.join(dir_select, cut_indicies_str)

        with open(cut_indicies_path, "rb") as file:
                cut_indices_H5list = pickle.load(file)

        # copy the meta group and the GET datasets of the cut's events (same paths, so raw_h5_file reads the new file;
        # its [meta][meta] event range is the run's, events outside the cut are absent)
        source = self.run_data.h5_file
        h5_path = os.path.join(dir_select, "h5_cut.h5")
        with h5py.File(self.run_data.h5_file_path, 'r') as original, h5py.File(h5_path, 'w') as new:
            original.copy('meta', new)
            new_trace_group = new.create_group('get')
            for i in tqdm(cut_indices_H5list):
                for kind in ('data', 'header'):
                    path = source._event_dataset_path(int(i), kind)
                    if path in original['get']:
                        original['get'].copy(path, new_trace_group, name=path)
        print('wrote', h5_path)

        self.destroy()
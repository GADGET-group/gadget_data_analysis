import os
import glob

import tkinter as tk
from tkinter import ttk
from tkinter import messagebox

from PIL import Image, ImageTk
from tqdm import tqdm

class ViewCutImagesWindow(tk.Toplevel):
    def __init__(self, parent, run_data, selected_dir,**kwargs):
        super().__init__(parent, **kwargs)
        self.run_data = run_data
        self.selected_dir = selected_dir

        #load images
        image_search_string = os.path.join(selected_dir, "*.jpg")
        self.image_path_list = glob.glob(image_search_string)

        #setup GUI
        self.notebook = ttk.Notebook(self)
        self.back_button = ttk.Button(self, text='<<', command=self.back)
        self.next_button = ttk.Button(self, text='>>', command=self.next)
        self.go_to_entry = ttk.Entry(self)
        self.go_to_button = ttk.Button(self, text='Go to Image', command=self.go_to)
        
        self.single_image_label = ttk.Label(self.notebook, anchor=tk.CENTER )
        self.notebook.add(self.single_image_label, text='single')

        self.grid_frame = ttk.Frame(self.notebook)
        self.grid_image_labels = []
        for i in range(9):
            self.grid_image_labels.append(ttk.Label(self.grid_frame))
            self.grid_image_labels[i].grid(row=int(i/3), column=i%3)
        self.notebook.add(self.grid_frame, text='3x3')

        self.back_button.grid(row=0, column=0)
        self.next_button.grid(row=0, column=3)
        self.go_to_entry.grid(row=1, column=1)
        self.go_to_button.grid(row=1, column=2)
        self.notebook.grid(row=2, column=0, columnspan=4)

        self.current_index = 0 #track which image we're viewing
        self.change_index(self.current_index)
        
    def change_index(self, index):
        self.current_index = index
        #enable/disable forward and backwards buttons based on selected tab
        if self.notebook.tab(self.notebook.select(), "text")=='single':
            if self.current_index == 0:
                self.back_button.state(["disabled"])
            else:
                self.back_button.state(["!disabled"])
            if self.current_index == (len(self.image_path_list) - 1):
                self.next_button.state(["disabled"])
            else:
                self.next_button.state(["!disabled"])
        else: #3x3 grid
            if self.current_index == 0:
                self.back_button.state(["disabled"])
            else:
                self.back_button.state(["!disabled"])
            if self.current_index == (len(self.image_path_list) - 9):
                self.next_button.state(["disabled"])
            else:
                self.next_button.state(["!disabled"])
        #display single image
        self.single_image = ImageTk.PhotoImage(
            Image.open(self.image_path_list[self.current_index]))
        self.single_image_label.configure(image=self.single_image)
        #update image grid
        self.grid_images = []
        for i in range(9):
            if self.current_index + i >= len(self.image_path_list):
                break
            im = Image.open(self.image_path_list[self.current_index + i])
            im.thumbnail((420, 340), Image.Resampling.LANCZOS)
            self.grid_images.append(ImageTk.PhotoImage(im))
            self.grid_image_labels[i].configure(image=self.grid_images[i])

    def next(self):
        if self.notebook.tab(self.notebook.select(), "text")=='single':
            new_index = self.current_index + 1
        else: #3x3 grid
            new_index = self.current_index + 9
            if new_index > len(self.image_path_list):
                new_index = len(self.image_path_list) - 1
        self.change_index(new_index)

    def back(self):
        if self.notebook.tab(self.notebook.select(), "text")=='single':
            new_index = self.current_index - 1
        else: #3x3 grid
            new_index = self.current_index - 9
            if new_index < 0:
                new_index = 0
        self.change_index(new_index)

    def go_to(self):
        event_num = int(self.go_to_entry.get())
        uncut_index = self.run_data.get_index(event_num)
        search_string = f'image_{uncut_index}.jpg'
        for i, fname in enumerate(self.image_path_list):
            if search_string in fname:
                self.change_index(i)
                return
        messagebox.showwarning('Event not found!', 'Event not found!')

import os
import glob
import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
from PIL import Image, ImageTk
import re
import numpy as np

class ViewCutImagesWindow(tk.Toplevel):
    def __init__(self, parent, run_data, selected_dir, **kwargs):
        super().__init__(parent, **kwargs)
        self.run_data = run_data
        self.selected_dir = selected_dir

        # Load images
        image_search_string = os.path.join(selected_dir, "*.png")
        self.image_path_list = glob.glob(image_search_string)

        # Setup GUI components
        self.notebook = ttk.Notebook(self)
        self.back_button = ttk.Button(self, text='<<', command=self.back)
        self.next_button = ttk.Button(self, text='>>', command=self.next)
        self.go_to_entry = ttk.Entry(self)
        self.go_to_button = ttk.Button(self, text='Go to Image', command=self.go_to)
        self.go_to_good_events_entry = ttk.Entry(self)
        self.go_to_good_events_button = ttk.Button(self, text='Go to Image (Good Event Number)', command=self.go_to2)
        
        # Setup single image frame with a frame to include the file name
        self.single_image_frame = ttk.Frame(self.notebook)
        self.single_image_label = ttk.Label(self.single_image_frame, anchor=tk.CENTER)
        self.single_image_label.grid(row=0, column=0)
        self.single_filename_label = tk.Label(self.single_image_frame, text="", font=('Helvetica', 14))
        self.single_filename_label.grid(row=1, column=0)
        self.good_event_label = tk.Label(self.single_image_frame, text="", font=('Helvetica', 14))
        self.good_event_label.grid(row=2, column=0)  # Ensure this is placed correctly within your layout
        self.notebook.add(self.single_image_frame, text='single')

        # Setup grid for displaying multiple images
        self.grid_frame = ttk.Frame(self.notebook)
        self.grid_image_labels = []
        self.grid_filename_labels = []  # Labels for the filenames in the grid
        for i in range(9):
            image_label = ttk.Label(self.grid_frame)
            self.grid_image_labels.append(image_label)
            image_label.grid(row=int(i/3)*2, column=i%3)
            
            filename_label = tk.Label(self.grid_frame, text="", font=('Helvetica', 12))
            self.grid_filename_labels.append(filename_label)
            filename_label.grid(row=int(i/3)*2+1, column=i%3)
        self.notebook.add(self.grid_frame, text='3x3')

        # Positioning the main GUI components
        self.back_button.grid(row=0, column=0)
        self.next_button.grid(row=0, column=3)
        self.go_to_entry.grid(row=1, column=0)
        self.go_to_button.grid(row=1, column=1)
        self.go_to_good_events_entry.grid(row=1, column=2)
        self.go_to_good_events_button.grid(row=1, column=3)
        self.notebook.grid(row=2, column=0, columnspan=4)

        self.current_index = 0  # Track which image we're viewing
        self.change_index(self.current_index)

        # Bind the <ButtonRelease-1> event to handle tab changes
        self.notebook.bind("<ButtonRelease-1>", self.on_tab_change)


    def on_tab_change(self, event=None):
        self.change_index(self.current_index)


    def change_index(self, index):
        self.current_index = index
        # Enable/disable forward and backward buttons
        self.back_button.state(["!disabled" if index > 0 else "disabled"])
        self.next_button.state(["!disabled" if index < len(self.image_path_list) - 1 else "disabled"])

        single_image_size = (840, 680)  # New, larger size for the single image
        grid_image_size = (420, 340)  # Existing size for grid images

        if index < len(self.image_path_list):
            # Open the current image
            image_path = self.image_path_list[index]
            image = Image.open(image_path)

            # Adjust the image size based on the selected tab
            if self.notebook.tab(self.notebook.select(), "text") == 'single':
                # Resize while maintaining the aspect ratio
                aspect_ratio = min(single_image_size[0] / image.width, single_image_size[1] / image.height)
                new_size = (int(image.width * aspect_ratio), int(image.height * aspect_ratio))
                image = image.resize(new_size, Image.Resampling.LANCZOS)
            else:
                image.thumbnail(grid_image_size, Image.Resampling.LANCZOS)

            # Update the image display
            self.single_image = ImageTk.PhotoImage(image)
            self.single_image_label.configure(image=self.single_image)
            file_name = os.path.basename(image_path)
            self.single_filename_label.config(text=file_name)
            match = re.search(r'image_(\d+)\.png', file_name)
            number = int(match.group(1))
            self.good_event_label.config(text=f'Good Event Number: {self.run_data.good_events[number]}')

        # Update the 3x3 grid and filenames
        self.grid_images = []
        for i in range(9):
            if index + i < len(self.image_path_list):
                im = Image.open(self.image_path_list[index + i])
                im.thumbnail(grid_image_size, Image.Resampling.LANCZOS)
                photo_image = ImageTk.PhotoImage(im)
                self.grid_images.append(photo_image)
                self.grid_image_labels[i].configure(image=photo_image)
                file_name = os.path.basename(self.image_path_list[index + i])
                self.grid_filename_labels[i].config(text=file_name)
            else:
                self.grid_image_labels[i].configure(image='')
                self.grid_filename_labels[i].config(text='')


    def next(self):
        new_index = self.current_index + 1 if self.notebook.tab(self.notebook.select(), "text") == 'single' else self.current_index + 9
        new_index = min(new_index, len(self.image_path_list) - 1)
        self.change_index(new_index)


    def back(self):
        new_index = self.current_index - 1 if self.notebook.tab(self.notebook.select(), "text") == 'single' else self.current_index - 9
        new_index = max(new_index, 0)
        self.change_index(new_index)


    def go_to(self):
        try:
            event_num = int(self.go_to_entry.get())
            search_string = f'image_{event_num}.png'
            for i, fname in enumerate(self.image_path_list):
                if search_string in fname:
                    self.change_index(i)
                    return
            messagebox.showwarning('Event not found!', 'Event not found!')
        except ValueError:
            messagebox.showerror('Invalid Input', 'Please enter a valid number.')

    def go_to2(self):
        try:
            good_event_num = int(self.go_to_good_events_entry.get())
            event_num = np.where(self.run_data.good_events == good_event_num)[0][0]
            search_string = f'image_{event_num}.png'
            for i, fname in enumerate(self.image_path_list):
                if search_string in fname:
                    self.change_index(i)
                    return
            messagebox.showwarning('Event not found!', 'Event not found!')
        except ValueError:
            messagebox.showerror('Invalid Input', 'Please enter a valid number.')

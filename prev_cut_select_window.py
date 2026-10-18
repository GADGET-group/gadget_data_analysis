"""Browse saved run-specific image exports, including exports from several runs."""
import glob
import json
import os
import pickle
import re

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from PIL import Image, ImageTk


class _SavedImagesWindow(tk.Toplevel):
    def __init__(self, parent, directory):
        super().__init__(parent)
        self.files = sorted(glob.glob(os.path.join(directory, 'run*_event*.png')))
        if not self.files:
            # Also display images exported by the old filename convention.
            self.files = sorted(glob.glob(os.path.join(directory, '*.png')))
        self.index = 0
        self.label = ttk.Label(self)
        self.label.grid(row=1, column=0, columnspan=3)
        self.back_button = ttk.Button(self, text='<<', command=lambda: self.show(self.index - 1))
        self.back_button.grid(row=0, column=0)
        ttk.Button(self, text='Close', command=self.destroy).grid(row=0, column=1)
        self.next_button = ttk.Button(self, text='>>', command=lambda: self.show(self.index + 1))
        self.next_button.grid(row=0, column=2)
        if self.files:
            self.show(0)
        else:
            self.title('No saved event images')
            self.label.configure(text='No event PNGs found in this export folder.')
            self.back_button.state(['disabled'])
            self.next_button.state(['disabled'])

    def show(self, index):
        if not 0 <= index < len(self.files):
            return
        self.index = index
        with Image.open(self.files[index]) as image:
            image = image.convert('RGB')
            image.thumbnail((700, 700))
            self.photo = ImageTk.PhotoImage(image.copy())
        self.label.configure(image=self.photo)
        self.title(f'{index + 1}/{len(self.files)} — {os.path.basename(self.files[index])}')
        self.back_button.state(['disabled'] if index == 0 else ['!disabled'])
        self.next_button.state(['disabled'] if index == len(self.files) - 1 else ['!disabled'])


class PrevCutSelectWindow(tk.Toplevel):
    def __init__(self, parent, run_data, **kwargs):
        super().__init__(parent, **kwargs)
        self.run_data = run_data
        self.directory = run_data.folder_path
        self.current_image_index = 0
        self.image_path_list = []
        self.image_label = ttk.Label(self)
        self.image_label.grid(row=2, column=0, columnspan=4)
        self.back_button = ttk.Button(self, text='<<', command=self.back)
        self.back_button.grid(row=0, column=0)
        self.select_button = ttk.Button(self, text='View Event Images', command=self.select_cut)
        self.select_button.grid(row=0, column=1)
        self.forward_button = ttk.Button(self, text='>>', command=self.forward)
        self.forward_button.grid(row=0, column=2)
        ttk.Button(self, text='Close', command=self.destroy).grid(row=0, column=3)
        ttk.Button(self, text='Browse Export Folder', command=self.browse).grid(row=1, column=0, columnspan=2)
        self.create_h5_button = ttk.Button(self, text='Copy Selected H5 Datasets', command=self.create_h5)
        self.create_h5_button.grid(row=1, column=2, columnspan=2)
        self.reload()

    def browse(self):
        directory = filedialog.askdirectory(parent=self, initialdir=self.directory,
                                            title='Choose the parent of run/date export folders')
        if directory:
            self.directory = directory
            self.reload()

    def reload(self):
        self.image_path_list = sorted(glob.glob(os.path.join(self.directory, '*.jpg')))
        self.current_image_index = 0
        if self.image_path_list:
            self.select_button.state(['!disabled'])
            self.create_h5_button.state(['!disabled'])
            self.change_image_index_to(0)
        else:
            self.title('Previous exports')
            self.image_label.configure(image='', text='No saved cut previews in this folder.')
            for button in (self.back_button, self.forward_button, self.select_button, self.create_h5_button):
                button.state(['disabled'])

    def change_image_index_to(self, new_index):
        if not 0 <= new_index < len(self.image_path_list):
            return
        self.current_image_index = new_index
        filename = self.image_path_list[new_index]
        with Image.open(filename) as image:
            image = image.convert('RGB')
            image.thumbnail((850, 500))
            self.photo = ImageTk.PhotoImage(image.copy())
        self.image_label.configure(image=self.photo, text='')
        self.title(os.path.basename(filename)[:-4])
        self.back_button.state(['disabled'] if new_index == 0 else ['!disabled'])
        self.forward_button.state(['disabled'] if new_index == len(self.image_path_list) - 1 else ['!disabled'])

    def forward(self):
        self.change_image_index_to(self.current_image_index + 1)

    def back(self):
        self.change_image_index_to(self.current_image_index - 1)

    def select_cut(self):
        directory = self.image_path_list[self.current_image_index][:-4]
        _SavedImagesWindow(self, directory)

    def create_h5(self):
        """Preserve the legacy dataset-copy tool, routing it by the saved run.

        This produces a sparse subset for inspection, not a complete contiguous
        run suitable for process_tpc_run.
        """
        directory = self.image_path_list[self.current_image_index][:-4]
        destination = os.path.join(directory, 'h5_cut.h5')
        try:
            import h5py
            status_path = os.path.join(directory, 'export_status.json')
            if os.path.isfile(status_path):
                with open(status_path, encoding='utf-8') as handle:
                    metadata = json.load(handle)
                if metadata['experiment'] != self.run_data.experiment:
                    raise ValueError('Load the experiment associated with this saved export first.')
                run_num = int(metadata['run'])
            else:
                match = re.match(r'run(\d+)_', os.path.basename(directory))
                if match is None:
                    raise ValueError('The source run cannot be determined from this legacy folder.')
                run_num = int(match.group(1))
            source = self.run_data.get_run(run_num).h5_file
            with open(os.path.join(directory, 'cut_indices_H5list.pkl'), 'rb') as handle:
                event_numbers = pickle.load(handle)
            original = source.h5_file
            with h5py.File(destination, 'x') as new:
                new.attrs['experiment'] = self.run_data.experiment
                new.attrs['source_run'] = run_num
                new.attrs['sparse_subset'] = True
                new.create_dataset('selected_event_numbers', data=event_numbers)
                get = new.create_group('get')
                for event in event_numbers:
                    for kind in ('data', 'header'):
                        dataset = source._get_event_dataset(int(event), kind=kind)
                        original.copy(dataset, get, name=f'evt{int(event)}_{kind}')
                if 'clouds' in original:
                    clouds = new.create_group('clouds')
                    for event in event_numbers:
                        key = f'evt{int(event)}_cloud'
                        if key in original['clouds']:
                            original.copy(original['clouds'][key], clouds, name=key)
            messagebox.showinfo('H5 datasets copied', destination, parent=self)
        except Exception as exc:
            messagebox.showerror('Cannot copy H5 datasets', str(exc), parent=self)

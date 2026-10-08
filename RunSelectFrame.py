import os
from tkinter import ttk
import gadget_widgets
import GadgetRunH5
from raw_viewer import process_runs

import tkinter as tk
import tkinter.filedialog

class RunSelectFrame(ttk.Frame):
    def __init__(self, parent, main_gui):
        super().__init__(parent)
        self.main_gui = main_gui
        
        self.background_image = gadget_widgets.get_background_image()
        self.background = tk.Label(self, image=self.background_image)
        self.background.place(relx=0.5, rely=0.5, anchor='center')
        
        #create&place label frames
        self.load_run_frame = ttk.LabelFrame(self, text='load processed run')
        self.load_run_frame.grid(row=0)
        self.process_run_frame = ttk.LabelFrame(self, text='process run')
        self.process_run_frame.grid(row=1)

        #populate load run frame
        ttk.Label(self.load_run_frame, text="Experiment:").grid(row=0, column=0)
        self.experiment_entry = ttk.Entry(self.load_run_frame)
        self.experiment_entry.insert(0, 'e25058')
        self.experiment_entry.grid(row=0, column=1)
        ttk.Label(self.load_run_frame, text="Enter Run #:").grid(row=1, column=0)
        self.load_run_number_entry = ttk.Entry(self.load_run_frame) 
        self.load_run_number_entry.grid(row=1, column=1)
        self.load_button = ttk.Button(self.load_run_frame, text='Load', command=self.load_button_clicked)
        self.load_button.grid(row=2, column=0,columnspan=2)
        
        #populate process run frame
        ttk.Label(self.process_run_frame, text="Run #:").grid(row=0, column=0, sticky=tk.E)
        self.create_run_number_entry = ttk.Entry(self.process_run_frame) 
        self.create_run_number_entry.grid(row=0, column=1)
        self.process_run_button = ttk.Button(self.process_run_frame, text='Process run (force reprocess)',
                                             command=self.process_run_button_clicked)
        self.process_run_button.grid(row=1, column=0, columnspan=2)
        ttk.Label(self.process_run_frame,
                  text='runs raw_viewer.process_runs.process_tpc_run; the GUI is blocked until it finishes'
                  ).grid(row=2, column=0, columnspan=2)

        #member variable which will hold run data, once loaded.
        #will be a GadgetRunH5 object
        self.run_data = None

    def get_experiment(self):
        return self.experiment_entry.get().strip()

    def load_button_clicked(self):
        experiment = self.get_experiment()
        self.run_number = int(self.load_run_number_entry.get())
        default_path = process_runs.get_save_path(experiment) or os.getcwd()
        selected_path = tk.filedialog.askdirectory(initialdir=default_path,
                                                   title='Select output directory for cut images')
        if selected_path:
            self.run_data = GadgetRunH5.GadgetRunH5(self.run_number, selected_path, experiment=experiment)
            #TODO: make it possible to see which files and run were selected on the GUI
            self.main_gui.new_run_loaded()

    def process_run_button_clicked(self):
        experiment = self.get_experiment()
        run_num = int(self.create_run_number_entry.get())
        process_runs.process_tpc_run(experiment, run_num, force_reprocess=True)

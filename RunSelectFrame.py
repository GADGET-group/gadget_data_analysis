from tkinter import ttk, messagebox
import gadget_widgets
from GadgetRunCollection import GadgetRunCollection
from rve_cuts import parse_run_numbers
from raw_viewer import process_runs

import tkinter as tk

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
        ttk.Label(self.load_run_frame, text="Runs (comma-separated):").grid(row=1, column=0)
        self.load_run_number_entry = ttk.Entry(self.load_run_frame, width=30)
        self.load_run_number_entry.grid(row=1, column=1)
        self.load_button = ttk.Button(self.load_run_frame, text='Load', command=self.load_button_clicked)
        self.load_button.grid(row=2, column=0,columnspan=2)
        self.load_status = ttk.Label(self.load_run_frame, text='Example: 145,147,149')
        self.load_status.grid(row=3, column=0, columnspan=2)
        
        #populate process run frame
        ttk.Label(self.process_run_frame, text="Runs (comma-separated):").grid(row=0, column=0, sticky=tk.E)
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
        previous = self.run_data
        candidate = None
        try:
            runs = parse_run_numbers(self.load_run_number_entry.get())
            if not experiment:
                raise ValueError('Enter an experiment identifier.')
            self.load_button.state(['disabled'])
            self.load_status.configure(text=f'Loading runs {runs}...')
            self.update_idletasks()
            candidate = GadgetRunCollection(runs, experiment=experiment)
            self.run_data = candidate
            self.main_gui.new_run_loaded()
            self.run_numbers = runs
        except Exception as exc:
            self.run_data = previous
            if candidate is not None:
                candidate.close()
            self.load_status.configure(text='Load failed; previous runs are retained.')
            messagebox.showerror('Cannot load runs', str(exc), parent=self)
        else:
            if previous is not None:
                previous.close()
            self.load_status.configure(text=f'Loaded {candidate.runs_label}: {len(candidate.counts):,} events')
            print(f'Runs {candidate.runs_label} loaded; cut images: {candidate.folder_path}')
        finally:
            self.load_button.state(['!disabled'])

    def process_run_button_clicked(self):
        try:
            runs = parse_run_numbers(self.create_run_number_entry.get())
            experiment = self.get_experiment()
            for run_num in runs:
                process_runs.process_tpc_run(experiment, run_num, force_reprocess=True)
        except Exception as exc:
            messagebox.showerror('Run processing failed', str(exc), parent=self)

import os
import gadget_widgets
import subprocess
import socket

from tkinter import ttk
import tkinter as tk
from RunSelectFrame import RunSelectFrame
from EnergySpectrumFrame import EnergySpectrumFrame
from RvE_Frame import RvE_Frame
from IndividualEventFrame import IndividualEventFrame

import matplotlib.pyplot

class GadgetAnalysisGUI(tk.Tk):
    def __init__(self):
        super().__init__()

        bg_color = "#18453b"

        #set up window
        self.title('GADGET II Analysis')
        self.configure(background=bg_color)
        #self.geometry('800x600')
        #put title and version at the top of the window
        self.title_frame = tk.Frame(self, bg=bg_color)
        self.title_frame.grid(row=0, column=0, columnspan=2)
        self.title_label = tk.Label(self.title_frame, 
            text="GADGET II Analysis Gadget",bg=bg_color,
            fg="white", font = ('times','22'))
        self.title_label.grid(row=0)
        try:
            git_hash = subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD'],
                                               stderr=subprocess.DEVNULL).decode().strip()
        except (OSError, subprocess.CalledProcessError):
            git_hash = 'unavailable'
        self.version_label = tk.Label(self.title_frame, text='hash=%s'%git_hash,
                                        bg="#18453b", fg="white", font = ('times','12'))
        self.version_label.grid(row=1)

        self.notebook = ttk.Notebook(self, height=630, width=660)
        self.notebook.grid(row=2, column=0)
        
        self.run_select_frame = RunSelectFrame(self.notebook, self)
        self.notebook.add(self.run_select_frame, text='run')

        #list of tabs that should be updated each time a new run is loaded
        self.run_specific_tabs = []
    

    def new_run_loaded(self):
        '''
        function that will be called by RunSelectFrame anytime a new run is loaded
        '''
        # Carry cut geometry across run loads, independently of its free filters.
        previous_rve = next((tab for tab in self.run_specific_tabs
                             if isinstance(tab, RvE_Frame)), None)
        cut = previous_rve.get_cut_definition() if previous_rve and previous_rve.rve_cut_verticies else None
        run_data = self.run_select_frame.run_data
        if cut is not None and cut['experiment'] != run_data.experiment:
            cut = None
        new_tabs = []
        try:
            for frame_type in (EnergySpectrumFrame, RvE_Frame, IndividualEventFrame):
                frame = frame_type(self.notebook, run_data)
                new_tabs.append(frame)
                if isinstance(frame, RvE_Frame) and cut is not None:
                    frame.apply_cut_definition(cut)
        except Exception:
            for frame in new_tabs:
                frame.destroy()
            raise
        for tab in self.run_specific_tabs:
            if isinstance(tab, RvE_Frame):
                tab.close_plots()
            self.notebook.forget(tab)
            tab.destroy()
        self.run_specific_tabs = new_tabs
        for frame, label in zip(new_tabs, ('Energy Spectrum', 'Range vs Energy', 'Individual Events')):
            self.notebook.add(frame, text=label)
        self.title('GADGET II Analysis: Runs ' + run_data.runs_label)

#function to close all matplot windows, if this is run as main program
def on_closing():
    matplotlib.pyplot.close('all')
    data = gui.run_select_frame.run_data
    if data is not None:
        data.close()
    gui.destroy()

if __name__ == '__main__':
    gui = GadgetAnalysisGUI()
    gui.protocol("WM_DELETE_WINDOW", on_closing)
    gui.mainloop()


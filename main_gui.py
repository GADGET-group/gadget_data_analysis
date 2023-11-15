import os
import gadget_widgets

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
        self.title('GADGET II Analysis Gadget')
        if "nt" == os.name:
            self.wm_iconbitmap(bitmap = "/mnt/projects/e21072/OfflineAnalysis/backups/icons8-physics-48.ico")
        else:
            self.wm_iconbitmap(bitmap = "@/mnt/projects/e21072/OfflineAnalysis/backups/icons8-physics-48.xbm")
        self.configure(background=bg_color)
        #self.geometry('800x600')
        #put title and version at the top of the window
        self.title_frame = tk.Frame(self, bg=bg_color)
        self.title_frame.grid(row=0, column=0, columnspan=2)
        self.title_label = tk.Label(self.title_frame, 
            text="GADGET II Analysis Gadget",bg=bg_color,
            fg="white", font = ('times','22'))
        self.title_label.grid(row=0)
        self.version_label = tk.Label(self.title_frame, text="Version 4.23",
                                        bg="#18453b", fg="white", font = ('times','12'))
        self.version_label.grid(row=1)

        self.notebook = ttk.Notebook(self, height=500, width=500)
        self.notebook.grid(row=2, column=0)
        
        self.run_select_frame = RunSelectFrame(self.notebook, self)
        self.notebook.add(self.run_select_frame, text='run')

        #list of tabs that should be updated each time a new run is loaded
        self.run_specific_tabs = []
    

    def new_run_loaded(self):
        '''
        function that will be called by RunSelectFrame anytime a new run is loaded
        '''
        #remove tabs pertaining to previous run/s
        for tab in self.run_specific_tabs:
            self.notebook.forget(tab)
        #add tabls back pertaining to current run
        energy_spectrum_frame = EnergySpectrumFrame(self.notebook,self.run_select_frame.run_data)
        self.run_specific_tabs.append(energy_spectrum_frame)
        self.notebook.add(energy_spectrum_frame, text='Energy Spectrum')
        rve_frame = RvE_Frame(self.notebook, self.run_select_frame.run_data)
        self.run_specific_tabs.append(rve_frame)
        self.notebook.add(rve_frame, text='Range vs Energy')
        single_event_frame = IndividualEventFrame(self.notebook, self.run_select_frame.run_data)
        self.notebook.add(single_event_frame, text='Individual Events')

#function to close all matplot windows, if this is run as main program
def on_closing():
    matplotlib.pyplot.close('all')
    gui.destroy()

if __name__ == '__main__':
    gui = GadgetAnalysisGUI()
    gui.protocol("WM_DELETE_WINDOW", on_closing)
    gui.mainloop()


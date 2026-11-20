'''
GADGET II analysis GUI: python -m gadget_analysis.main_gui (from any directory; needs an X display).
'''
import os
import subprocess
import traceback
import importlib.metadata

from tkinter import ttk
import tkinter as tk
import tkinter.messagebox

import matplotlib.pyplot

from gadget_analysis.main_gui.RunSelectFrame import RunSelectFrame
from gadget_analysis.main_gui.EnergySpectrumFrame import EnergySpectrumFrame
from gadget_analysis.main_gui.RvE_Frame import RvE_Frame
from gadget_analysis.main_gui.IndividualEventFrame import IndividualEventFrame

def get_library_version():
    '''
    Commit of the library checkout the GUI runs from ("-dirty" when tracked files are modified), whatever the working
    directory; the installed version for a non-git install.
    '''
    package_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    try:
        def git(*args):
            return subprocess.run(['git', *args], cwd=package_dir, capture_output=True, text=True, check=True, timeout=30).stdout
        return git('rev-parse', 'HEAD').strip() + ('-dirty' if git('status', '--porcelain', '--untracked-files=no').strip() else '')
    except (OSError, subprocess.SubprocessError):
        try:
            return 'version ' + importlib.metadata.version('gadget_analysis')
        except importlib.metadata.PackageNotFoundError:
            return 'unknown version'

class GadgetAnalysisGUI(tk.Tk):
    def __init__(self):
        super().__init__()

        bg_color = "#18453b"

        #set up window
        self.title('GADGET II Analysis')
        self.configure(background=bg_color)
        #put title and version at the top of the window
        self.title_frame = tk.Frame(self, bg=bg_color)
        self.title_frame.grid(row=0, column=0, columnspan=2)
        self.title_label = tk.Label(self.title_frame,
            text="GADGET II Analysis Gadget",bg=bg_color,
            fg="white", font = ('times','22'))
        self.title_label.grid(row=0)
        self.version_label = tk.Label(self.title_frame, text='hash=%s'%get_library_version(),
                                        bg=bg_color, fg="white", font = ('times','12'))
        self.version_label.grid(row=1)

        self.notebook = ttk.Notebook(self, height=500, width=500)
        self.notebook.grid(row=2, column=0)

        self.run_select_frame = RunSelectFrame(self.notebook, self)
        self.notebook.add(self.run_select_frame, text='run')

        #list of tabs that should be updated each time a new run is loaded
        self.run_specific_tabs = []
        self.protocol("WM_DELETE_WINDOW", self.on_closing)

    def report_callback_exception(self, exc, val, tb):
        '''Errors in button callbacks: traceback in the terminal and a message box, instead of the terminal only.'''
        traceback.print_exception(exc, val, tb)
        tk.messagebox.showerror('GADGET II Analysis', f'{exc.__name__}: {val}')

    def new_run_loaded(self):
        '''
        function that will be called by RunSelectFrame anytime a new run is loaded
        '''
        #remove tabs pertaining to previous run/s
        for tab in self.run_specific_tabs:
            self.notebook.forget(tab)
            tab.destroy()
        self.run_specific_tabs = []
        #add tabs back pertaining to current run
        run_data = self.run_select_frame.run_data
        for frame_class, text in ((EnergySpectrumFrame, 'Energy Spectrum'), (RvE_Frame, 'Range vs Energy'),
                                  (IndividualEventFrame, 'Individual Events')):
            tab = frame_class(self.notebook, run_data)
            self.notebook.add(tab, text=text)
            self.run_specific_tabs.append(tab)
        self.title(f'GADGET II Analysis: {run_data.experiment} Run {run_data.run_num} ({run_data.config_filename})')

    def on_closing(self):
        '''Close all matplotlib windows with the GUI.'''
        matplotlib.pyplot.close('all')
        self.destroy()

if __name__ == '__main__':
    gui = GadgetAnalysisGUI()
    gui.mainloop()

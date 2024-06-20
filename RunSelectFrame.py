from tkinter import ttk
import gadget_widgets
import GadgetRunH5

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
        self.load_run_frame = ttk.LabelFrame(self, text='load existing files')
        self.load_run_frame.grid(row=0)
        self.create_files_frame = ttk.LabelFrame(self, text='Create Files')
        self.create_files_frame.grid(row=1)

        #populate load existing files frame
        ttk.Label(self.load_run_frame, text="Enter Run #:").grid(row=0, column=0)
        self.load_run_number_entry = ttk.Entry(self.load_run_frame) 
        self.load_run_number_entry.grid(row=0, column=1)
        self.load_button = ttk.Button(self.load_run_frame, text='Load', command=self.load_button_clicked)
        self.load_button.grid(row=1, column=0,columnspan=2)
        
        #populate files creation frame
        current_row=0
        tk.Label(self.create_files_frame, text="Run #:"
                 ).grid(row=current_row, column=0, sticky=tk.E)
        self.create_run_number_entry = ttk.Entry(self.create_files_frame) 
        self.create_run_number_entry.grid(row=current_row, column=1)
        current_row +=1

        ttk.Label(self.create_files_frame, text='Length Veto'
                  ).grid(row=current_row, column=0, sticky=tk.E)
        self.length_veto_entry = ttk.Entry(self.create_files_frame)
        self.length_veto_entry.insert(0,80)
        self.length_veto_entry.grid(row=current_row, column=1)
        ttk.Label(self.create_files_frame, text='mm (max)').grid(row=current_row, column=2)
        current_row += 1
        
        ttk.Label(self.create_files_frame, text='Integrated Charge Veto'
                  ).grid(row=current_row, column=0, sticky=tk.E)
        self.ic_veto_entry = ttk.Entry(self.create_files_frame)
        self.ic_veto_entry.insert(0,8)
        self.ic_veto_entry.grid(row=current_row, column=1)
        #the old gui label said x10e5 (eg 1e6), but this was wrong
        ttk.Label(self.create_files_frame, text='x10^5').grid(row=current_row, column=2)
        current_row += 1

        ttk.Label(self.create_files_frame, text='Points Threshold'
                  ).grid(row=current_row, column=0, sticky=tk.E)
        self.point_entry = ttk.Entry(self.create_files_frame)
        self.point_entry.insert(0, 21)
        self.point_entry.grid(row=current_row, column=1)
        current_row += 1

        ttk.Label(self.create_files_frame, text='DBSCAN Param \"eps\"'
                  ).grid(row=current_row, column=0, sticky=tk.E)
        self.eps_entry = ttk.Entry(self.create_files_frame)
        self.eps_entry.insert(0, 7)
        self.eps_entry.grid(row=current_row, column=1)
        current_row += 1

        ttk.Label(self.create_files_frame, text='DBSCAN Param \"min samples\"'
                  ).grid(row=current_row, column=0, sticky=tk.E)
        self.min_samples_entry = ttk.Entry(self.create_files_frame)
        self.min_samples_entry.insert(0,8)
        self.min_samples_entry.grid(row=current_row, column=1)
        current_row += 1

        ttk.Label(self.create_files_frame, text='IMOD-Poly param \"poly degree\"'
                  ).grid(row=current_row, column=0, sticky=tk.E)
        self.poly_entry = ttk.Entry(self.create_files_frame)
        self.poly_entry.insert(0, 2)
        self.poly_entry.grid(row=current_row, column=1)
        current_row += 1

        self.create_files_button = ttk.Button(self.create_files_frame, text='Create Files',
                                               command=self.create_files_button_clicked)
        self.create_files_button.grid(row=current_row, column=0, columnspan=3)
        current_row += 1

        #member variable which will hold run data, once loaded.
        #will be a GadgetRunH5 object
        self.run_data = None

    def load_button_clicked(self):
        self.run_number = self.load_run_number_entry.get()
        if 's' not in self.run_number:
            self.run_number = int(self.run_number)
        default_path = GadgetRunH5.get_default_path(self.run_number)
        selected_path = tk.filedialog.askdirectory(initialdir=default_path, title='Select a Directory')
        if selected_path:
            self.run_data = GadgetRunH5.GadgetRunH5(self.run_number, selected_path)
            #TODO: make it possible to see which files and run were selected on the GUI
            self.main_gui.new_run_loaded()

    def create_files_button_clicked(self):
        #TODO: handle case where files already exist
        length = int(self.length_veto_entry.get())
        ic = int(self.ic_veto_entry.get())*int(1e5)
        points = int(self.point_entry.get())
        eps = int(self.eps_entry.get())
        samps = int(self.min_samples_entry.get())
        poly = int(self.poly_entry.get())
        run_num = (self.create_run_number_entry.get())
        if 's' not in run_num:
            run_num = int(run_num)
            
        GadgetRunH5.generate_files(run_num, length, ic, points, eps, samps, poly)
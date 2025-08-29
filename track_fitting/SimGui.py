import numpy as np
import tkinter as tk
from tkinter import ttk

import  numpy as np
import matplotlib.pylab as plt
import scipy.optimize as opt

from track_fitting.SimulatedEvent import SimulatedEvent
from track_fitting. MultiParticleEvent import MultiParticleEvent

class SimGui(ttk.Frame):
    def __init__(self, parent, sim:SimulatedEvent, expose_arrays={'initial_point':float}):
        '''
        sim: Simulation to adjust parameters of
        expose_arrays: Dict of arrays which should be visbible in the gui. Keys are variable name, and index the 
                        type that values should be case as when updating the array from the GUI. These arrays
                        are currently assumed not to change length.
        '''
        super().__init__(parent)
        self.sim = sim
        sim.gui_after_sim()

        param_frame = ttk.LabelFrame(self, text='simulation parameters')
        row = 0
        #Iterable objects are only exposed if in "expose_iterables"
        self.array_types = expose_arrays
        self.array_entries = {}
        self.array_fit_variables = {}
        for array_name in expose_arrays:
            array = sim.__dict__[array_name]
            ttk.Label(param_frame, text=array_name).grid(row=row, column=0)
            col = 1
            entries = []
            self.array_fit_variables[array_name] = []
            for x in array:
                new_entry = ttk.Entry(param_frame)
                new_entry.insert(0, str(x))
                new_entry.grid(row=row, column=col)
                entries.append(new_entry)
                check_var = tk.BooleanVar()
                tk.Checkbutton(param_frame, variable=check_var).grid(row=row+1, column=col)
                self.array_fit_variables[array_name].append(check_var)
                col += 1
            row += 2
            self.array_entries[array_name] = entries

        #expose all ints, floats, and strings
        self.param_names = []
        self.param_types = []
        self.param_entries = []
        self.plottable_vars = []
        self.single_var_fit_variables = []

        for var_name in sim.__dict__:
            value = sim.__dict__[var_name]
            if isinstance(value, np.floating) or isinstance(value, np.integer) or isinstance(value, int) or isinstance(value, float) or isinstance(value, str   ):
                self.param_names.append(var_name)
                self.param_types.append(type(value))
                new_entry = ttk.Entry(param_frame)
                self.param_entries.append(new_entry)
                new_entry.insert(0, str(value))
                ttk.Label(param_frame, text=var_name).grid(row=row, column=0)
                new_entry.grid(row=row, column=1)
                ttk.Label(param_frame, text=str(type(value))).grid(row=row, column=2)
                check_var = tk.BooleanVar()
                tk.Checkbutton(param_frame, variable=check_var).grid(row=row, column=3)
                self.single_var_fit_variables.append(check_var)
                row += 1
            if (isinstance(value, np.floating) or isinstance(value, np.integer) or isinstance(value, int) or isinstance(value, float)) and type(value) != bool:
                self.plottable_vars.append(var_name) 
        param_frame.grid()



        #simulate button 
        sim_frame = ttk.LabelFrame(self, text='simulate')
        row = 0
        ttk.Button(sim_frame, text='simulate', command=self.sim_button_clicked).grid(row=row, column=0)
        ttk.Button(sim_frame, text='maximize likelihood', command=self.maximize_likelihood).grid(row=row, column=1)
        ttk.Button(sim_frame, text='save params', command=self.save_params_clicked).grid(row=row, column = 2)
        ttk.Button(sim_frame, text='load params', command=self.load_params_clicked).grid(row=row, column = 3)
        row += 1
        ttk.Label(sim_frame, text='max iter:').grid(row=row, column=0)
        self.max_iter_entry = ttk.Entry(sim_frame)
        self.max_iter_entry.insert(0, '5')
        self.max_iter_entry.grid(row=row, column=1)
        row += 1
        ttk.Label(sim_frame, text='view threshold:').grid(row=row, column=0)
        self.view_thresh_entry = ttk.Entry(sim_frame)
        self.view_thresh_entry.insert(0, '20')
        self.view_thresh_entry.grid(row=row, column=1)
        row += 1
        
        sim_frame.grid()
        
        

        likelihood_plot_frame = ttk.LabelFrame(self, text='Log Likelihood Plotting')
        row = 0
        ttk.Label(likelihood_plot_frame, text='current log likelihood:').grid(row=row, column=0)
        self.likelihood_label = ttk.Label(likelihood_plot_frame, text='---------')
        self.likelihood_label.grid(row=row, column=1)
        row += 1
        self.independent_plot_var = tk.StringVar(self)
        ttk.Label(likelihood_plot_frame, text='independent variable:').grid(row=row, column=0)
        ttk.OptionMenu(likelihood_plot_frame, self.independent_plot_var, self.plottable_vars[0], *self.plottable_vars).grid(row=row, column=1)
        self.thing_to_plot_var = tk.StringVar(self)
        row += 1
        ttk.Label(likelihood_plot_frame, text="+").grid(row=row, column=0)
        self.ll_plot_plus_entry = ttk.Entry(likelihood_plot_frame)
        self.ll_plot_plus_entry.insert(0, '1')
        self.ll_plot_plus_entry.grid(row=row, column=1)
        ttk.Label(likelihood_plot_frame, text="-").grid(row=row, column=2)
        self.ll_plot_minus_entry = ttk.Entry(likelihood_plot_frame)
        self.ll_plot_minus_entry.insert(0, '1')
        self.ll_plot_minus_entry.grid(row=row, column=3)
        row += 1
        ttk.Label(likelihood_plot_frame, text='Number of points:').grid(row=row, column=0)
        self.ll_plot_points_entry = ttk.Entry(likelihood_plot_frame)
        self.ll_plot_points_entry.insert(0, '9')
        self.ll_plot_points_entry.grid(row=row, column=1)
        row += 1
        ttk.Button(likelihood_plot_frame, text='plot log likelihood', command=self.ll_plot_clicked).grid(row=row, column=0)
        ttk.Button(likelihood_plot_frame, text='plot X^2', command=self.chi_squared_plot_clicked).grid(row=row, column=1)
        likelihood_plot_frame.grid()

        self.saved_entry_vals = [] #strings to repopulate entries when load is clicked
        self.save_params_clicked()

    def load_entries_to_sim(self):
        #set individual variables
        for name, entry, param_type in zip(self.param_names, self.param_entries, self.param_types):
            if param_type != bool:
                self.sim.__dict__[name] = param_type(entry.get())
            else: #handle boolean variables
                if entry.get().lower() == 'false':
                    self.sim.__dict__[name] = False
                elif entry.get().lower() == 'true':
                    self.sim.__dict__[name] = True
                else:
                    assert False
        #set array element values
        for array_name in self.array_types:
            array_type = self.array_types[array_name]
            entries = self.array_entries[array_name]
            self.sim.__dict__[array_name]= [array_type(entries[i].get()) for i in range(len(entries))]
        #reload srim table to match values set through gui, and then resimulate event
        self.sim.gui_before_sim()
        self.sim.simulate_event()
        self.sim.gui_after_sim()

    def update_entries_to_reflect_sim(self):
        #self.sim.gui_after_sim()
        for name, entry in zip(self.param_names, self.param_entries):
            entry.delete(0, tk.END)
            entry.insert(0, str(self.sim.__dict__[name]))

    def sim_button_clicked(self):
        self.load_entries_to_sim()
        #make plots
        view_thresh = float(self.view_thresh_entry.get())
        self.sim.plot_real_data_3d(threshold=view_thresh)
        self.sim.plot_simulated_3d_data(threshold=view_thresh)
        self.sim.plot_residuals_3d(threshold=view_thresh)
        plt.show(block=False)

        self.likelihood_label['text'] = '%e'%self.sim.log_likelihood()
        self.update_entries_to_reflect_sim()

    def ll_plot_clicked(self):
        self.load_entries_to_sim() #make sure sim is up to date with entries
        var_to_plot = self.independent_plot_var.get()
        p, m = float(self.ll_plot_plus_entry.get()), float(self.ll_plot_minus_entry.get())
        current_val = self.sim.__dict__[var_to_plot]
        vals_to_plot = np.linspace(current_val - m, current_val + p, int(self.ll_plot_points_entry.get()))
        ll_vals = []
        for v in vals_to_plot:
            self.sim.__dict__[var_to_plot] = v
            self.sim.load_srim_table(self.sim.particle, self.sim.gas_density)
            self.sim.simulate_event()
            ll_vals.append(self.sim.log_likelihood())
        plt.figure()
        plt.scatter(vals_to_plot, ll_vals)
        plt.xlabel(var_to_plot)
        plt.ylabel('log likilihood')
        plt.show(block=False)

    def chi_squared_plot_clicked(self):
        self.load_entries_to_sim() #make sure sim is up to date with entries
        var_to_plot = self.independent_plot_var.get()
        p, m = float(self.ll_plot_plus_entry.get()), float(self.ll_plot_minus_entry.get())
        current_val = self.sim.__dict__[var_to_plot]
        vals_to_plot = np.linspace(current_val - m, current_val + p, int(self.ll_plot_points_entry.get()))
        ll_vals = []
        for v in vals_to_plot:
            self.sim.__dict__[var_to_plot] = v
            self.sim.load_srim_table(self.sim.particle, self.sim.gas_density)
            self.sim.simulate_event()
            residuals = self.sim.get_residuals()
            residuals = np.array([residuals[p] for p in residuals])
            ll_vals.append(np.sum(residuals*residuals))
        plt.figure()
        plt.scatter(vals_to_plot, ll_vals)
        plt.xlabel(var_to_plot)
        plt.ylabel('sum of residuals squared')
        plt.show(block=False)
    

    def save_params_clicked(self):
        self.saved_entry_vals = []
        for entry in self.param_entries:
            self.saved_entry_vals.append(entry.get())

    def load_params_clicked(self):
        for saved_val, entry in zip(self.saved_entry_vals, self.param_entries):
             entry.delete(0, tk.END)
             entry.insert(0, saved_val)

    def maximize_likelihood(self):
        entries_to_fit = []
        for array_name in self.array_entries:
            for entry, var in zip(self.array_entries[array_name], self.array_fit_variables[array_name]):
                if var.get():
                    entries_to_fit.append(entry)
        for entry, var in zip(self.param_entries, self.single_var_fit_variables):
            if var.get():
                entries_to_fit.append(entry)
        
        def to_minimize(vals):
            for v, entry in zip(vals, entries_to_fit):
                entry.delete(0, tk.END)
                entry.insert(0, v)
            self.load_entries_to_sim()
            to_return = -self.sim.log_likelihood()
            if np.isnan(to_return):
                to_return = np.inf
            #print(vals, to_return)
            return to_return
        def callback(intermediate_result:opt.OptimizeResult):
            print(intermediate_result)

        starting_guess = [float(entry.get()) for entry in entries_to_fit]
        print(opt.minimize(to_minimize, starting_guess, options={'maxiter':float(self.max_iter_entry.get())}, callback=callback))
        self.update_entries_to_reflect_sim()
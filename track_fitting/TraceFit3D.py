import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
import matplotlib.colors
import numpy as np

from track_fitting import SingleParticleEvent

class TraceFit3D(ttk.Frame):
    def __init__(self, parent):
        super().__init__(parent)
        pressure = 800
        self.trace_function = SingleParticleEvent.SingleParticleEvent(pressure, 'proton')
        self.create_widgets()

    def create_widgets(self):
        # Create labels and entry fields for the parameters
        parameters = [
            ("Initial Energy (MeV)", "initial_energy"),
            ("Initial X (mm)", "initial_x"),
            ("Initial Y (mm)", "initial_y"),
            ("Initial Z (mm)", "initial_z"),
            ("Theta angle (radians)", "theta"),
            ("Phi angle (radians)", "phi")
        ]
        
        self.entries = {}
        
        for i, (label_text, var_name) in enumerate(parameters):
            label = ttk.Label(self, text=label_text)
            label.grid(row=i, column=0, padx=10, pady=5, sticky=tk.W)
            
            entry = ttk.Entry(self)
            entry.grid(row=i, column=1, padx=10, pady=5, sticky=tk.W)
            
            self.entries[var_name] = entry
        
        # Add a combobox for selecting particle type
        particle_label = ttk.Label(self, text="Particle Type")
        particle_label.grid(row=len(parameters), column=0, padx=10, pady=5, sticky=tk.W)
        
        self.particle_type = tk.StringVar()
        self.particle_combobox = ttk.Combobox(self, textvariable=self.particle_type, values=["Proton", "Alpha"])
        self.particle_combobox.grid(row=len(parameters), column=1, padx=10, pady=5, sticky=tk.W)
        self.particle_combobox.current(0)  # Set default to "Proton"
        
        # Create a button to plot the data
        self.plot_button = ttk.Button(self, text="Plot", command=self.plot_data)
        self.plot_button.grid(row=len(parameters) + 1, columnspan=2, pady=10)

    def load_parameters(self):
        try:
            # Get the values from the entry fields and store them as instance variables
            self.trace_function.initial_energy = float(self.entries["initial_energy"].get())
            self.trace_function.initial_point = (float(self.entries["initial_x"].get()), float(self.entries["initial_y"].get()), float(self.entries["initial_z"].get()))
            self.trace_function.theta = float(self.entries["theta"].get())
            self.trace_function.phi = float(self.entries["phi"].get())
            self.trace_function.load_srim_table(self.particle_type.get(), 1.57)
        except ValueError:
            print("Please enter valid numbers for all parameters.")

    def plot_data(self):
        self.load_parameters()
        self.trace_function.simulate_event()
        x, y, z, e = self.trace_function.get_xyze(threshold=0.001)
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Plot the 3D scatter plot with energy values as color
        sc = ax.scatter(x, y, z, c=e,  alpha=0.5)#, cmap='inferno', marker='o',)# norm=matplotlib.colors.LogNorm())
        
        # Add colorbar
        cbar = plt.colorbar(sc, ax=ax)
        cbar.set_label('Energy (a.u.)')
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        ax.axes.set_xlim3d(left=-100, right=100) 
        ax.axes.set_ylim3d(bottom=-100, top=100) 
        ax.axes.set_zlim3d(bottom=0, top=200)
        
        plt.title(f'3D Energy Distribution of {self.trace_function.initial_energy} MeV {self.trace_function.particle}, θ = {self.trace_function.theta} rad, φ = {self.trace_function.phi} rad')
        plt.show()

    
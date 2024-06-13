import tkinter as tk
from tkinter import ttk
import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
from scipy.ndimage import convolve

class TraceFit3D(ttk.Frame):
    def __init__(self, parent, data_path):
        super().__init__(parent)
        self.gas_density = 1.56 #mg/cm^3
        self.energies, self.stopping_powers = self.load_energies(data_path) #MeV, MeV/mm
        self.z_scale = 1.74
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
        
        # Create a button to plot the data
        self.plot_button = ttk.Button(self, text="Plot", command=self.plot_data)
        self.plot_button.grid(row=len(parameters), columnspan=2, pady=10)

    def load_parameters(self):
        try:
            # Get the values from the entry fields and store them as instance variables
            self.initial_energy = float(self.entries["initial_energy"].get())
            self.initial_point = (float(self.entries["initial_x"].get()), float(self.entries["initial_y"].get()), float(self.entries["initial_z"].get()))
            self.theta = float(self.entries["theta"].get())
            self.phi = float(self.entries["phi"].get())
            
            print(f"Initial Energy: {self.initial_energy}")
            print(f"Theta: {self.theta}")
            print(f"Phi: {self.phi}")

        except ValueError:
            print("Please enter valid numbers for all parameters.")

    def plot_data(self):

        # Load Prameters for the Plot
        self.load_parameters()
        self.sigma_xy = 2
        self.sigma_z = 2
        self.kernel_size = 6

        # Make distances where stopping_power is to be calculated
        distances = np.linspace(0,100,500)

        def dEdx(E):
            """
            Differential function to calculate stopping power based on energy.
            """
            to_return = -np.interp(E, self.energies, self.stopping_powers)
            to_return[E < 0] = 0
            return to_return

        # Find energy at each distance by integrating
        Es = np.squeeze(integrate.odeint(lambda E, x: dEdx(E), self.initial_energy, distances))
        
        # Calculate stopping powers
        calculated_stopping_powers = -dEdx(Es)*100/500

        # Find Points where energy is evaluated
        points = self.compute_points(distances)
        energy_map = self.create_grid(points, calculated_stopping_powers)
        convolved_energies = self.convolve_with_gaussian(energy_map)
        self.display_image(convolved_energies)

    def compute_points(self, distances):
        """
        Compute 3D points along a given direction from an initial point based on distances.
        """
        direction_vector = np.round(np.array((np.sin(self.theta) * np.cos(self.phi), np.sin(self.theta) * np.sin(self.phi), np.cos(self.theta))), 6)
        points = np.array([np.array(self.initial_point) + distance * direction_vector for distance in distances])
        return points
    
    def create_grid(self, coords, stopping_powers, grid_resolution=0.5):
        """
        Create a fine grid for the given coordinate range and resolution.
        grid_resolution (fin mm): Resolution of the grid.
        """
        min_coord = np.min(coords)
        max_coord = np.max(coords)
        self.x = np.arange(min_coord - 10, max_coord + grid_resolution + 10, grid_resolution)
        self.y = np.arange(min_coord - 10, max_coord + grid_resolution + 10, grid_resolution)
        self.z = np.arange(min_coord - 10, max_coord + grid_resolution + 10, grid_resolution)
        X, Y, Z = np.meshgrid(self.x, self.y, self.z, indexing='ij')
        grid_shape = X.shape
        energy_grid = np.zeros(grid_shape)
        for point, energy in zip(coords, stopping_powers):
            # Find the nearest grid coordinates
            x_idx = int(np.round((point[0] - min_coord + 10) / grid_resolution))
            y_idx = int(np.round((point[1] - min_coord + 10) / grid_resolution))
            z_idx = int(np.round((point[2] - min_coord + 10) / grid_resolution))
            energy_grid[x_idx, y_idx, z_idx] += energy

        return energy_grid
    
    def convolve_with_gaussian(self, energy_grid):
        """
        Convolve the energy grid with a 3D Gaussian kernel.
        """

        def gaussian_kernel_3d(kernel_size, sigma, sigma_z):
            """Create a 3D Gaussian kernel."""
            ax = np.linspace(-(kernel_size - 1) / 2., (kernel_size - 1) / 2., kernel_size)
            xx, yy, zz = np.meshgrid(ax, ax, ax)
            kernel = np.exp(-((xx**2 + yy**2) / (2 * sigma**2) + zz**2 / (2 * sigma_z**2)))
            return kernel / np.sum(kernel)
        
        kernel = gaussian_kernel_3d(self.kernel_size, self.sigma_xy, self.sigma_z)
        convolved_grid = convolve(energy_grid, kernel, mode='constant', cval=0)
        return convolved_grid
    
    def display_image(self, energy_grid):
        """
        Map to the Padplanes and Plot 
        Plot the 3D energy distribution.
        """
        # Creating PadPlane positions
        pad_x = np.arange(-38.5,38.5+2.2,2.2)
        pad_y = np.arange(-38.5,38.5+2.2,2.2)
        pad_z = np.arange(0,512*self.z_scale,self.z_scale)

        # Get the indices where energy is not zero for more efficient plotting
        non_zero_indices = np.nonzero(energy_grid)
        x_non_zero = self.x[non_zero_indices[0]]
        y_non_zero = self.y[non_zero_indices[1]]
        z_non_zero = self.z[non_zero_indices[2]]
        energy_non_zero = energy_grid[non_zero_indices]
        pad_energy_dict = {}

        for i, j, k, e in zip(x_non_zero, y_non_zero, z_non_zero, energy_non_zero):
            # Map x and y to the nearest padplane positions
            pad_x_nearest = pad_x[np.abs(pad_x - i).argmin()]
            pad_y_nearest = pad_y[np.abs(pad_y - j).argmin()]
            pad_z_nearest = pad_z[np.abs(pad_z - k).argmin()]
            if (pad_x_nearest, pad_y_nearest, pad_z_nearest) not in pad_energy_dict:
                pad_energy_dict[(pad_x_nearest, pad_y_nearest, pad_z_nearest)] = e
            else:
                pad_energy_dict[(pad_x_nearest, pad_y_nearest, pad_z_nearest)] += e

        # Extract coordinates and energies from the dictionary
        pad_coords = np.array(list(pad_energy_dict.keys()))
        pad_energies = np.array(list(pad_energy_dict.values()))
        pad_x_coords = pad_coords[:, 0]
        pad_y_coords = pad_coords[:, 1]
        pad_z_coords = pad_coords[:, 2]

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Plot the 3D scatter plot with energy values as color
        sc = ax.scatter(pad_x_coords, pad_y_coords, pad_z_coords, c=pad_energies, cmap='viridis', marker='o', alpha=0.5)
        
        # Add colorbar
        cbar = plt.colorbar(sc, ax=ax)
        cbar.set_label('Energy (a.u.)')
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        ax.axes.set_xlim3d(left=-40, right=40) 
        ax.axes.set_ylim3d(bottom=-40, top=40) 
        ax.axes.set_zlim3d(bottom=0, top=100)
        
        plt.title(f'3D Energy Distribution of {self.initial_energy} MeV proton, θ = {self.theta} rad, φ = {self.phi} rad')
        plt.show()

    def load_energies(self, data_path):
        # Initialize lists to store the data
        energy_MeV = []
        electronic_stopping_MeV_um = []  # Stopping power in MeV/(mg/cm^2)
        nuclear_stopping_MeV_um = []
        path_length_mm = []

        # Read the file
        with open(data_path, 'r') as file:
            for i, line in enumerate(file):
                # Start reading from the 26th line and stop after the 105th line
                if 26 <= i <= 104:
                    if line.strip():  # Ensure the line is not empty
                        parts = line.split()
                        # Handle energy conversion based on unit
                        energy_value = float(parts[0])
                        energy_unit = parts[1]
                        if energy_unit == 'keV':
                            energy = energy_value / 1000  # Convert from keV to MeV
                        elif energy_unit == 'MeV':
                            energy = energy_value
                        
                        # Stopping powers are in MeV/(mg/cm^2)
                        elec = float(parts[2])
                        nucl = float(parts[3])

                        # Convert path length to mm
                        path_value = float(parts[4])  # Get the number part before the unit
                        path_unit = parts[5]         # Get the unit part

                        if path_unit == 'um':
                            path = path_value / 1000  # Convert from um to mm
                        elif path_unit == 'mm':
                            path = path_value
                        elif path_unit == 'm':
                            path = path_value * 1000  # Convert from m to mm
                        
                        # Append to lists
                        energy_MeV.append(energy)
                        electronic_stopping_MeV_um.append(elec)
                        nuclear_stopping_MeV_um.append(nucl)
                        path_length_mm.append(path)

        # Convert lists to NumPy arrays
        energy_MeV = np.array(energy_MeV)
        electronic_stopping_MeV_um = np.array(electronic_stopping_MeV_um)
        nuclear_stopping_MeV_um = np.array(nuclear_stopping_MeV_um)
        path_length_mm = np.array(path_length_mm)
        stopping_power_converted = self.gas_density * (electronic_stopping_MeV_um + nuclear_stopping_MeV_um)/10
        return energy_MeV, stopping_power_converted
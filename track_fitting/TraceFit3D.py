import tkinter as tk
from tkinter import ttk
import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
from scipy.signal import convolve
import time 

class TraceFit3D(ttk.Frame):
    def __init__(self, parent):
        super().__init__(parent)
        self.gas_density = 1.56 #mg/cm^3
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
        
        # Add a combobox for selecting particle type
        particle_label = ttk.Label(self, text="Particle Type")
        particle_label.grid(row=len(parameters), column=0, padx=10, pady=5, sticky=tk.W)
        
        self.particle_type = tk.StringVar()
        self.particle_combobox = ttk.Combobox(self, textvariable=self.particle_type, values=["Proton", "Alpha"])
        self.particle_combobox.grid(row=len(parameters), column=1, padx=10, pady=5, sticky=tk.W)
        self.particle_combobox.current(0)  # Set default to "Proton"
        print(self.particle_type)
        
        # Create a button to plot the data
        self.plot_button = ttk.Button(self, text="Plot", command=self.plot_data)
        self.plot_button.grid(row=len(parameters) + 1, columnspan=2, pady=10)

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
        self.particle = self.particle_type.get()
        # Load Prameters for the Plot
        if self.particle == "Proton":  
            self.data_path = 'track_fitting/H_in_P10.txt'
        elif self.particle == "Alpha":
            self.data_path = 'track_fitting/He_in_P10.txt'
        self.energies, self.stopping_powers = self.load_energies() #MeV, MeV/mm
        self.load_parameters()
        self.sigma_xy = 2
        self.sigma_z = 2
        self.kernel_size = 31

        # Make distances where stopping_power is to be calculated
        distances = np.linspace(0,100,500)
        times1 = time.time()
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
        timef1 = time.time()
        print("Time for Integrating: ",timef1-times1)
        # Find Points where energy is evaluated
        times2 = time.time()
        points = self.compute_points(distances)
        timef2 = time.time()
        print("Time for Finding Points: ",timef2-times2)
        times3 = time.time()
        # energy_map, self.x, self.y, self.z = create_grid(points, calculated_stopping_powers)
        energy_map = self.create_grid(points, calculated_stopping_powers)
        timef3 = time.time()
        print("Time for Creating Grid: ",timef3-times3)
        times4 = time.time()
        print("energy: ", np.sum(energy_map))
        convolved_energies = self.convolve_with_gaussian(energy_map)
        print("energy 2: ", np.nansum(convolved_energies))
        timef4 = time.time()
        print("Time for Convulation: ",timef4-times4)
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
        grid_resolution (in mm): Resolution of the grid.
        """
        padding = 15

        # Find min and max for each dimension
        min_x, min_y, min_z = np.min(coords, axis=0) - padding
        max_x, max_y, max_z = np.max(coords, axis=0) + padding

        # Create ranges for each dimension with padding
        self.x = np.arange(min_x, max_x + grid_resolution, grid_resolution)
        self.y = np.arange(min_y, max_y + grid_resolution, grid_resolution)
        self.z = np.arange(min_z, max_z + grid_resolution, grid_resolution)
        print(self.x.shape, self.y.shape, self.z.shape)
        # Create meshgrid
        X, Y, Z = np.meshgrid(self.x, self.y, self.z, indexing='ij')
        grid_shape = X.shape
        # Calculate indices
        indices = ((coords - [min_x, min_y, min_z]) / grid_resolution).round().astype(int)

        # Initialize energy grid
        energy_grid = np.zeros(grid_shape)

        # Accumulate energy values using advanced indexing
        np.add.at(energy_grid, (indices[:, 0], indices[:, 1], indices[:, 2]), stopping_powers)

        return energy_grid
    
    def convolve_with_gaussian(self, energy_grid):
        """
        Convolve the energy grid with a 3D Gaussian kernel with varying sigma based on z position.
        """
        def gaussian_kernel_2d(kernel_size, sigma_xy):
            """Create a 2D Gaussian kernel."""
            ax = np.linspace(-(kernel_size - 1) / 2., (kernel_size - 1) / 2., kernel_size)
            xx, yy = np.meshgrid(ax, ax)
            kernel = np.exp(-(xx**2 + yy**2) / (2 * sigma_xy**2))
            return kernel / np.sum(kernel)
        
        def gaussian_kernel_1d(kernel_size, sigma_z):
            """Create a 1D Gaussian kernel."""
            ax = np.linspace(-(kernel_size - 1) / 2., (kernel_size - 1) / 2., kernel_size)
            kernel = np.exp(-ax**2 / (2 * sigma_z**2))
            return kernel / np.sum(kernel)
        
        def gaussian_kernel_3d(kernel_size, sigma_xy, sigma_z):
            """Create a 3D Gaussian kernel."""
            ax = np.linspace(-(kernel_size - 1) / 2., (kernel_size - 1) / 2., kernel_size)
            xx, yy, zz = np.meshgrid(ax, ax, ax)
            kernel = np.exp(-((xx**2 + yy**2) / (2 * sigma_xy**2) + zz**2 / (2 * sigma_z**2)))
            return kernel / np.sum(kernel)
        
        # Determine the sigma values based on z position
        k_xy = 0.0554 
        k_z = 0.0338
        z_positions = np.arange(energy_grid.shape[2])
        sigma_xy_array = np.sqrt(10) * k_xy * np.sqrt(z_positions)
        sigma_z_array = np.sqrt(10) * k_z * np.sqrt(z_positions)
        
        # Ensure sigma values are positive
        sigma_xy_array = np.clip(sigma_xy_array, 1e-5, None)
        sigma_z_array = np.clip(sigma_z_array, 1e-5, None)

        # Initialize the convolved grid
        convolved_grid = np.zeros_like(energy_grid)

        # Apply the convolution for each z slice
        half_kernel_size = self.kernel_size // 2

        # Convolve along the XY plane for each Z slice
        for z in range(energy_grid.shape[2]):
            # Create 2D Gaussian kernel for xy-plane
            kernel_2d_xy = gaussian_kernel_2d(self.kernel_size, sigma_xy_array[z])
            
            # Convolve each slice along the x and y axes
            convolved_grid[:, :, z] = convolve(energy_grid[:, :, z], kernel_2d_xy, mode='same')

        # for z in range(energy_grid.shape[2]):
        #     kernel_1d_z = gaussian_kernel_1d(self.kernel_size, sigma_z_array[z])
        #     for i in np.arange(energy_grid.shape[0]):
        #         for j in np.arange(energy_grid.shape[1]):
        #             onepoint = energy_grid[i,j,z]
        #             if (z-half_kernel_size >= 0) and (z+half_kernel_size < energy_grid.shape[2]):
        #                 convolved_grid [i,j,z-half_kernel_size:z+half_kernel_size] = np.convolve(onepoint, kernel_1d_z, mode='same')

        # Convolve along the z-axis
        temp_grid = np.zeros_like(convolved_grid)
        for z in range(energy_grid.shape[2]):
            kernel_1d_z = gaussian_kernel_1d(self.kernel_size, sigma_z_array[z])
            for dz in range(-half_kernel_size, half_kernel_size + 1):
                kernel_index = dz + half_kernel_size
                
                if 0 <= z + dz < energy_grid.shape[2]:
                    temp_grid[:, :, z] += convolved_grid[:, :, z + dz] * kernel_1d_z[kernel_index]

        return temp_grid
    
    def display_image(self, energy_grid):
        """
        Map to the Padplanes and Plot 
        Plot the 3D energy distribution.
        """
        # Creating PadPlane positions
        pad_x = np.arange(-38.5,38.5+2.2,2.2)
        pad_y = np.arange(-38.5,38.5+2.2,2.2)
        pad_z = np.arange(0,512*self.z_scale,self.z_scale)
        times5 = time.time()
        # Get the indices where energy is not zero for more efficient plotting
        # non_zero_indices = np.nonzero(energy_grid)
        max_value = np.max(energy_grid)
        significant_mask = energy_grid > (0.001 * max_value)
        non_zero_indices = np.nonzero(significant_mask)
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

        timef5 = time.time()
        print("Time for Final Arrays: ",timef5-times5)
        times6 = time.time()
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Plot the 3D scatter plot with energy values as color
        sc = ax.scatter(pad_x_coords, pad_y_coords, pad_z_coords, c=pad_energies, cmap='inferno', marker='o', alpha=0.5)
        
        # Add colorbar
        cbar = plt.colorbar(sc, ax=ax)
        cbar.set_label('Energy (a.u.)')
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        ax.axes.set_xlim3d(left=-40, right=40) 
        ax.axes.set_ylim3d(bottom=-40, top=40) 
        ax.axes.set_zlim3d(bottom=0, top=200)
        
        plt.title(f'3D Energy Distribution of {self.initial_energy} MeV proton, θ = {self.theta} rad, φ = {self.phi} rad')
        timef6 = time.time()
        print("Time for 3d Plot: ",timef6-times6)
        plt.show()
        

    def load_energies(self):
        # Initialize lists to store the data
        energy_MeV = []
        electronic_stopping_MeV_um = []  # Stopping power in MeV/(mg/cm^2)
        nuclear_stopping_MeV_um = []
        path_length_mm = []

        # Read the file
        with open(self.data_path, 'r') as file:
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
    


# def convolve_with_gaussian(self, energy_grid):
    #     """
    #     Convolve the energy grid with a 3D Gaussian kernel.
    #     """

    #     def gaussian_kernel_3d(kernel_size, sigma, sigma_z):
    #         """Create a 3D Gaussian kernel."""
    #         ax = np.linspace(-(kernel_size - 1) / 2., (kernel_size - 1) / 2., kernel_size)
    #         xx, yy, zz = np.meshgrid(ax, ax, ax)
    #         kernel = np.exp(-((xx**2 + yy**2) / (2 * sigma**2) + zz**2 / (2 * sigma_z**2)))
    #         return kernel / np.sum(kernel)
        
    #     kernel = gaussian_kernel_3d(self.kernel_size, self.sigma_xy, self.sigma_z)
    #     convolved_grid = convolve(energy_grid, kernel, mode='constant', cval=0)
    #     return convolved_grid

    

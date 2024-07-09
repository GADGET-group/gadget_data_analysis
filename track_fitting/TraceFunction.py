import numpy as np
from scipy import integrate
from scipy.signal import convolve
import time
from scipy.spatial import KDTree


class TraceFunction:
    def __init__(self, pressure):
        # Load Constants
        self.gas_density = pressure / 760 * 1.56  # mg/cm^3, assuming standard pressure
        self.z_scale = 400. / 512  # Scale factor for z
        self.kernel_size = 31
        self.grid_resolution = 0.5  # mm
        self.k_xy = 0.0554 
        self.k_z = 0.0338
        self.charge_spreading = 0
        self.shaping_time = 2.5 * self.z_scale
        self.p_energies, self.p_stopping_powers, self.p_path_length = self.load_energies('Proton')
        self.a_energies, self.a_stopping_powers, self.a_path_length = self.load_energies('Alpha')
        self.padxy = np.loadtxt('raw_viewer/padxy.txt', delimiter=',')
        self.xy_to_pad = {tuple(np.round(self.padxy[pad], 1)):pad for pad in range(len(self.padxy))}
        self.pad_to_xy = {a: b for b, a in self.xy_to_pad.items()}

    def calculate_xyze(self, initial_energy, initial_point, theta, phi, particle):
        # Load Function Parameters
        self.initial_energy = initial_energy
        self.initial_point = initial_point
        self.theta = theta
        self.phi = phi
        if particle == 'Proton':
            self.energies, self.stopping_powers, self.path_length = self.p_energies, self.p_stopping_powers, self.p_path_length
        elif particle == 'Alpha':
            self.energies, self.stopping_powers, self.path_length = self.a_energies, self.a_stopping_powers, self.a_path_length

        # Integrate stopping powers
        integrated_stopping_powers = self.integrate_stopping_powers()

        # Compute the points where energy is evaluated
        times2 = time.time()
        points = self.compute_points()
        timef2 = time.time()
        print("Time for Finding Points: ", timef2 - times2)

        # Create the grid to convolve
        times3 = time.time()
        energy_map = self.create_grid(points, integrated_stopping_powers)
        timef3 = time.time()
        print("Time for Creating Grid: ", timef3 - times3)

        # Convolution
        times4 = time.time()
        print("Energy before convolution: ", np.sum(energy_map))
        convolved_energies = self.convolve_with_gaussian(energy_map)
        print("Energy after convolution: ", np.sum(convolved_energies))
        timef4 = time.time()
        print("Time for Convolution: ", timef4 - times4)

        # Map to pads and return the results
        return self.map_to_pads_extended(convolved_energies)

    def integrate_stopping_powers(self):
        max_path_length = self.path_length[np.searchsorted(self.energies, self.initial_energy, side='right')]
        ds = 500  # Number of steps for integration
        self.distances = np.linspace(0, max_path_length, ds)  # Distances where stopping power is calculated
        
        def dEdx(E):
            """Differential function to calculate stopping power based on energy."""
            to_return = -np.interp(E, self.energies, self.stopping_powers)
            to_return[E < 0] = 0  # Set stopping power to 0 for negative energies
            return to_return

        # Integrate to find energy at each distance
        Es = np.squeeze(integrate.odeint(lambda E, x: dEdx(E), self.initial_energy, self.distances))

        # Calculate stopping powers
        calculated_stopping_powers = -dEdx(Es) * max_path_length / ds
        return calculated_stopping_powers
    
    def create_grid(self, coords, stopping_powers):
        """
        Create a fine grid for the given coordinate range and resolution.
        """
        padding = 15  # Padding around the min and max coordinates

        # Find min and max for each dimension
        min_x, min_y, min_z = np.min(coords, axis=0) - padding
        max_x, max_y, max_z = np.max(coords, axis=0) + padding

        # Create ranges for each dimension with padding
        self.x = np.arange(min_x, max_x + self.grid_resolution, self.grid_resolution)
        self.y = np.arange(min_y, max_y + self.grid_resolution, self.grid_resolution)
        self.z = np.arange(min_z, max_z + self.grid_resolution, self.grid_resolution)
        grid_shape = (len(self.x), len(self.y), len(self.z))

        # Calculate indices for coordinates in the grid
        indices = ((coords - [min_x, min_y, min_z]) / self.grid_resolution).round().astype(int)

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
        
        # Determine the sigma values based on z position
        z_positions = self.z
        z_positions[z_positions < 0] = 0
        sigma_xy_array = 1 / self.grid_resolution * np.sqrt(10) * self.k_xy * np.sqrt(z_positions) + self.charge_spreading
        sigma_z_array = 1 / self.grid_resolution * np.sqrt(10) * self.k_z * np.sqrt(z_positions) + self.shaping_time
        
        # Ensure sigma values are positive
        sigma_xy_array = np.clip(sigma_xy_array, 1e-5, None)
        sigma_z_array = np.clip(sigma_z_array, 1e-5, None)
        
        # Initialize the convolved grid
        convolved_grid = np.zeros_like(energy_grid)

        # Convolve along the XY plane for each Z slice
        half_kernel_size = self.kernel_size // 2
        for z in range(energy_grid.shape[2]):
            # Create 2D Gaussian kernel for xy-plane
            kernel_2d_xy = gaussian_kernel_2d(self.kernel_size, sigma_xy_array[z])
            # Convolve each slice along the x and y axes
            convolved_grid[:, :, z] = convolve(energy_grid[:, :, z], kernel_2d_xy, mode='same')

        # Convolve along the z-axis
        temp_grid = np.zeros_like(convolved_grid)
        for z in range(energy_grid.shape[2]):
            kernel_1d_z = gaussian_kernel_1d(self.kernel_size, sigma_z_array[z])
            for dz in range(-half_kernel_size, half_kernel_size + 1):
                kernel_index = dz + half_kernel_size
                if 0 <= z + dz < energy_grid.shape[2]:
                    temp_grid[:, :, z] += convolved_grid[:, :, z + dz] * kernel_1d_z[kernel_index]

        return temp_grid
    
    def compute_points(self):
        """
        Compute 3D points along a given direction from an initial point based on distances.
        """
        # Direction vector based on theta and phi angles
        direction_vector = np.round(np.array((np.sin(self.theta) * np.cos(self.phi), 
                                              np.sin(self.theta) * np.sin(self.phi), 
                                              np.cos(self.theta))), 6)
        # Compute points along the direction vector
        points = np.array([np.array(self.initial_point) + distance * direction_vector for distance in self.distances])
        return points
    
    def map_to_pads(self, energy_grid):
        """
        Map the energy grid to the Padplanes and Plot 
        """
        # Creating PadPlane positions
        pad_x = np.arange(-38.5, 38.5 + 2.2, 2.2)
        pad_y = np.arange(-38.5, 38.5 + 2.2, 2.2)

        # Get the indices where energy is significant (greater than 0.1% of max value)
        max_value = np.max(energy_grid)
        significant_mask = energy_grid > (0.001 * max_value)
        non_zero_indices = np.nonzero(significant_mask)
        x_non_zero = self.x[non_zero_indices[0]]
        y_non_zero = self.y[non_zero_indices[1]]
        z_non_zero = self.z[non_zero_indices[2]]
        energy_non_zero = energy_grid[non_zero_indices]
        
        # Map energies to the nearest padplane positions
        pad_energy_dict = {}
        for i, j, k, e in zip(x_non_zero, y_non_zero, z_non_zero, energy_non_zero):
            # Find nearest padplane positions
            pad_x_nearest = pad_x[np.abs(pad_x - i).argmin()]
            pad_y_nearest = pad_y[np.abs(pad_y - j).argmin()]
            if (pad_x_nearest, pad_y_nearest, k) not in pad_energy_dict:
                pad_energy_dict[(pad_x_nearest, pad_y_nearest, k)] = e
            else:
                pad_energy_dict[(pad_x_nearest, pad_y_nearest, k)] += e

        print('Pad energy dict length: ', len(pad_energy_dict.keys()))
        
        # Extract coordinates and energies from the dictionary
        pad_coords = np.array(list(pad_energy_dict.keys()))
        pad_energies = np.array(list(pad_energy_dict.values()))
        pad_x_coords = pad_coords[:, 0]
        pad_y_coords = pad_coords[:, 1]
        pad_z_coords = pad_coords[:, 2]

        return pad_x_coords, pad_y_coords, pad_z_coords, pad_energies
    
    def map_to_pads_extended(self, energy_grid):
        time1 = time.time()
        pad_plane_positions = np.array(list(self.xy_to_pad.keys()))

        # Get the indices where energy is significant (greater than 0.1% of max value)
        max_value = np.max(energy_grid)
        significant_mask = energy_grid > (0.001 * max_value)
        non_zero_indices = np.nonzero(significant_mask)
        x_non_zero = self.x[non_zero_indices[0]]
        y_non_zero = self.y[non_zero_indices[1]]
        z_non_zero = self.z[non_zero_indices[2]]
        energy_non_zero = energy_grid[non_zero_indices]

        def find_nearest_pad_positions(points, pad_plane_positions):
            tree = KDTree(pad_plane_positions)
            distances, indices = tree.query(points)
            nearest_pad_positions = pad_plane_positions[indices]
            return nearest_pad_positions
        
        def get_pad_from_xy(xy):
            '''
            xy: tuple of (x,y) to lookup pad number for
            '''
            xy = tuple(np.round(xy, 1))
            return self.xy_to_pad[xy]

        pad_to_energies = {value: np.zeros_like(self.z) for value in self.xy_to_pad.values()}

        # Map energies to the nearest padplane positions
        # Points (i, j)
        points = np.column_stack((x_non_zero, y_non_zero))

        # Find nearest pad positions for all points
        nearest_pad_positions = find_nearest_pad_positions(points, pad_plane_positions)

        for (i, j, k, e), (pad_x_nearest, pad_y_nearest) in zip(zip(x_non_zero, y_non_zero, z_non_zero, energy_non_zero), nearest_pad_positions):
            pad_no = get_pad_from_xy((pad_x_nearest, pad_y_nearest))
            pad_to_energies[pad_no][np.where(self.z == k)[0]] += e
        time2 = time.time()
        print("Time for mapping and data foermatting: ", time2 - time1)
        print(self.z.shape, len(pad_to_energies.values()))
        return pad_to_energies, self.z
    
    def convert_pad_to_coords(self, pad_to_energies, z_values):
        
        # Initialize the dictionary for the pad coordinates and energies
        pad_energy_dict = {}

        for pad_no, energies in pad_to_energies.items():
            pad_x, pad_y = self.pad_to_xy[pad_no] 
            for z_index, energy in enumerate(energies):
                if energy > 0:  # Only consider significant energies
                    pad_z = z_values[z_index]
                    if (pad_x, pad_y, pad_z) not in pad_energy_dict:
                        pad_energy_dict[(pad_x, pad_y, pad_z)] = energy
                    else:
                        pad_energy_dict[(pad_x, pad_y, pad_z)] += energy

        print('Pad energy dict length: ', len(pad_energy_dict.keys()))

        # Extract coordinates and energies from the dictionary
        pad_coords = np.array(list(pad_energy_dict.keys()))
        pad_energies = np.array(list(pad_energy_dict.values()))
        pad_x_coords = pad_coords[:, 0]
        pad_y_coords = pad_coords[:, 1]
        pad_z_coords = pad_coords[:, 2]

        return pad_x_coords, pad_y_coords, pad_z_coords, pad_energies

    def load_energies(self, particle='Proton'):
        # Select data file based on particle type
        if particle == 'Proton':
            data_path = 'track_fitting/H_in_P10.txt'
        elif particle == 'Alpha':
            data_path = 'track_fitting/He_in_P10.txt'

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
                        path_unit = parts[5]  # Get the unit part
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
        
        # Convert stopping powers using gas density
        stopping_power_converted = self.gas_density * (electronic_stopping_MeV_um + nuclear_stopping_MeV_um) / 10
        return energy_MeV, stopping_power_converted, path_length_mm

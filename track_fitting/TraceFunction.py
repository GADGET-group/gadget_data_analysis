'''
@author bjain
'''

import numpy as np
from scipy.signal import convolve
import time
from scipy.spatial import KDTree
import srim_interface


class TraceFunction:
    '''
    Class for simulating detector response to a single charged particle.
    Usage:
        1. construct for a given gas density and particle (currently proton and alpha are supported)
        2. Configure parameters by editing member variables. Gas density can be changed by calling "load_srim_table".
        3. Call "simulate_event". This will simulate a particle stopping in the detector, diffusion of electrons, and charge spreading.
           Results of this process are stored in member variables which are most conveniently accessed using on of the following functions:
            a.
            b.
    '''

    def __init__(self, gas_density, particle):

        #physical constants relating to the detector
        self.grid_padding = 15  #mm; grid created will extend this far beyond particle track w/o diffusion
        self.k_xy = 0.0554 #transverse diffusion, sqrt cm?
        self.k_z = 0.0338 #longitudinal diffusion
        self.charge_spreading_sigma = 0 #additional width from charge spreading in mm, sigma=charge_spreading + k_xy*sqrt(z)
        self.shaping_time_sigma = 0 #stddev of gaussian induced by amplifier shaping time, in mm, not time bins

        #load SRIM table for particle. These need to be reloaded if gas desnity is changed.
        self.load_srim_table(particle, gas_density)
        
        #parameters for grid size and other numerics
        self.num_stopping_power_points = 500 #number of points at which to compute 1D energy deposition
        self.kernel_size = 31 #size of gaussian kernels. MUST BE ODD!
        self.grid_resolution = 0.5  #spacing between grid lines mm

        self.padxy = np.loadtxt('raw_viewer/padxy.txt', delimiter=',')
        self.xy_to_pad = {tuple(np.round(self.padxy[pad], 1)):pad for pad in range(len(self.padxy))}
        self.pad_to_xy = {a: b for b, a in self.xy_to_pad.items()}

        self.enable_print_statements = True

        self.initial_energy = 6 #MeV
        self.initial_point = (0,0,0) #(x,y,z) mm
        self.theta, self.phi = 0,0 #angles describing direction in which emmitted particle travels, in radians

    def load_srim_table(self, particle:str, gas_density:float)
        '''
        Reload SRIM table
        particle: proton or alpha
        gas density: mg/cm^3
        '''
        if particle.lower() == 'proton':
            self.srim_table = srim_interface.SRIM_Table('H_in_P10.txt', gas_density)
        elif particle.lower() == 'alpha'
            self.srim_table = srim_interface.SRIM_Table('He_inP10.txt', gas_density)

    def simulate_event(self):
        '''
        
        '''
        # Integrate stopping powers
        stopping_distance = self.srim_table.get_stopping_distance(self.initial_energy)
        self.distances = np.linspace(0, stopping_distance, self.num_stopping_power_points)
        dx = self.distances[1] - self.distances[0]
        stopping_powers = self.srim_table.get_stopping_power_after_distances(self.distances)*dx

        # Compute the points where energy is evaluated
        times2 = time.time()
        direction_vector = np.array((np.sin(self.theta) * np.cos(self.phi), 
                                              np.sin(self.theta) * np.sin(self.phi), 
                                              np.cos(self.theta)))
        #get positions at which energy should be deposited in 3d
        points = np.zeros(self.num_stopping_power_points,3)
        for i in range(3):
            points[:,i] = self.initial_point[i] + direction_vector[i]*self.distances
        #remove any point with negative z (eg on wrong side of micro-megas)
        points = points[points[:,2]>0]
        timef2 = time.time()

        # create a 3d grid, surounding the event, on which all future operations will be performed
        times3 = time.time()
        min_x, min_y, min_z = np.min(points, axis=0) - self.grid_padding
        max_x, max_y, max_z = np.max(points, axis=0) + self.grid_padding
        self.x = np.arange(min_x, max_x + self.grid_resolution, self.grid_resolution)
        self.y = np.arange(min_y, max_y + self.grid_resolution, self.grid_resolution)
        self.z = np.arange(min_z, max_z + self.grid_resolution, self.grid_resolution)
        grid_shape = (len(self.x), len(self.y), len(self.z))
        energy_grid = np.zeros(grid_shape)
        #map stopping power points in 3-space to the nearest grid points
        indices = ((points - [min_x, min_y, min_z]) / self.grid_resolution).round().astype(int)
        np.add.at(energy_grid, (indices[:, 0], indices[:, 1], indices[:, 2]), stopping_powers)
        timef3 = time.time()

        # Convolution
        times4 = time.time()#TODO: left off here
        #calculate sigma for gaussian kernels at each z-slice of the grid
        #the sqrt(10) is to convert a diffusion coeficient with units of sqrt(cm) to mm
        sigma_xy_array = 1 / self.grid_resolution * np.sqrt(10) * self.k_xy * np.sqrt(self.z) + self.charge_spreading_sigma
        sigma_z_array = 1 / self.grid_resolution * np.sqrt(10) * self.k_z * np.sqrt(self.z) + self.shaping_time_sigma
        #do xy convolution
        kernel_end = (self.kernel_size-1)/2*self.grid_resolution
        kernel_axis = np.arange(-kernel_end, kernel_end, self.kernel_size)
        kernel_xx, kernel_yy = np.meshgrid(kernel_axis, kernel_axis)
        kernel_r2 = kernel_xx**2 + kernel_yy**2
        for z_index in range(len(self.z)):
            kernel = np.exp(-kernel_r2/2/sigma_xy_array[z_index]**2)
            kernel /= np.sum(kernel)
            energy_grid[:,:,z_index] = convolve(energy_grid[:,:,z_index])
        #spread charge in z-direction
        convolved_energies = np.zeros_like(energy_grid)
        half_kernel_size = self.kernel_size // 2
        for z_index in range(len(self.z)):
            kernel = np.exp(-kernel_axis**2/2/sigma_z_array[z_index]**2)
            for dz in range(-half_kernel_size, half_kernel_size + 1):
                kernel_index = dz + half_kernel_size
                if 0 <= z + dz < energy_grid.shape[2]:
                    energy_grid[:, :, z] += energy_grid[:, :, z + dz] * kernel[kernel_index]

        timef4 = time.time()
        
        if self.enable_print_statements:
            print("Time for Finding Points: ", timef2 - times2)
            print("Time for Creating Grid: ", timef3 - times3)
            print("Energy before convolution: ", np.sum(energy_grid)
            print("Energy after convolution: ", np.sum(convolved_energies))
            print("Time for Convolution: ", timef4 - times4)

        # Map to pads and return the results
        self.convolved_energies = convolved_energies
        #return self.map_to_pads_extended(convolved_energies)
    
 
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
        '''
        
        '''
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

    

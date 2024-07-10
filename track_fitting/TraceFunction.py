'''
@author bjain
'''

import numpy as np
import scipy.signal
import time
from scipy.spatial import KDTree
from track_fitting import srim_interface


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
        self.particle = particle #this variable should only be changed using the load_srim_table function
    
        self.enable_print_statements = True

        #physical constants relating to the detector
        self.grid_padding = 15  #mm; grid created will extend this far beyond particle track w/o diffusion
        self.k_xy = 0.0554 #transverse diffusion, sqrt cm?
        self.k_z = 0.0338 #longitudinal diffusion
        self.charge_spreading_sigma = 0 #additional width from charge spreading in mm, sigma=charge_spreading + k_xy*sqrt(z)
        self.shaping_time_sigma = 0 #stddev of gaussian induced by amplifier shaping time, in mm, not time bins
        self.threshold = 0.001 #bins with less than this fraction of max bin value will not be mapped to pad plane

        #load SRIM table for particle. These need to be reloaded if gas desnity is changed.
        self.load_srim_table(particle, gas_density)
        
        #parameters for grid size and other numerics
        self.num_stopping_power_points = 500 #number of points at which to compute 1D energy deposition
        self.kernel_size = 31 #size of gaussian kernels. MUST BE ODD!
        self.grid_resolution = 0.5  #spacing between grid lines mm

        self.padxy = np.loadtxt('raw_viewer/padxy.txt', delimiter=',')
        self.xy_to_pad = {tuple(np.round(self.padxy[pad], 1)):pad for pad in range(len(self.padxy))}
        self.pad_to_xy = {a: b for b, a in self.xy_to_pad.items()}

        #event parameters
        self.initial_energy = 6 #MeV
        self.initial_point = (0,0,0) #(x,y,z) mm
        self.theta, self.phi = 0,0 #angles describing direction in which emmitted particle travels, in radians

        #initial_charge_distribution holds the charge deposited in the gas prior to diffusion.
        #charge_distribution holds the charge distribution as observed by the detector. It 
        #is a 3d array. grid_xs,ys,zs hold the x,y, and z coordinates in mm relative to the center of the micromegas
        #of each point in the charge_distribution array. These will be populated when "simulate_event" is called.
        self.initial_charge_distribution = np.zeros((0,0,0))
        self.observed_charge_distribution = np.zeros((0,0,0))
        self.grid_xs, self.grid_ys, self.grid_zs = np.zeros(0), np.zeros(0), np.zeros(0)
        #variables which are equivalent to the get_xyze retrun value of the raw_h5_file class
        #x & y will be mapped to the nearest pad, but z will have the same spacing as the grid used for computation
        #populated by map_to_pads
        self.xs, self.ys, self.zs, self.es = np.zeros(0), np.zeros(0), np.zeros(0), np.zeros(0)

    def load_srim_table(self, particle:str, gas_density:float):
        '''
        Reload SRIM table
        particle: proton or alpha
        gas density: mg/cm^3
        '''
        if particle.lower() == 'proton':
            self.srim_table = srim_interface.SRIM_Table('track_fitting/H_in_P10.txt', gas_density)
        elif particle.lower() == 'alpha':
            self.srim_table = srim_interface.SRIM_Table('track_fitting/He_in_P10.txt', gas_density)

    def simulate_event(self, map_to_pads=True):
        '''
        
        '''
        # Integrate stopping powers
        stopping_distance = self.srim_table.get_stopping_distance(self.initial_energy)
        self.distances = np.linspace(0, stopping_distance, self.num_stopping_power_points)
        dx = self.distances[1] - self.distances[0]
        stopping_powers = self.srim_table.get_stopping_power_after_distances(self.initial_energy, self.distances)*dx

        # Compute the points where energy is evaluated
        time2 = time.time()
        direction_vector = np.array((np.sin(self.theta) * np.cos(self.phi), 
                                              np.sin(self.theta) * np.sin(self.phi), 
                                              np.cos(self.theta)))
        #get positions at which energy should be deposited in 3d
        points = np.zeros((self.num_stopping_power_points,3))
        for i in range(3):
            points[:,i] = self.initial_point[i] + direction_vector[i]*self.distances
        #points = np.array([np.array(self.initial_point) + distance * direction_vector for distance in self.distances])
        #remove any point with negative z (eg on wrong side of micro-megas)
        points = points[points[:,2]>0]
        timef2 = time.time()

        # create a 3d grid, surounding the event, on which all future operations will be performed
        time3 = time.time()
        min_x, min_y, min_z = np.min(points, axis=0) - self.grid_padding
        max_x, max_y, max_z = np.max(points, axis=0) + self.grid_padding
        self.grid_xs = np.arange(min_x, max_x + self.grid_resolution, self.grid_resolution)
        self.grid_ys = np.arange(min_y, max_y + self.grid_resolution, self.grid_resolution)
        self.grid_zs = np.arange(min_z, max_z + self.grid_resolution, self.grid_resolution)
        grid_shape = (len(self.grid_xs), len(self.grid_ys), len(self.grid_zs))
        energy_grid = np.zeros(grid_shape)
        #map stopping power points in 3-space to the nearest grid points
        indices = ((points - [min_x, min_y, min_z]) / self.grid_resolution).round().astype(int)
        np.add.at(energy_grid, (indices[:, 0], indices[:, 1], indices[:, 2]), stopping_powers)
        self.initial_charge_distribution = np.copy(energy_grid)
        #np.add.at(energy_grid, np.transpose(indices), stopping_powers)
        timef3 = time.time()

        # Convolution
        #calculate sigma for gaussian kernels at each z-slice of the grid
        #the sqrt(10) is to convert a diffusion coeficient with units of sqrt(cm) to mm
        sigma_xy_array = 1 / self.grid_resolution * np.sqrt(10) * self.k_xy * np.sqrt(self.grid_zs) + self.charge_spreading_sigma
        sigma_z_array = 1 / self.grid_resolution * np.sqrt(10) * self.k_z * np.sqrt(self.grid_zs) + self.shaping_time_sigma
        #do xy convolution
        kernel_end = (self.kernel_size-1)/2*self.grid_resolution
        kernel_axis = np.linspace(-kernel_end, kernel_end, self.kernel_size)
        kernel_xx, kernel_yy = np.meshgrid(kernel_axis, kernel_axis)
        kernel_r2 = kernel_xx**2 + kernel_yy**2
        for z_index in range(len(self.grid_zs)): #todo: consider breaking this apart into an x convolution and y direction convolution
            kernel = np.exp(-kernel_r2/2/sigma_xy_array[z_index]**2)
            kernel /= np.sum(kernel)
            energy_grid[:,:,z_index] = scipy.signal.convolve(energy_grid[:,:,z_index], kernel, mode='same')
        time4 = time.time()
        #spread charge in z-direction
        charge_distribution = np.zeros_like(energy_grid)
        half_kernel_size = self.kernel_size // 2
        for z_index in range(len(self.grid_zs)):
            kernel = np.exp(-kernel_axis**2/2/sigma_z_array[z_index]**2)
            kernel /= np.sum(kernel)
            for dz in range(-half_kernel_size, half_kernel_size + 1): #TODO: can we do this with np.add.at?
                kernel_index = dz + half_kernel_size
                if 0 <= z_index + dz < energy_grid.shape[2]:
                    charge_distribution[:, :, z_index] += energy_grid[:, :, z_index + dz] * kernel[kernel_index]

        time5 = time.time()
        
        if self.enable_print_statements:
            print("Time for Finding Points: ", timef2 - time2)
            print("Time for Creating Grid: ", timef3 - time3)
            print("Energy before convolution: ", np.sum(energy_grid))
            print("Energy after convolution: ", np.sum(charge_distribution))
            print("Time for xy convolution: ", time4 - time3)
            print("Time for z charge spreading: ", time5 - time4)

        # Map to pads and return the results
        self.observed_charge_distribution = charge_distribution
        #return self.map_to_pads_extended(convolved_energies)
        if map_to_pads:
            self.map_to_pads()
    
 
    def map_to_pads(self):
        """
        Map the energy grid to the pad plane
        """
        #TODO: this function is very inefficient, change to doing mapping on a per-column basis
        start_time = time.time()
        # Creating PadPlane positions
        pad_x = np.arange(-38.5, 38.5 + 2.2, 2.2)
        pad_y = np.arange(-38.5, 38.5 + 2.2, 2.2)

        #get indices above threshold
        max_value = np.max(self.observed_charge_distribution)
        above_threshold_mask = self.observed_charge_distribution > (self.threshold * max_value)
        above_theshold_indices = np.nonzero(above_threshold_mask)
        x_above_theshold = self.grid_xs[above_theshold_indices[0]]
        y_above_theshold = self.grid_ys[above_theshold_indices[1]]
        z_above_theshold = self.grid_zs[above_theshold_indices[2]]
        energy_above_theshold = self.observed_charge_distribution[above_theshold_indices]
        
        # Map energies to the nearest padplane positions
        pad_energy_dict = {}
        for x, y, z, e in zip(x_above_theshold, y_above_theshold, z_above_theshold, energy_above_theshold):
            # Find nearest padplane positions
            pad_x_nearest = pad_x[np.abs(pad_x - x).argmin()]
            pad_y_nearest = pad_y[np.abs(pad_y - y).argmin()]
            if (pad_x_nearest, pad_y_nearest, z) not in pad_energy_dict:
                pad_energy_dict[(pad_x_nearest, pad_y_nearest, z)] = e
            else:
                pad_energy_dict[(pad_x_nearest, pad_y_nearest, z)] += e
        
        # Extract coordinates and energies from the dictionary
        pad_coords = np.array(list(pad_energy_dict.keys()))
        self.es = np.array(list(pad_energy_dict.values()))
        self.xs = pad_coords[:, 0]
        self.ys = pad_coords[:, 1]
        self.zs = pad_coords[:, 2]

        end_time = time.time()
        if self.enable_print_statements:
            print('Pad energy dict length: ', len(pad_energy_dict.keys()))
            print('pad map time: ', end_time - start_time)

    
    def map_to_pads_extended(self, energy_grid):
        '''
        
        '''
        time1 = time.time()
        pad_plane_positions = np.array(list(self.xy_to_pad.keys()))

        # Get the indices where energy is significant (greater than 0.1% of max value)
        max_value = np.max(energy_grid)
        significant_mask = energy_grid > (0.001 * max_value)
        non_zero_indices = np.nonzero(significant_mask)
        x_non_zero = self.grid_xs[non_zero_indices[0]]
        y_non_zero = self.grid_ys[non_zero_indices[1]]
        z_non_zero = self.grid_zs[non_zero_indices[2]]
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

        pad_to_energies = {value: np.zeros_like(self.grid_zs) for value in self.xy_to_pad.values()}

        # Map energies to the nearest padplane positions
        # Points (i, j)
        points = np.column_stack((x_non_zero, y_non_zero))

        # Find nearest pad positions for all points
        nearest_pad_positions = find_nearest_pad_positions(points, pad_plane_positions)

        for (i, j, k, e), (pad_x_nearest, pad_y_nearest) in zip(zip(x_non_zero, y_non_zero, z_non_zero, energy_non_zero), nearest_pad_positions):
            pad_no = get_pad_from_xy((pad_x_nearest, pad_y_nearest))
            pad_to_energies[pad_no][np.where(self.grid_zs == k)[0]] += e
        time2 = time.time()
        print("Time for mapping and data foermatting: ", time2 - time1)
        print(self.grid_zs.shape, len(pad_to_energies.values()))
        return pad_to_energies, self.grid_zs
    
    def get_xyze_before_pad_map(self):
        
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

    

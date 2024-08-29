'''
@author bjain and aadams
'''

import numpy as np
import scipy.signal
import time
from scipy.spatial import KDTree
from track_fitting import srim_interface
import matplotlib.pylab as plt


class SingleParticleEvent:
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
        '''
        gas_density: density in mg/cm^3
        '''
        self.particle = particle #this variable should only be changed using the load_srim_table function
    
        self.enable_print_statements = False

        #physical constants relating to the detector
        self.grid_padding = 15  #mm; grid created will extend this far beyond particle track w/o diffusion
        self.k_xy = 0.0554 #transverse diffusion, in sqrt cm
        self.k_z = 0.0338 #longitudinal diffusion, in sqrt cm
        self.charge_spreading_sigma = 0 #additional width from charge spreading in mm, sigma=charge_spreading + k_xy*sqrt(z)
        self.shaping_width = 7 #FWHM of the shaping amplifier in time bins
        self.zscale = 1.45 #mm/time bin
        self.counts_per_MeV = 1
        
        self.pad_gain_match_uncertainty = 0 #unitless
        self.other_systematics = 0 #adc counts

        #load SRIM table for particle. These need to be reloaded if gas desnity is changed.
        self.load_srim_table(particle, gas_density)
        
        #parameters for grid size and other numerics
        self.num_stopping_power_points = 500 #number of points at which to compute 1D energy deposition
        self.kernel_size = 31 #size of gaussian kernels. MUST BE ODD!
        self.grid_resolution = 1.3  #spacing between grid lines mm
        self.shaping_kernel_size = 31 #must be odd

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
        #dictionary containing energy deposition on each pad as a function of z coordinate
        self.sim_pad_traces = {} #charge distribution over pads, binned per self.grid_zs. Charge units are in MeV
        self.traces_to_fit = {} #trace data to try to fit. Populated by calling self.set_real_data
        self.aligned_sim_traces = {} #simulated data prepared for comparison with traces_to_fit by calling align_pad_traces. Charge units are in ADC units
        
        self.peak_bins = {} #dictionairy of peak bin indices, indexed by pad number
        self.peak_vals = {} #dictionary holding the max value of each trace to be fit
        
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

    def get_pad_from_xy(self, xy):
            '''
            xy: tuple of (x,y) to lookup pad number for
            '''
            xy = tuple(np.round(xy, 1))
            if xy in self.xy_to_pad:
                return self.xy_to_pad[xy]
            else:
                return None

    def get_energy_deposition(self):
        '''
        Return energy deposition vs distance.
        returns distances, energy deposition
        '''
        # Integrate stopping powers
        stopping_distance = self.srim_table.get_stopping_distance(self.initial_energy)
        distances = np.linspace(0, stopping_distance, self.num_stopping_power_points)
        dx = distances[1] - distances[0]
        energy_deposition = self.srim_table.get_stopping_power_after_distances(self.initial_energy, distances)*dx
        return distances, energy_deposition

    def simulate_event(self, map_to_pads=True):
        '''
        
        '''
        self.distances, energy_deposition = self.get_energy_deposition()
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
        #points = points[points[:,2]>0]
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
        np.add.at(energy_grid, (indices[:, 0], indices[:, 1], indices[:, 2]), energy_deposition)
        self.initial_charge_distribution = np.copy(energy_grid)
        #np.add.at(energy_grid, np.transpose(indices), stopping_powers)
        timef3 = time.time()

        # Convolution
        #calculate sigma for gaussian kernels at each z-slice of the grid
        #the sqrt(10) is to convert a diffusion coeficient with units of sqrt(cm) to mm
        sigma_xy_array = np.sqrt(10) * self.k_xy * np.sqrt(self.grid_zs) + self.charge_spreading_sigma
        sigma_z_array = np.sqrt(10) * self.k_z * np.sqrt(self.grid_zs)
        #do xy convolution
        kernel_end = (self.kernel_size-1)/2
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
        pad_coords = np.arange(-38.5, 38.5 + 2.2, 2.2)
        pad_boundaries = pad_coords + 1.1 #right/top boundary of pad

        #remap each grid x,y to nearest pad x,y
        grid_xs = self.grid_xs[(self.grid_xs > -38.5-1.1) & (self.grid_xs < 38.5+1.1)]
        pad_xs_indices = np.searchsorted(pad_boundaries, grid_xs)
        pad_xs = pad_coords[pad_xs_indices]

        grid_ys = self.grid_ys[(self.grid_ys > -38.5-1.1) & (self.grid_ys < 38.5+1.1)]
        pad_ys_indices = np.searchsorted(pad_boundaries, grid_ys)
        pad_ys_indices = pad_ys_indices[pad_ys_indices < len(self.grid_ys)]
        pad_ys = pad_coords[pad_ys_indices]

        self.sim_pad_traces = {}

        for padx, grid_sliced_by_x in zip(pad_xs, self.observed_charge_distribution):
            for pady, trace in zip(pad_ys, grid_sliced_by_x):
                pad_num = self.get_pad_from_xy((padx, pady))
                if pad_num == None: #coordinate not in pad map
                    continue
                if pad_num not in self.sim_pad_traces:
                    self.sim_pad_traces[pad_num] = np.zeros(len(self.grid_zs))
                self.sim_pad_traces[pad_num] += trace

        if self.enable_print_statements:
            print('time for pad map:', time.time() - start_time)

    def get_xyze(self, source='pad map', threshold=-np.inf):
        '''
        returns x,y,z,e arrays, similar to the same method in raw_h5_file
        
        source: can be 'energy grid', 'pad map', or 'aligned'
        threshold: only bins with more than this much energy deposition (in MeV) will be returned
        '''
        xs, ys, es = [],[],[]
        if source == 'pad map':
            for pad in self.sim_pad_traces:
                x,y = self.pad_to_xy[pad]
                xs.append(x)
                ys.append(y)
                es.append(self.sim_pad_traces[pad])
            num_z_bins = len(self.grid_zs)
        elif source == 'energy grid':
            for i,x in enumerate(self.grid_xs):
                for j,y in enumerate(self.grid_ys):
                    xs.append(x)
                    ys.append(y)
                    es.append(self.observed_charge_distribution[i,j,:])
                num_z_bins = len(self.grid_zs)
        elif source == 'aligned':
            for pad in self.aligned_sim_traces:
                x,y = self.pad_to_xy[pad]
                xs.append(x)
                ys.append(y)
                es.append(self.aligned_sim_traces[pad])
            num_z_bins = self.num_trimmed_trace_bins
        xs = np.repeat(xs, num_z_bins)
        ys = np.repeat(ys, num_z_bins)
        es = np.array(es).flatten()
        if source == 'energy grid' or source == 'pad map':
            z_axis = self.grid_zs
        elif source == 'aligned':
            z_axis = np.arange(self.num_trimmed_trace_bins)*self.zscale
        zs = np.tile(z_axis, int(len(xs)/len(z_axis)))
        if threshold != -np.inf:
            xs = xs[es>threshold]
            ys = ys[es>threshold]
            zs = zs[es>threshold]
            es = es[es>threshold]
        return xs, ys, zs, es
    
    def set_real_data(self, pads, traces, trim_threshold, trim_pad = 5):
        '''
        Prepares real pad traces for coomparison to simulated data.
        This function does the following:
        1. Stores traces to member variables
        2. Trim traces if possible, keeping the length of all traces the same but the portion of regions where 
           all traces are less than fit threshold.
        3. Find time bin in each trace with peak value, or average of time bins with peak values if there are more than one. 

        pads: list of pads
        traces: list of traces, one for each pad
        trim_threshold: only portions of the traces above this threshold will be used when fitting
        trim_pad: number of elements to leave on each side of the trimmed traces
        '''
        #steps 1 & 2
        self.traces_to_fit = {pad: trace for pad, trace in zip(pads, traces)}
        trim_before = 512 #will be set to the first non-zero time bin in any trace
        trim_after = -1
        #perform thresholding and find indecies for trimming
        for pad in self.traces_to_fit:
            trace = self.traces_to_fit[pad]
            above_threshold_bins = np.nonzero(trace >= trim_threshold)
            first, last = np.min(above_threshold_bins), np.max(above_threshold_bins)
            if first < trim_before:
                trim_before = first
            if last > trim_after:
                trim_after = last
        #trim traces
        trim_start = max(trim_before - trim_pad, 0)
        trim_end = min(trim_after + trim_pad, len(trace))
        for pad in self.traces_to_fit:
            self.traces_to_fit[pad] = self.traces_to_fit[pad][trim_start:trim_end]
        #find peaks
        self.peak_bins = {}
        self.peak_vals = {}
        for pad in self.traces_to_fit:
            trace = self.traces_to_fit[pad]
            max_val = np.max(trace)
            self.peak_bins[pad] = np.average(np.where(trace == max_val))
            self.peak_vals[pad] = max_val
        self.num_trimmed_trace_bins = len(trace)



    def align_pad_traces(self):
        '''
        First, find "t_offset" such that time bins can be computed using t = z/zscale + t_offset
        This is done such that the difference in peak positions is minimized, weighted by peak height
          in the real data by minimizing: sum over pads peak_value*(t_actual peak - t_simulated_peak)^2
          The sum is over all pads that are in both the simulated and real dataset. peak_value is taken from
          the target/real trace.
        
        Then distribute simulated charge into a histogram with the same binning as the real data, distributing charge
        between bins based on the specified shaping time.
        '''
        #determin offset to apply to simulated trace in time bin units
        start_time = time.time()
        sum_pdeltax = 0
        sum_p = 0
        for pad in self.traces_to_fit:
            if pad in self.sim_pad_traces:
                p = self.peak_vals[pad]
                sum_p += p

                simulated_peak_val = np.max(self.sim_pad_traces[pad])
                simulated_peak_bin = np.average(self.grid_zs[np.where(self.sim_pad_traces[pad] == simulated_peak_val)])/self.zscale
                deltax = self.peak_bins[pad] - simulated_peak_bin
                sum_pdeltax += p*deltax
        if sum_p == 0: #no pads in common, rebin without alignment
            t_offset = 0
        else:
            t_offset = sum_pdeltax/sum_p
        #shift charge distribution and put charge in nearest bins
        time_axis = self.grid_zs/self.zscale + t_offset #new charge locations in time bin units
        time_bin_map = np.round(time_axis).astype(int) #where each charge should go
        #don't map bins that would be out of range
        valid_bins_mask = (time_bin_map>0) & (time_bin_map < self.num_trimmed_trace_bins)

        kernel_size = self.shaping_kernel_size
        kernel_ax = np.linspace(-(kernel_size - 1) / 2., (kernel_size - 1) / 2., kernel_size)
        sigma = self.shaping_width/2.3548
        shaping_kernel = np.exp(-kernel_ax**2 / (2 * sigma**2))
        shaping_kernel *= self.counts_per_MeV/np.sum(shaping_kernel) #norm of the kernel will be conversion to counts from MeV
        
        self.aligned_sim_traces = {}
        for pad in self.sim_pad_traces:
            aligned_trace = np.zeros(self.num_trimmed_trace_bins)
            np.add.at(aligned_trace, time_bin_map[valid_bins_mask], self.sim_pad_traces[pad][valid_bins_mask])
            self.aligned_sim_traces[pad] = scipy.signal.convolve(aligned_trace, shaping_kernel, 'same')
        if self.enable_print_statements:
            print('trace alignment took %E s'%(time.time() - start_time))

    def get_residuals(self):
        sim_trace_dict = self.aligned_sim_traces
        real_trace_dict = self.traces_to_fit
        residuals_dict = {}
        for pad in sim_trace_dict:
            if pad not in real_trace_dict:
                residuals_dict[pad] = sim_trace_dict[pad]
            else:
                residuals_dict[pad] = sim_trace_dict[pad] - real_trace_dict[pad]
        for pad in real_trace_dict:
            if pad not in sim_trace_dict:
                residuals_dict[pad] = -real_trace_dict[pad]
        return residuals_dict
    
    def get_residuals_xyze(self):
        residuals_dict = self.get_residuals()
        xs, ys, es = [],[],[]
        for pad in residuals_dict:
            es.append(residuals_dict[pad])
            x,y = self.pad_to_xy[pad]
            xs.append(x)
            ys.append(y)
        num_z_bins = len(es[0])
        xs = np.repeat(xs, num_z_bins)
        ys = np.repeat(ys, num_z_bins)
        es = np.array(es).flatten()
        z_axis = np.arange(num_z_bins)*self.zscale
        zs = np.tile(z_axis, int(len(xs)/len(z_axis)))
        return xs, ys, zs, es

    def log_likelihood(self):
        start_time = time.time()
        to_return = 0
        for pad in  self.pad_to_xy: #iterate over all pads, regardless of if they fired
            if pad in self.aligned_sim_traces: #if the pad trace was simulated
                sigma = self.aligned_sim_traces[pad]*self.pad_gain_match_uncertainty + self.other_systematics
                to_return += np.sum(-np.log(np.sqrt(2*np.pi)*sigma))
                if pad in self.traces_to_fit: #pad fired and was simulated
                    residuals = self.aligned_sim_traces[pad] - self.traces_to_fit[pad]
                    to_return += np.sum(-residuals*residuals/(2*sigma**2))
                else: #pad was simulated firing, but did not
                    sigma = self.other_systematics
                    to_return += -self.num_trimmed_trace_bins*np.log(np.sqrt(2*np.pi)*sigma)
                    to_return += -np.sum(self.aligned_sim_traces[pad]*self.aligned_sim_traces[pad])/2/sigma**2
            else: #pad was not simulated
                if pad in self.traces_to_fit: #pad fired, but was not simulated
                    sigma = self.other_systematics
                    to_return += -self.num_trimmed_trace_bins*np.log(np.sqrt(2*np.pi)*sigma)
                    to_return += -np.sum(self.traces_to_fit[pad]*self.traces_to_fit[pad])/2/sigma**2
                else: #pad did not fire and was not simulated
                    sigma = self.other_systematics
                    to_return += -self.num_trimmed_trace_bins*np.log(np.sqrt(2*np.pi)*sigma)

        if self.enable_print_statements:
            print('likelihood time: %f s'%(time.time() - start_time))
        return to_return
    
    #######################
    # Visualization Tools #
    #######################
    def plot_traces(self, trace_dict, title=''):
        plt.figure()
        for pad in trace_dict:
            r = pad/1024*.8
            g = (pad%512)/512*.8
            b = (pad%256)/256*.8
            plt.plot(trace_dict[pad], label=str(pad), color=(r,g,b))
        plt.legend()
        plt.title(title)

    def plot_residuals(self):
        self.plot_traces(self.get_residuals(), 'residuals')

    def plot_xyze(self, xs, ys, zs, es, title='', energy_threshold=-np.inf):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        # Plot the 3D scatter plot with energy values as color
        xs = xs[es>energy_threshold]
        ys = ys[es>energy_threshold]
        zs = zs[es>energy_threshold]
        es = es[es>energy_threshold]
        sc = ax.scatter(xs, ys,zs, c=es,  alpha=0.5)#, cmap='inferno', marker='o',)# norm=matplotlib.colors.LogNorm())
        # Add colorbar
        cbar = plt.colorbar(sc, ax=ax)
        cbar.set_label('Energy (adc units)')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.axes.set_xlim3d(left=-100, right=100) 
        ax.axes.set_ylim3d(bottom=-100, top=100) 
        ax.axes.set_zlim3d(bottom=0, top=200)
        plt.title(title)

    def plot_simulated_3d_data(self, mode='simulated traces',  title='simulated_data', threshold=-np.inf): #show plots of initial guess
        self.plot_xyze(*self.get_xyze(mode, threshold), title, threshold)
    
    def plot_residuals_3d(self, title='residuals', energy_threshold=0):
        #in this case treshold is applied to absolute value
        xs, ys, zs, es = self.get_residuals_xyze()
        xs = xs[np.abs(es)>energy_threshold]
        ys = ys[np.abs(es)>energy_threshold]
        zs = zs[np.abs(es)>energy_threshold]
        es = es[np.abs(es)>energy_threshold]
        self.plot_xyze(xs, ys, zs, es, title)

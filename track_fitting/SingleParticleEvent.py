'''
@author bjain and aadams
'''
import os
import time

import numpy as np
import matplotlib.pylab as plt
import scipy

from track_fitting import srim_interface

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
        self.gas_density = gas_density  #this variable should only be changed using the load_srim_table function
    
        self.enable_print_statements = False

        #physical constants relating to the detector
        self.sigma_xy = 1. #mm
        self.sigma_z = 1. #mm
        self.zscale = 1.45 #mm/time bin
        self.counts_per_MeV = 1.
        #minimum number of counts above background required for the pad to fire
        #only used when calculating log liklihood
        self.pad_threshold = 0 
        self.zero_traces_less_than = -np.inf
        #TODO: implement clipping
        
        self.pad_gain_match_uncertainty = 0 #unitless
        self.other_systematics = 0 #adc counts

        #load SRIM table for particle. These need to be reloaded if gas desnity is changed.
        self.load_srim_table(particle, gas_density)
        
        #parameters for grid size and other numerics
        self.points_per_bin = 1
        #number of points at which to compute 1D energy deposition
        self.num_stopping_power_points = 50 
        self.adaptive_stopping_power = True #if true, will compute number of stopping poer points based on points per bin and track length

        self.padxy = np.loadtxt('raw_viewer/padxy.txt', delimiter=',')
        self.xy_to_pad = {tuple(np.round(self.padxy[pad], 1)):pad for pad in range(len(self.padxy))}
        self.pad_to_xy = {a: b for b, a in self.xy_to_pad.items()}
        self.pad_width = 2.2 #mm

        #event parameters
        self.initial_energy = 1. #MeV
        self.initial_point = [0.,0.,0.] #(x,y,z) mm. z coordinate only effects peak position in trace.
        self.theta, self.phi = 0.,0. #angles describing direction in which emmitted particle travels, in radians

        #dictionary containing energy deposition on each pad as a function of z coordinate
        self.traces_to_fit = {} #trace data to try to fit. Populated by calling self.set_real_data
        self.timing_offsets = None #Optional dictionary of per pad timing offsets to simulate. 
        self.sim_traces = {} #simulated traces.
        self.pads_to_sim = [pad for pad in self.pad_to_xy] #can be set automatically when loading real traces
        self.num_trace_bins = 512 #set to length of trimmed traces when trimmed traces are loaded

        self.pad_plane = np.genfromtxt('raw_viewer/PadPlane.csv', delimiter=',', 
                                       filling_values=-1, dtype=np.int64) #used for mapping pad numbers to a 2D grid
        self.pad_to_xy_index = {} #maps pad number to (x_index,y_index)
        for y in range(len(self.pad_plane)):
            for x in range(len(self.pad_plane[0])):
                pad = self.pad_plane[x,y]
                if pad != -1:
                    self.pad_to_xy_index[int(pad)] = (x,y)
    
    def get_num_stopping_points_for_energy(self, E):
        to_return = int(np.ceil(self.points_per_bin*self.srim_table.get_stopping_distance(E)/np.min((self.pad_width, self.zscale))))
        if to_return < self.points_per_bin:
            return self.points_per_bin
        return to_return
        
    def load_srim_table(self, particle:str, gas_density:float):
        '''
        Reload SRIM table
        particle: proton or alpha
        gas density: mg/cm^3
        '''
        self.particle = particle
        self.gas_density = gas_density
        if particle.lower() == 'proton':
            self.srim_table = srim_interface.SRIM_Table('track_fitting/H_in_P10.txt', gas_density)
        elif particle.lower() == 'alpha':
            self.srim_table = srim_interface.SRIM_Table('track_fitting/He_in_P10.txt', gas_density)
        else:
            assert False

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
        stopping_distance = self.srim_table.get_stopping_distance(self.initial_energy)
        if self.adaptive_stopping_power:
            self.num_stopping_power_points = self.get_num_stopping_points_for_energy(self.initial_energy)
        distances = np.linspace(0, stopping_distance, self.num_stopping_power_points+1)
        energy_remaining = self.srim_table.get_energy_w_stopping_distance(stopping_distance - distances)
        energy_deposition = energy_remaining[0:-1] - energy_remaining[1:]
        distances = (distances[0:-1] + distances[1:])/2
        return distances, energy_deposition

    def simulate_event(self):
        '''
        
        '''
        #TODO: do a better job of veto pads
        time1=time.time()
        self.distances, energy_deposition = self.get_energy_deposition()
        # Compute the points where energy is evaluated
        direction_vector = np.array((np.sin(self.theta) * np.cos(self.phi), 
                                              np.sin(self.theta) * np.sin(self.phi), 
                                              np.cos(self.theta)))
        #get positions at which energy should be deposited in 3d
        points = np.zeros((self.num_stopping_power_points,3))
        for i in range(3):
            points[:,i] = self.initial_point[i] + direction_vector[i]*self.distances
        time2 = time.time()

        zs = np.arange(self.num_trace_bins)*self.zscale
        #loop over pads, and calculate energy deposition in each bin
        self.sim_traces = {pad:np.zeros(self.num_trace_bins) for pad in self.pads_to_sim}
        erf_dict = {}
        def erf(val):
            if val not in erf_dict:
                erf_dict[val] = scipy.special.erf(val)
            return erf_dict[val]
        
        erf_array = np.vectorize(erf)

        for point, edep in zip(points, energy_deposition):
            if self.timing_offsets == None:
                dz = zs - point[2]
                zfrac = 0.5*(erf_array((dz + self.zscale)/np.sqrt(2*self.sigma_z)) - erf_array(dz/np.sqrt(2*self.sigma_z)))
            else:
                zfrac_dict = {} #used to avoid calculating the array of zfracs multiple times for the same timing offset
            for pad in self.pads_to_sim:
                if self.timing_offsets != None:
                    if pad not in self.timing_offsets:
                        continue #this pad never fired in any event
                    if self.timing_offsets[pad] not in zfrac_dict:
                        dz = zs - (point[2] + self.timing_offsets[pad]*self.zscale)
                        zfrac_dict[self.timing_offsets[pad]] = 0.5*(erf_array((dz + self.zscale)/np.sqrt(2*self.sigma_z)) - erf_array(dz/np.sqrt(2*self.sigma_z)))
                    zfrac = zfrac_dict[self.timing_offsets[pad]]
                dx = self.pad_to_xy[pad][0] - point[0]
                xfrac = 0.5*(erf((dx + self.pad_width/2)/np.sqrt(2*self.sigma_xy)) - \
                             erf((dx - self.pad_width/2)/np.sqrt(2*self.sigma_xy)))
                dy = self.pad_to_xy[pad][1] - point[1]
                yfrac = 0.5*(erf((dy + self.pad_width/2)/np.sqrt(2*self.sigma_xy)) - \
                             erf((dy - self.pad_width/2)/np.sqrt(2*self.sigma_xy)))
                self.sim_traces[pad] += edep *xfrac*yfrac*zfrac*self.counts_per_MeV
            #adc_correction_factor = 1+1.558e-1 - 2.968e-5*trace
        for pad in self.pads_to_sim:
            self.sim_traces[pad][self.sim_traces[pad]<self.zero_traces_less_than] = 0
        time3 = time.time()
        
        
        if self.enable_print_statements:
            print("Time for energy deposition and finding points: ", time2 - time1)
            print("Time to compute traces: ", time3 - time2)


    def get_xyze(self, threshold=-np.inf, traces=None):
        '''
        returns x,y,z,e arrays, similar to the same method in raw_h5_file
        
        source: can be 'energy grid', 'pad map', or 'aligned'
        threshold: only bins with more than this much energy deposition (in MeV) will be returned
        traces: If none, use simulated traces dictionary. Otherwise, use passed in trace dict.
        '''
        if traces == None:
            traces = self.sim_traces
        xs, ys, es = [],[],[]
        for pad in traces:
            x,y = self.pad_to_xy[pad]
            xs.append(x)
            ys.append(y)
            es.append(traces[pad])
        num_z_bins = self.num_trace_bins
        xs = np.repeat(xs, num_z_bins)
        ys = np.repeat(ys, num_z_bins)
        es = np.array(es).flatten()
        z_axis = np.arange(self.num_trace_bins)*self.zscale
        zs = np.tile(z_axis, int(len(xs)/len(z_axis)))
        if threshold != -np.inf:
            xs = xs[es>threshold]
            ys = ys[es>threshold]
            zs = zs[es>threshold]
            es = es[es>threshold]
        return xs, ys, zs, es
    
    def get_adjacent_pads(self, pad):
        x,y = self.pad_to_xy_index[pad]
        to_return = []
        for dx in [-1,0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                if x +dx < 0 or x + dx >= np.shape(self.pad_plane)[0]:
                    continue
                if y +dy < 0 or y + dx >= np.shape(self.pad_plane)[1]:
                    continue
                candidate = self.pad_plane[x+dx,y+dy]
                if candidate != -1:
                    to_return.append(candidate)
        return to_return

    def get_observed_energy_from_ic(self):
        '''
        Returns energy of event being fit based on integrated charge
        '''
        to_return = 0
        for pad in self.traces_to_fit:
            to_return += np.sum(self.traces_to_fit[pad])/self.counts_per_MeV
        return to_return

    def set_real_data(self, pads, traces, trim_threshold, trim_pad = 5, pads_to_sim_select='adjacent'):
        '''
        Prepares real pad traces for coomparison to simulated data.
        This function does the following:
        1. Stores traces to member variables
        2. Trim traces if possible, keeping the length of all traces the same but the portion of regions where 
           all traces are less than fit threshold. Sets self.num_trace_bins to length of trimmed traces
        3. Set self.pads_to_sim based on pads_to_sim_select

        pads: list of pads
        traces: list of traces, one for each pad
        trim_threshold: only portions of the traces above this threshold will be used when fitting
        trim_pad: number of elements to leave on each side of the trimmed traces
        pads_to_sim_select:
          -- unchanged: don't update
          -- observed: only those pads which actually fired
          -- adjacent: pads which fired as well as adjacent pads
        '''
        #steps 1 & 2
        self.traces_to_fit = {pad: trace for pad, trace in zip(pads, traces)}
        trim_before = 512 #will be set to the first non-zero time bin in any trace
        trim_after = -1
        #perform thresholding and find indecies for trimming
        for pad in self.traces_to_fit:
            trace = self.traces_to_fit[pad]
            above_threshold_bins = np.nonzero(trace >= trim_threshold)
            if len(above_threshold_bins[0]) >0:
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
            self.traces_to_fit[pad][self.traces_to_fit[pad] < self.zero_traces_less_than] = 0
        self.num_trace_bins = trim_end - trim_start
        #select pads for simulation
        assert pads_to_sim_select in ['adjacent', 'observed', 'unchanged']
        if pads_to_sim_select == 'adjacent' or pads_to_sim_select == 'observed':
            self.pads_to_sim = list(pads)
        if pads_to_sim_select == 'adjacent':
            for pad in pads:
                for adj_pad in self.get_adjacent_pads(pad):
                    if adj_pad not in self.pads_to_sim:
                        self.pads_to_sim.append(adj_pad)

    def get_residuals(self):
        sim_trace_dict = self.sim_traces
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
        self.pad_ll = {}
        start_time = time.time()
        to_return = 0
        for pad in  self.pad_to_xy: #iterate over all pads, regardless of if they fired
            pad_ll = 0
            if pad in self.sim_traces: #if the pad trace was simulated
                cov_matrix = np.matrix(np.zeros((self.num_trace_bins, self.num_trace_bins)))
                for i in range(self.num_trace_bins):
                    for j in range(self.num_trace_bins):
                        cov_matrix[i,j] = self.pad_gain_match_uncertainty**2*self.sim_traces[pad][i]*self.sim_traces[pad][j]
                        if i ==j:
                            cov_matrix[i,j] += self.other_systematics**2
                if pad in self.traces_to_fit: #pad fired and was simulated
                    residuals = self.sim_traces[pad] - self.traces_to_fit[pad]
                    pad_ll -= self.num_trace_bins*0.5*np.log(2*np.pi)
                    pad_ll -= 0.5*np.log(np.linalg.det(cov_matrix))
                    residuals = np.matrix(residuals)
                    pad_ll -= 0.5*(residuals*(cov_matrix**-1)*residuals.T)[0]
                else: #pad was simulated firing, but did not
                    #if trace < self.pad_threshold, pad would not have fired. Use probability that time bin at top of simulated peak is less than
                    #pad threshold
                    #TODO: should I consider more than just the peak time bin?
                    sim_peak_height = np.max(self.sim_traces[pad])
                    peak_sigma = (self.other_systematics**2 + (self.pad_gain_match_uncertainty*sim_peak_height)**2)**0.5
                    pad_ll = np.log(0.5 + 0.5*scipy.special.erf((self.pad_threshold - sim_peak_height)/2**0.5/peak_sigma))

            else: #pad was not simulated
                assert False #this case should never happen anymore, but might need to add it back in if I add adaptive charge spreading
                pad_ll -= 0.5*self.num_trace_bins*np.log(np.sqrt(2*np.pi*self.other_systematics**2))
                if pad in self.traces_to_fit: #pad fired, but was not simulated
                    residuals = np.matrix(-self.traces_to_fit[pad])
                    pad_ll -= 0.5*(residuals*residuals.T)[0]/self.other_systematics**2
            to_return += pad_ll
            self.pad_ll[pad] = pad_ll

        if self.enable_print_statements:
            print('likelihood time: %f s'%(time.time() - start_time))
        return to_return[0,0]
    
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

    def plot_xyze(self, xs, ys, zs, es, title='', threshold=-np.inf):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        # Plot the 3D scatter plot with energy values as color
        xs = xs[es>threshold]
        ys = ys[es>threshold]
        zs = zs[es>threshold]
        es = es[es>threshold]
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

    def plot_simulated_3d_data(self,  title='simulated_data', threshold=-np.inf): #show plots of initial guess
        self.plot_xyze(*self.get_xyze(threshold), title, threshold)

    def plot_real_data_3d(self, title='observed_data', threshold=-np.inf):
        self.plot_xyze(*self.get_xyze(threshold, traces=self.traces_to_fit), title, threshold)

    
    def plot_residuals_3d(self, title='residuals', threshold=0):
        #in this case treshold is applied to absolute value
        xs, ys, zs, es = self.get_residuals_xyze()
        xs = xs[np.abs(es)>threshold]
        ys = ys[np.abs(es)>threshold]
        zs = zs[np.abs(es)>threshold]
        es = es[np.abs(es)>threshold]
        self.plot_xyze(xs, ys, zs, es, title)

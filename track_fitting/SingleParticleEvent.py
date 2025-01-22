'''
@author bjain and aadams
'''
import os
import time

import numpy as np

from track_fitting import srim_interface
from track_fitting.SimulatedEvent import SimulatedEvent


class SingleParticleEvent(SimulatedEvent):
    '''
    Class for simulating detector response to a single charged particle.
    '''

    def __init__(self, gas_density, particle):
        '''
        gas_density: density in mg/cm^3
        '''
        super().__init__()
        self.particle = particle #this variable should only be changed using the load_srim_table function
        self.gas_density = gas_density  #this variable should only be changed using the load_srim_table function
        
        #load SRIM table for particle. These need to be reloaded if gas desnity is changed.
        self.load_srim_table(particle, gas_density)
        
        #parameters for grid size and other numerics
        self.points_per_bin = 1
        #number of points at which to compute 1D energy deposition
        self.num_stopping_power_points = 50 
        self.adaptive_stopping_power = True #if true, will compute number of stopping poer points based on points per bin and track length

        #event parameters
        self.initial_energy = 1. #MeV
        self.initial_point = [0.,0.,0.] #(x,y,z) mm. z coordinate only effects peak position in trace.
        self.theta, self.phi = 0.,0. #angles describing direction in which emmitted particle travels, in radians

    
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
            self.srim_table = srim_interface.SRIM_Table('track_fitting/stopping_powers/1H_in_P10.txt', gas_density, 'track_fitting/ionization_fractions/1H_in_P10_ionization.csv')
        elif particle.lower() == 'alpha':
            self.srim_table = srim_interface.SRIM_Table('track_fitting/stopping_powers/4He_in_P10.txt', gas_density, 'track_fitting/ionization_fractions/4He_in_P10_ionization.csv')
        else:
            assert False

    

    def get_energy_deposition(self):
        '''
        Return energy deposition vs distance.
        returns distances, energy deposition
        '''
        #TODO: do a better job of veto pads
        time1=time.time()
        stopping_distance = self.srim_table.get_stopping_distance(self.initial_energy)
        if self.adaptive_stopping_power:
            self.num_stopping_power_points = self.get_num_stopping_points_for_energy(self.initial_energy)
        distances = np.linspace(0, stopping_distance, self.num_stopping_power_points+1)
        energy_remaining = self.srim_table.get_energy_w_stopping_distance(stopping_distance - distances)
        ionization_remaining = self.srim_table.get_energy_as_ionization(energy_remaining) #energy yet to be deposited as ionization
        energy_deposition = ionization_remaining[0:-1] - ionization_remaining[1:]
        distances = (distances[0:-1] + distances[1:])/2

        
        self.distances, energy_deposition = distances, energy_deposition
        # Compute the points where energy is evaluated
        direction_vector = np.array((np.sin(self.theta) * np.cos(self.phi), 
                                              np.sin(self.theta) * np.sin(self.phi), 
                                              np.cos(self.theta)))
        #get positions at which energy should be deposited in 3d
        points = np.zeros((self.num_stopping_power_points,3))
        for i in range(3):
            points[:,i] = self.initial_point[i] + direction_vector[i]*self.distances
        time2 = time.time()
        return points, energy_deposition


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
    
    



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

    def __init__(self, gas_density:float, particle:str, material:str):
        '''
        gas_density: density in mg/cm^3
        '''
        super().__init__()
        self.particle = particle #this variable should only be changed using the load_srim_table function
        self.material = material
        self.gas_density = gas_density  #this variable should only be changed using the load_srim_table function
        
        #load SRIM table for particle. These need to be reloaded if gas desnity is changed.
        self.load_srim_table(particle, material, gas_density)
        
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
    
    def gui_before_sim(self):
        self.load_srim_table(self.particle, self.material, self.gas_density)
        
    def load_srim_table(self, particle:str, material:str, gas_density:float):
        '''
        Reload SRIM table
        gas density: mg/cm^3
        '''
        self.particle = particle
        self.material = material
        self.gas_density = gas_density
        stopping_power_path = 'track_fitting/stopping_powers/%s_in_%s.txt'%(particle, material)
        ionization_path = 'track_fitting/ionization_fractions/%s_in_%s_ionization.csv'%(particle, material)
        self.srim_table = srim_interface.SRIM_Table(stopping_power_path, gas_density, ionization_path)

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

        
        self.distances,  self.energy_deposition = distances, energy_deposition
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


    
    
    



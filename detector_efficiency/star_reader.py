# -*- coding: utf-8 -*-
"""
Reads in files downloaded from
https://www.nist.gov/pml/stopping-power-range-tables-electrons-protons-and-helium-ions

When downloading data, should select "space selected" and put a check mark in
the total stopping power collumn.

All calcualtions in this file use continues stopping approximation, which may
signifigantly overpredict stopping distance for low energy particles

@author: Alex Adams
"""

import pandas as pd
import numpy as np
import physical_constants as pc

class Material:
    def __init__(self, filename, density):
        table = pd.read_csv(filename, sep=' ', header=None).to_numpy()[:,0:2]
        self.energies = table[:, 0]*pc.MeV
        self.stopping_powers = table[:, 1]*pc.MeV*pc.cm**2/pc.g
        #interpolation will be done on ln(energy) and ln(stopping power)
        self.ln_energy = np.log(self.energies)
        self.ln_stopping_powers = np.log(self.stopping_powers)
        self.density = density
        
    def get_stopping_power(self, energy):
        '''
        Energy in MeV. Returns stopping power in energy/distance
        '''
        #return np.exp(np.interp(np.log(energy), self.ln_energy, self.ln_stopping_powers))
        return np.interp(energy, self.energies, self.stopping_powers)*self.density
    
    def get_stopping_distance(self, energy):
        '''
        Returns distance of material required to start particle with energy in MeV
        '''
        dE = 0.0001*pc.MeV
        x = 0
        while energy > 0:
            energy -= dE
            x += dE/self.get_stopping_power(energy)
        return x

class CompositeMaterial(Material):
    def __init__(self, filenames, densities):
        self.materials = [Material(fn, d) for fn,d in zip(filenames, densities)]
        
    def get_stopping_power(self, energy):
        return sum([m.get_stopping_power(energy) for m in self.materials])
   
'''
ap = Material('ArgonProton.txt', 1*pc.g/pc.cm**3)
print(ap.get_stopping_distance(1*pc.MeV)/pc.cm)
'''

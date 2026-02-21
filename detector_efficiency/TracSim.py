# -*- coding: utf-8 -*-
"""
Simulate tracks in a material

@author: Alex Adams
"""

import star_reader
import physical_constants as pc
import numpy as np
import matplotlib.pylab as plt

def isotropic3d(length):
    '''
    Gets a random point on a sphere of radius "length"
    '''
    v=np.random.normal(size=3)
    v = v/np.dot(v,v)**0.5
    return v*length
    
class TPCVolume:
    def __init__(self, material, volume_definition, source_distribution):
        '''
        material: star_reader.Material object
        volume_definition: function that takes an (x,y,z) tuple and returns true if the point is inside the volume
        source_distribution: function that returns (x,y,z) for source position when called
        '''
        self.material = material
        self.volume_def = volume_definition
        self.source_distribution = source_distribution
        #variables used to avoid unecessary integratio of path length
        self.last_energy = -1
        self.last_length = -1
        
    def sample_endpoint(self, energy):
        if self.last_energy == energy:
            path_length = self.last_length
        else:
            path_length = self.material.get_stopping_distance(energy)
            self.last_energy = energy
            self.last_length = path_length
        
        starting_point = np.array(self.source_distribution())
        path = isotropic3d(path_length)
        #print(path, path_length)
        return starting_point + path
    
    def get_efficiency(self, energy, samples=40000):
        samples_inside = 0.
        
        i = 0
        while i < samples:
            i += 1
            if self.volume_def(self.sample_endpoint(energy)):
                samples_inside += 1.
        
        return samples_inside/samples

class DiffusionTPCVolume(TPCVolume):
    def __init__(self, material, volume_definition, source_distribution, width):
        TPCVolume.__init__(self, material, volume_definition, source_distribution)
        self.width = width
    def sample_endpoint(self, energy):
        if self.last_energy == energy:
            path_length = self.last_length
        else:
            path_length = self.material.get_stopping_distance(energy) + self.width
            self.last_energy = energy
            self.last_length = path_length
        
        
        starting_point = np.array(self.source_distribution())
        path = isotropic3d(path_length)
        return starting_point + path
    
def gadget2_active_volume(rbar):
    '''
    Let the z axis be along the detector
    Assume detector is 40 cm long with 10 cm diameter (so 5cm radius)
    Origin at point where beam eneters detector
    '''
    x,y,z = rbar
    r = (x*x + y*y)**0.5
    return (r<4*pc.cm) and (z>0) and (z<40*pc.cm)

def beam_60Ga():
    '''
    Generates points for beam starting point, using inputs from LISE++ simulation.
    Will use gaussian distributions, but then assert beam stops inside Gadget
    '''
    z = -1
    x = -1
    y = -1
    while not gadget2_active_volume([x,y,z]):
        sigma_z = 4.914e+01*pc.mm
        mu_z = 2.07e2*pc.mm
        z = np.random.normal(mu_z, sigma_z)
        
        sigma_x = 2.69*pc.mm
        mu_x = 0.39*pc.mm
        x  = np.random.normal(mu_x, sigma_x)
        
        sigma_y = 21.3*pc.mm
        mu_y = 5.86*pc.mm
        y = np.random.normal(mu_y, sigma_y)
        
    return np.array([x,y,z])

def uniform_distribution(r=4*pc.cm, zmin=0, zmax=40*pc.cm):
    z = np.random.sample()*(zmax - zmin) + zmin
    theta = np.random.sample()*2*np.pi
    r_ = np.random.sample()**0.5*r
    x = r_*np.cos(theta)
    y = r_*np.sin(theta)
    return np.array([x,y,z])
    


P_tot = 800*pc.torr
T = 300.
R_ar= pc.Runiv/39.948
R_methane = pc.Runiv/(12.01+4*1.01)
rho_ar = P_tot*0.9/R_ar/T
rho_methane = P_tot*0.1/R_methane/T

P10_alpha = star_reader.CompositeMaterial(['MethaneAlpha.txt','ArgonAlpha.txt'], [rho_methane,rho_ar])
P10_proton = star_reader.CompositeMaterial(['MethaneProton.txt','ArgonProton.txt'], [rho_methane,rho_ar])

source_distribution = uniform_distribution
gadget_alpha = TPCVolume(P10_alpha, gadget2_active_volume, source_distribution)
gadget_proton = TPCVolume(P10_proton, gadget2_active_volume, source_distribution)

if False:
    width = 10*pc.mm
    gadget_alpha_width = DiffusionTPCVolume(P10_alpha, gadget2_active_volume, source_distribution, width)
    gadget_proton_width = DiffusionTPCVolume(P10_proton, gadget2_active_volume, source_distribution, width)
    
    energies = np.linspace(0*pc.MeV, 8*pc.MeV)
    alpha_efficiency = np.array([gadget_alpha.get_efficiency(energy) for energy in energies])
    proton_efficiency = np.array([gadget_proton.get_efficiency(energy) for energy in energies])
    alpha_efficiency_width = np.array([gadget_alpha_width.get_efficiency(energy) for energy in energies])
    proton_efficiency_width = np.array([gadget_proton_width.get_efficiency(energy) for energy in energies])


    plt.figure()
    plt.title('%d torr'%(P_tot/pc.torr))
    plt.plot(energies/pc.MeV, alpha_efficiency*100, '.', label='alpha, width=0 mm')
    plt.plot(energies/pc.MeV, alpha_efficiency_width*100, '.', label='alpha, width=%f mm'%(width/pc.mm))
    plt.legend()
    plt.xlabel('energy (MeV)')
    plt.ylabel('% stopped in active volume')
    
    plt.figure()
    plt.title('%d torr'%(P_tot/pc.torr))
    plt.plot(energies/pc.MeV, proton_efficiency*100, '.', label='proton, width = 0 mm')
    plt.plot(energies/pc.MeV, proton_efficiency_width*100, '.', label='proton, width=%f mm'%(width/pc.mm))
    plt.legend()
    plt.xlabel('energy (MeV)')
    plt.ylabel('% stopped in active volume')

if True:
    energies = np.linspace(6*pc.MeV, 6.5*pc.MeV)
    plt.figure()
    plt.title("alpha stopping in %d torr P10 gas"%(P_tot/pc.torr))
    plt.xlabel('energy (MeV')
    plt.ylabel('stopping distance (mm)')
    alpha_stopping_distances = np.array([P10_alpha.get_stopping_distance(E) for E in energies])
    plt.plot(energies/pc.MeV, alpha_stopping_distances*1e3)
    
    plt.figure()
    plt.title("proton stopping in %d torr P10 gas"%(P_tot/pc.torr))
    plt.xlabel('energy (MeV')
    plt.ylabel('stopping distance (mm)')
    proton_stopping_distances = np.array([P10_proton.get_stopping_distance(E) for E in energies])
    plt.plot(energies/pc.MeV, proton_stopping_distances*1e3)
    
    plt.figure()
    plt.title("stopping distances in %d torr P10 gas"%(P_tot/pc.torr))
    plt.xlabel('energy (MeV')
    plt.ylabel('stopping distance (mm)')
    proton_stopping_distances = np.array([P10_proton.get_stopping_distance(E) for E in energies])
    plt.plot(energies/pc.MeV, proton_stopping_distances*1e3, label='proton')
    plt.plot(energies/pc.MeV, alpha_stopping_distances*1e3, label='alpha')
    plt.legend()
    plt.yscale('log')
    
    
    

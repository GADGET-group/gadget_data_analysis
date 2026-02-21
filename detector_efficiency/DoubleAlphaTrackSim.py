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
    def __init__(self, material, volume_definition, source_distribution, mode='back_to_back'):
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
        self.mode = mode
        
    def sample_endpoint(self, energy):
        if self.last_energy == energy:
            path_length = self.last_length
        else:
            path_length = self.material.get_stopping_distance(energy)
            self.last_energy = energy
            self.last_length = path_length
        
        starting_point = np.array(self.source_distribution())
        path = isotropic3d(path_length)
        if(self.mode == 'back_to_back'):
            return (starting_point + path, starting_point - path)
        elif(self.mode == 'isotropic'):
             path2 = isotropic3d(path_length)
             return (starting_point + path, starting_point +path2)
        else:
             assert False
    
    def get_efficiency(self, energy, samples=40000):
        samples_inside = 0.
        i = 0
        while i < samples:
            i += 1
            endpoints = self.sample_endpoint(energy)
            if self.volume_def(endpoints[0]) and self.volume_def(endpoints[1]):
                samples_inside += 1.
        
        return samples_inside/samples
    
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


def do_validation():
    R = 100000
    L = 200*pc.mm
    E = 6.5*pc.MeV

    def test_volume(rbar):
        x,y,z = rbar
        r = (x*x + y*y)**0.5
        return (r<R) and (z>0) and (z<L)
    P_tot = 800*pc.torr
    T = 300.
    R_ar= pc.Runiv/39.948
    R_methane = pc.Runiv/(12.01+4*1.01)
    rho_ar = P_tot*0.9/R_ar/T
    rho_methane = P_tot*0.1/R_methane/T
    
    P10_alpha = star_reader.CompositeMaterial(['MethaneAlpha.txt','ArgonAlpha.txt'], [rho_methane,rho_ar])
    gadget_alpha = TPCVolume(P10_alpha, test_volume, lambda: uniform_distribution(R, 0, L), mode='back_to_back')
    
    print('simulated_efficiency: ', gadget_alpha.get_efficiency(E))
    l = P10_alpha.get_stopping_distance(E)
    print("Joe and Alex's math equation for what the efficiency should be: ", 1 - l/L)
    
E = 6.5*pc.MeV
pressures =  np.linspace(1*pc.torr, 5000*pc.torr)
effs_iso = []
effs_b2b = []
for P_tot in pressures:
    T = 300.
    R_ar= pc.Runiv/39.948
    R_methane = pc.Runiv/(12.01+4*1.01)
    rho_ar = P_tot*0.9/R_ar/T
    rho_methane = P_tot*0.1/R_methane/T
    
    P10_alpha = star_reader.CompositeMaterial(['MethaneAlpha.txt','ArgonAlpha.txt'], [rho_methane,rho_ar])
    gadget_alpha = TPCVolume(P10_alpha, gadget2_active_volume, uniform_distribution)
    effs_b2b.append(gadget_alpha.get_efficiency(E))
    
    P10_alpha = star_reader.CompositeMaterial(['MethaneAlpha.txt','ArgonAlpha.txt'], [rho_methane,rho_ar])
    gadget_alpha = TPCVolume(P10_alpha, gadget2_active_volume, uniform_distribution, mode='isotropic')
    effs_iso.append(gadget_alpha.get_efficiency(E))
<<<<<<< HEAD
   
plt.figure(figsize=(17,10))
plt.plot(pressures/pc.torr, effs_b2b, label='back to back')
plt.plot(pressures/pc.torr, effs_iso, label='isotropic')
=======

plt.figure()
plt.scatter(pressures/pc.torr, effs_b2b, label='back to back')
plt.scatter(pressures/pc.torr, effs_iso, label='isotropic')
plt.xlabel('pressure (torr)')
plt.ylabel('efficiency')
>>>>>>> 6a1ca51906d8858ff1c565d90117495e56250213
plt.legend()

'''
P10_alpha = star_reader.CompositeMaterial(['MethaneAlpha.txt','ArgonAlpha.txt'], [rho_methane,rho_ar])
P10_proton = star_reader.CompositeMaterial(['MethaneProton.txt','ArgonProton.txt'], [rho_methane,rho_ar])

gadget_alpha = TPCVolume(P10_alpha, gadget2_active_volume, beam_60Ga)
gadget_proton = TPCVolume(P10_proton, gadget2_active_volume, beam_60Ga)

energies = np.linspace(0*pc.MeV, 8*pc.MeV)
alpha_efficiency = np.array([gadget_alpha.get_efficiency(energy) for energy in energies])
proton_efficiency = np.array([gadget_proton.get_efficiency(energy) for energy in energies])
plt.figure()
plt.plot(energies/pc.MeV, alpha_efficiency*100, label='alpha')
plt.plot(energies/pc.MeV, proton_efficiency*100, label='proton')
plt.legend()
plt.xlabel('energy (MeV)')
plt.ylabel('% stopped in active volume')

energies = np.linspace(0.5*pc.MeV, 8*pc.MeV)
plt.figure()
plt.title("alpha stopping in 800 torr P10 gas")
plt.xlabel('energy (MeV')
plt.ylabel('stopping distance (mm)')
alpha_stopping_distances = np.array([P10_alpha.get_stopping_distance(E) for E in energies])
plt.plot(energies/pc.MeV, alpha_stopping_distances*1e3)

plt.figure()
plt.title("proton stopping in 800 torr P10 gas")
plt.xlabel('energy (MeV')
plt.ylabel('stopping distance (mm)')
proton_stopping_distances = np.array([P10_proton.get_stopping_distance(E) for E in energies])
plt.plot(energies/pc.MeV, proton_stopping_distances*1e3)

plt.figure()
plt.title("stopping distances in 800 torr P10 gas")
plt.xlabel('energy (MeV')
plt.ylabel('stopping distance (mm)')
proton_stopping_distances = np.array([P10_proton.get_stopping_distance(E) for E in energies])
plt.plot(energies/pc.MeV, proton_stopping_distances*1e3, label='proton')
plt.plot(energies/pc.MeV, alpha_stopping_distances*1e3, label='alpha')
plt.legend()
plt.yscale('log')
'''
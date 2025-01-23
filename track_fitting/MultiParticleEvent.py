import numpy as np

from track_fitting.SingleParticleEvent import SingleParticleEvent
from track_fitting.SimulatedEvent import SimulatedEvent

class MultiParticleEvent(SimulatedEvent):
    def __init__(self, sims):
        super().__init__()#this class won't use it's own SRIM table, so particle and density chosen don't matter
        self.sims = sims
        self.gas_density = 1.
        self.update_configuration()

    def update_configuration(self):
        for sim in self.sims:
            sim.load_srim_table(sim.particle, self.gas_density)

    def get_energy_deposition(self):
        points, edeps = [],[]
        for sim in self.sims:
            p,e = sim.get_energy_deposition()
            points.append(p)
            edeps.append(e)
        return np.concatenate(points), np.concatenate(edeps)


class MultiparticleDecay(MultiParticleEvent):
    '''
    Decay with multible particles emmitted and a recoiling nucleus which is treated as a point energy deposition.
    '''
    pass

class ProtonAlphaEvent(MultiParticleEvent):
    '''
    class for compatibility with sim_gui
    '''
    def __init__(self, proton:SingleParticleEvent, alpha:SingleParticleEvent):
        super().__init__([proton, alpha])
        self.proton = proton
        self.alpha = alpha
        self.per_particle_params = ['initial_energy', 'theta', 'phi', 'adaptive_stopping_power', 'num_stopping_power_points']
        self.shared_params = ['initial_point', 'sigma_xy', 'sigma_z', 'pad_threshold', 
                              'counts_per_MeV', 'other_systematics', 'pad_gain_match_uncertainty'] 
        for param in self.per_particle_params:
            self.__dict__['alpha_' + param] = alpha.__dict__[param]
            self.__dict__['proton_' + param] = proton.__dict__[param]
        for param in self.shared_params:
            self.__dict__[param] = proton.__dict__[param]
    
    def simulate_event(self):
        for param in self.per_particle_params:
            self.alpha.__dict__[param] = self.__dict__['alpha_' + param] 
            self.proton.__dict__[param] = self.__dict__['proton_' + param]
        for param in self.shared_params:
            self.proton.__dict__[param] = self.__dict__[param]
            self.alpha.__dict__[param] = self.__dict__[param]
        self.proton.load_srim_table('proton', self.gas_density)
        self.alpha.load_srim_table('alpha', self.gas_density)
        super().simulate_event()
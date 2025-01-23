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


class MultiParticleDecay(MultiParticleEvent):
    '''
    Decay with multible particles emmitted and a recoiling nucleus.
    Uses energy and momentum conservation to automatically update recoil energy and direction when simulating events.
    '''
    def __init__(self, products:list[SingleParticleEvent], product_masses:list[float], recoil:SingleParticleEvent, recoil_mass:float):
        '''
        Mass units for recoil_mass and product_masses just need to have the same units.
        '''
        sims = list(products)
        sims.append(recoil)
        super().__init__(sims)
        self.products, self.product_masses = products, product_masses
        self.recoil, self.recoil_mass = recoil, recoil_mass

    def get_energy_deposition(self):
        #calculate sqrt(recoil energy)*recoil_direction_vector, and use this to update recoil theta, phi, and energy
        v = np.zeros(3)
        for p, m_p in zip(self.products, self.product_masses):
            vhat = np.array([np.sin(p.theta)*np.cos(p.phi), np.sin(p.theta)*np.sin(p.phi), np.cos(p.theta)])
            v += vhat*np.sqrt(p.initial_energy*m_p/self.recoil_mass)
        self.recoil.initial_energy = np.dot(v,v)
        self.recoil.theta = np.arctan2( np.sqrt(v[0]**2 + v[1]**2), v[2])
        self.recoil.phi = np.arctan2(v[1], v[0])
        return super().get_energy_deposition()

class MultiParticleDecay(MultiParticleEvent):
    '''
    class for compatibility with sim_gui
    '''
    def __init__(self, proton:SingleParticleEvent, alpha:SingleParticleEvent, recoil):
        super().__init__([proton, alpha])
        self.proton = proton
        self.alpha = alpha
        self.per_particle_params = ['initial_energy', 'theta', 'phi']
        self.shared_params = ['initial_point', 'sigma_xy', 'sigma_z', 'pad_threshold', 
                              'counts_per_MeV', 'other_systematics', 'pad_gain_match_uncertainty'] 
        for param in self.per_particle_params:
            self.__dict__['alpha_' + param] = alpha.__dict__[param]
            self.__dict__['proton_' + param] = proton.__dict__[param]
            #make recoil parameters show up on GUI for viewing. Of course, any input will be overriden when event is simulated
            self.__dict__['recoil_' + param] = recoil.__dict__[param] 
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
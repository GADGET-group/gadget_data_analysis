import numpy as np

from track_fitting.SingleParticleEvent import SingleParticleEvent
from track_fitting.SimulatedEvent import SimulatedEvent

class MultiParticleEvent(SimulatedEvent):
    def __init__(self, sims:list[SingleParticleEvent]):
        super().__init__()#this class won't use it's own SRIM table, so particle and density chosen don't matter
        self.sims = sims

        #params for display on gui
        self.sim_names = [] #used to label particles on gui
        ptype_dict = {}
        for sim in sims:
            if sim.particle not in ptype_dict:
                ptype_dict[sim.particle] = 0
            else:
                ptype_dict[sim.particle] += 1
            self.sim_names.append('%s_%d'%(sim.particle, ptype_dict[sim.particle]))
        self.per_particle_params = ['initial_energy', 'theta', 'phi', 'num_stopping_power_points'] 
        self.shared_params = ['initial_point', 
                              'gas_density'] 
        
    
    def gui_after_sim(self):
        '''
        Make the gui params reflect the underlying sims.
        self.sims[0] will be used for all shared params.
        '''
        for param in self.shared_params:
<<<<<<< HEAD
            self.proton.__dict__[param] = self.__dict__[param]
            self.alpha.__dict__[param] = self.__dict__[param]
        self.proton.load_srim_table('proton', self.gas_density)
        self.alpha.load_srim_table('alpha', self.gas_density)
        super().simulate_event()

class DoubleAlphaEvent(MultiParticleEvent):
    '''
    class for compatibility with sim_gui
    '''
    def __init__(self, alpha1, alpha2):

        super().__init__([alpha1, alpha2])
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.per_particle_params = ['initial_energy', 'theta', 'phi']
        self.alpha1_initial_point, self.alpha2_initial_point = np.array([0., 0., 0.]), np.array([0., 0., 0.])
        self.shared_params = ['sigma_xy', 'sigma_z', 'pad_threshold', 'adaptive_stopping_power', 'num_stopping_power_points']
        for param in self.per_particle_params:
            self.__dict__['alpha1_' + param] = alpha1.__dict__[param]
            self.__dict__['alpha2_' + param] = alpha2.__dict__[param]
        for param in self.shared_params:
            self.__dict__[param] = alpha1.__dict__[param]
    
    def simulate_event(self):
        for param in self.per_particle_params:
            self.alpha1.__dict__[param] = self.__dict__['alpha1_' + param] 
            self.alpha2.__dict__[param] = self.__dict__['alpha2_' + param]
        for param in self.shared_params:
            self.alpha1.__dict__[param] = self.__dict__[param]
            self.alpha2.__dict__[param] = self.__dict__[param]
        self.alpha1.load_srim_table('alpha', self.gas_density)
        self.alpha2.load_srim_table('alpha', self.gas_density)
        self.alpha1.initial_point = self.alpha1_initial_point
        self.alpha2.initial_point = self.alpha2_initial_point
        super().simulate_event()
=======
            self.__dict__[param] = self.sims[0].__dict__[param]
        
        for prefix,sim in zip(self.sim_names, self.sims):
            for param in self.per_particle_params:
                self.__dict__[prefix + '_' + param] = sim.__dict__[param]
        
        for sim in self.sims:
            sim.gui_after_sim()

    def gui_before_sim(self):
        '''
        Calling this function will update the individual sims to reflect the gui parameters
        '''
        for sim in self.sims:
            sim.gui_before_sim()
        for prefix,sim in zip(self.sim_names, self.sims):
            for param in self.per_particle_params:
                sim.__dict__[param] = self.__dict__[prefix + '_' + param] 
            for param in self.shared_params:
                sim.__dict__[param] = self.__dict__[param]
        

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
        self.initial_point = [0.,0.,0.] #this will always be loaded to children when get_energy_deposition is called

    def get_energy_deposition(self):
        #calculate sqrt(recoil energy)*recoil_direction_vector, and use this to update recoil theta, phi, and energy
        v = np.zeros(3)
        for p, m_p in zip(self.products, self.product_masses):
            p.initial_point = self.initial_point
            vhat = np.array([np.sin(p.theta)*np.cos(p.phi), np.sin(p.theta)*np.sin(p.phi), np.cos(p.theta)])
            v -= vhat*np.sqrt(p.initial_energy*m_p/self.recoil_mass)
        self.recoil.initial_point = self.initial_point
        self.recoil.initial_energy = np.dot(v,v)
        self.recoil.theta = np.arctan2( np.sqrt(v[0]**2 + v[1]**2), v[2])
        self.recoil.phi = np.arctan2(v[1], v[0])
        return super().get_energy_deposition()
>>>>>>> alex_track_fitting

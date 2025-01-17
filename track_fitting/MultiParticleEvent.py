import numpy as np

from track_fitting.SingleParticleEvent import SingleParticleEvent

class MultiParticleEvent(SingleParticleEvent):
    def __init__(self, sims:SingleParticleEvent):
        super().__init__(1, 'proton')#this class won't use it's own SRIM table, so particle and density chosen don't matter
        self.sims = sims

    def set_real_data(self, pads, traces, trim_threshold, trim_pad=5, pads_to_sim_select='adjacent'):
        super().set_real_data(pads, traces, trim_threshold, trim_pad, pads_to_sim_select)
        for sim in self.sims:
            sim.set_real_data(pads, traces, trim_threshold, trim_pad, pads_to_sim_select)


    def simulate_event(self):
        self.sim_traces = {pad:np.zeros(self.num_trace_bins) for pad in self.pads_to_sim}
        for sim in self.sims:
            sim.simulate_event()
            for pad in self.pads_to_sim:
                self.sim_traces[pad] += sim.sim_traces[pad]

class ProtonAlphaEvent(MultiParticleEvent):
    '''
    class for compatibility with sim_gui
    '''
    def __init__(self, proton, alpha):
        super().__init__([proton, alpha])
        self.proton = proton
        self.alpha = alpha
        self.per_particle_params = ['initial_energy', 'theta', 'phi']
        self.shared_params = ['initial_point', 'sigma_xy', 'sigma_z', 'pad_threshold', 'adaptive_stopping_power', 'num_stopping_power_points']
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
from track_fitting.SingleParticleEvent import SingleParticleEvent

class ParticleAndPointDeposition(SingleParticleEvent):
    def __init__(self, gas_density, particle):
        '''
        gas_density: density in mg/cm^3
        '''
        super().__init__(gas_density, particle)

        self.point_energy_deposition = 0 #MeV

    def get_energy_deposition(self):
        distances, energy_dep = super().get_energy_deposition()
        energy_dep[0] += self.point_energy_deposition
        return distances, energy_dep


    
def show_simulated_palpha_track(decay_point, theta, phi, Eproton, Ealpha, charge_spread, shaping_time, pressure):
    rho0 = 1.5256 #mg/cm^3, P10 at 300K and 760 torr
    T = 20+273.15 #K
    density = rho0*(pressure/760)*(300./T)
    sim = ParticleAndPointDeposition(density, 'proton')
    sim.enable_print_statements = True
    sim.theta, sim.phi, sim.initial_point = theta, phi, decay_point
    sim.initial_energy = Eproton
    sim.point_energy_deposition = Ealpha
    sim.charge_spreading_sigma = charge_spread
    sim.shaping_width = shaping_time
    sim.simulate_event()
    sim.plot_xyze(*sim.get_xyze(), energy_threshold=20)
    return sim


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


    

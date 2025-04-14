import numpy as np

class SRIM_Table:
    def __init__(self, data_path:str, material_density:float, ionization_file=None):
        '''
        data_path: path to SRIM file
        material_density: Density of material in mg/cm^3
        ionization file: Optional csv file with ion_energy in keV, fraction_energy_as_ionization. 
                         Ionization fraction is assumed to be 1 if not provided
        '''
        # Initialize lists to store the data
        energy_MeV = [0]
        electronic_stopping_MeV_um = [0]  # Stopping power in MeV/(mg/cm^2)
        nuclear_stopping_MeV_um = [0]
        path_length_mm = [0]
        
        # Read the file
        with open(data_path, 'r') as file:
            for i, line in enumerate(file):
                if i == 10:#read in density at which calculation was done
                    table_density = float(line.split()[3])*1e3#convert g/cm^3 to mg/cm^3
                # Start reading from the 26th line and stop after the 105th line
                if 27 <= i <= 124:
                    if line == '-----------------------------------------------------------':
                        break
                    if line.strip():  # Ensure the line is not empty
                        parts = line.split()
                        # Handle energy conversion based on unit
                        energy_value = float(parts[0])
                        energy_unit = parts[1]
                        if energy_unit == 'keV':
                            energy = energy_value / 1000  # Convert from keV to MeV
                        elif energy_unit == 'MeV':
                            energy = energy_value
                        
                        # Stopping powers are in MeV/(mg/cm^2)
                        elec = float(parts[2])
                        nucl = float(parts[3])
        
                        # Convert path length to mm
                        path_value = float(parts[4])  # Get the number part before the unit
                        path_unit = parts[5]         # Get the unit part
        
                        if path_unit == 'um':
                            path = path_value / 1000  # Convert from um to mm
                        elif path_unit == 'mm':
                            path = path_value
                        elif path_unit == 'm':
                            path = path_value * 1000  # Convert from m to mm
                        
                        # Append to lists
                        energy_MeV.append(energy)
                        electronic_stopping_MeV_um.append(elec)
                        nuclear_stopping_MeV_um.append(nucl)
                        path_length_mm.append(path)
        
        # Convert lists to NumPy arrays
        self.energy_MeV = np.array(energy_MeV)
        electronic_stopping_MeV_um = np.array(electronic_stopping_MeV_um)
        nuclear_stopping_MeV_um = np.array(nuclear_stopping_MeV_um)
        self.stopping_distance_mm = np.array(path_length_mm)/material_density*table_density
        self.stopping_power_MeV_mm = material_density * (electronic_stopping_MeV_um + nuclear_stopping_MeV_um)/10

        if type(ionization_file) != type(None):
            #load data and sort by ascending energy
            ionization_data = np.loadtxt(ionization_file, skiprows=1, delimiter=',')
            ion_Es = ionization_data[:,0]/1000
            ion_frac = ionization_data[:,1]
            sort_i = np.argsort(ion_Es)
            ion_Es, ion_frac = ion_Es[sort_i], ion_frac[sort_i]
            #interpolate to get ionization fraction ateach stopping power energy
            self.ionization_fractions = np.interp(self.energy_MeV, ion_Es, ion_frac)
        else:
            self.ionization_fractions = np.ones(len(self.energy_MeV))
        
    def get_stopping_distance(self, E_MeV):
        '''
        returns stopping distance in mm
        '''
        return np.interp(E_MeV, self.energy_MeV, self.stopping_distance_mm)
    
    def get_energy_w_stopping_distance(self, distance_mm):
        return np.interp(distance_mm, self.stopping_distance_mm, self.energy_MeV)
    
    def get_energy_as_ionization(self, energies):
        return np.interp(energies, self.energy_MeV, self.ionization_fractions)*energies
    
    def get_stopping_power(self, E_MeV):
        '''
        Returns stopping power in MeV/mm
        '''
        return np.interp(E_MeV, self.energy_MeV, self.stopping_power_MeV_mm)

    def get_stopping_power_after_distances(self, E0_MeV, distances_mm):
        '''
        Gets stopping power assuming particle starts at distance = 0. Will be 0 for distance
        less than 0 or once particle stops.
        distances_mm: array of distances at which to get stopping power
        '''
        distance_remaining = self.get_stopping_distance(E0_MeV) - distances_mm
        E_now = self.get_energy_w_stopping_distance(distance_remaining)
        to_return = self.get_stopping_power(E_now)
        to_return[distances_mm < 0] = 0
        #handle case where distance remaining is negative
        to_return[distance_remaining < 0] = 0
        return to_return


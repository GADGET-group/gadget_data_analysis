import numpy as np
from scipy import integrate
from scipy.signal import convolve

gas_density = 1.56 #mg/cm^3
z_scale = 1.74
kernel_size = 31

def trace_3d(initial_energy, initial_coords, theta, phi, particle="Proton"):
    if particle == "Proton":  
        data_path = 'H_in_P10.txt'
    elif particle == "Alpha":
        data_path = 'He_in_P10.txt'
    energies, stopping_powers = load_energies(data_path)
    
    # Make distances where stopping_power is to be calculated
    distances = np.linspace(0,100,500)
    def dEdx(E):
        """
        Differential function to calculate stopping power based on energy.
        """
        to_return = -np.interp(E, energies, stopping_powers)
        to_return[E < 0] = 0
        return to_return

    # Find energy at each distance by integrating
    Es = np.squeeze(integrate.odeint(lambda E, x: dEdx(E), initial_energy, distances))
    
    # Calculate stopping powers
    calculated_stopping_powers = -dEdx(Es)*100/500
    points = compute_points(distances, initial_coords, theta, phi)
    energy_map, x, y, z = create_grid(points, calculated_stopping_powers)
    convolved_energies = convolve_with_gaussian(energy_map, kernel_size)
    # Creating PadPlane positions
    pad_x = np.arange(-38.5,38.5+2.2,2.2)
    pad_y = np.arange(-38.5,38.5+2.2,2.2)
    pad_z = np.arange(0, 512*z_scale, z_scale)
    # Get the indices where energy is not zero for more efficient plotting
    # non_zero_indices = np.nonzero(energy_grid)
    max_value = np.max(convolved_energies)
    significant_mask = convolved_energies > (0.001 * max_value)
    non_zero_indices = np.nonzero(significant_mask)
    x_non_zero = x[non_zero_indices[0]]
    y_non_zero = y[non_zero_indices[1]]
    z_non_zero = z[non_zero_indices[2]]
    energy_non_zero = convolved_energies[non_zero_indices]
    pad_energy_dict = {}

    for i, j, k, e in zip(x_non_zero, y_non_zero, z_non_zero, energy_non_zero):
        # Map x and y to the nearest padplane positions
        pad_x_nearest = pad_x[np.abs(pad_x - i).argmin()]
        pad_y_nearest = pad_y[np.abs(pad_y - j).argmin()]
        pad_z_nearest = pad_z[np.abs(pad_z - k).argmin()]
        if (pad_x_nearest, pad_y_nearest, pad_z_nearest) not in pad_energy_dict:
            pad_energy_dict[(pad_x_nearest, pad_y_nearest, pad_z_nearest)] = e
        else:
            pad_energy_dict[(pad_x_nearest, pad_y_nearest, pad_z_nearest)] += e

    # Extract coordinates and energies from the dictionary
    pad_coords = np.array(list(pad_energy_dict.keys()))
    pad_energies = np.array(list(pad_energy_dict.values()))
    pad_x_coords = pad_coords[:, 0]
    pad_y_coords = pad_coords[:, 1]
    pad_z_coords = pad_coords[:, 2]
    return pad_x_coords, pad_y_coords, pad_z_coords, pad_energies


def load_energies(data_path):
    # Initialize lists to store the data
    energy_MeV = []
    electronic_stopping_MeV_um = []  # Stopping power in MeV/(mg/cm^2)
    nuclear_stopping_MeV_um = []
    path_length_mm = []

    # Read the file
    with open(data_path, 'r') as file:
        for i, line in enumerate(file):
            # Start reading from the 26th line and stop after the 105th line
            if 26 <= i <= 104:
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
    energy_MeV = np.array(energy_MeV)
    electronic_stopping_MeV_um = np.array(electronic_stopping_MeV_um)
    nuclear_stopping_MeV_um = np.array(nuclear_stopping_MeV_um)
    path_length_mm = np.array(path_length_mm)
    stopping_power_converted = gas_density * (electronic_stopping_MeV_um + nuclear_stopping_MeV_um)/10
    return energy_MeV, stopping_power_converted

def compute_points(distances, initial_point, theta, phi):
        """
        Compute 3D points along a given direction from an initial point based on distances.
        """
        direction_vector = np.round(np.array((np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta))), 6)
        points = np.array([np.array(initial_point) + distance * direction_vector for distance in distances])
        return points

def create_grid(coords, stopping_powers, grid_resolution=0.5):
        """
        Create a fine grid for the given coordinate range and resolution.
        grid_resolution (in mm): Resolution of the grid.
        """
        padding = 15

        # Find min and max for each dimension
        min_x, min_y, min_z = np.min(coords, axis=0) - padding
        max_x, max_y, max_z = np.max(coords, axis=0) + padding

        # Create ranges for each dimension with padding
        x = np.arange(min_x, max_x + grid_resolution, grid_resolution)
        y = np.arange(min_y, max_y + grid_resolution, grid_resolution)
        z = np.arange(min_z, max_z + grid_resolution, grid_resolution)
        # Create meshgrid
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        grid_shape = X.shape
        # Calculate indices
        indices = ((coords - [min_x, min_y, min_z]) / grid_resolution).round().astype(int)

        # Initialize energy grid
        energy_grid = np.zeros(grid_shape)

        # Accumulate energy values using advanced indexing
        np.add.at(energy_grid, (indices[:, 0], indices[:, 1], indices[:, 2]), stopping_powers)

        return energy_grid, x, y, z

def convolve_with_gaussian(energy_grid, kernel_size):
    """
    Convolve the energy grid with a 3D Gaussian kernel with varying sigma based on z position.
    """
    def gaussian_kernel_2d(kernel_size, sigma_xy):
        """Create a 2D Gaussian kernel."""
        ax = np.linspace(-(kernel_size - 1) / 2., (kernel_size - 1) / 2., kernel_size)
        xx, yy = np.meshgrid(ax, ax)
        kernel = np.exp(-(xx**2 + yy**2) / (2 * sigma_xy**2))
        return kernel / np.sum(kernel)
    
    def gaussian_kernel_1d(kernel_size, sigma_z):
        """Create a 1D Gaussian kernel."""
        ax = np.linspace(-(kernel_size - 1) / 2., (kernel_size - 1) / 2., kernel_size)
        kernel = np.exp(-ax**2 / (2 * sigma_z**2))
        return kernel / np.sum(kernel)
    
    # Determine the sigma values based on z position
    k_xy = 0.0554 
    k_z = 0.0338
    z_positions = np.arange(energy_grid.shape[2])
    sigma_xy_array = np.sqrt(10) * k_xy * np.sqrt(z_positions)
    sigma_z_array = np.sqrt(10) * k_z * np.sqrt(z_positions)
    
    # Ensure sigma values are positive
    sigma_xy_array = np.clip(sigma_xy_array, 1e-5, None)
    sigma_z_array = np.clip(sigma_z_array, 1e-5, None)

    # Initialize the convolved grid
    convolved_grid = np.zeros_like(energy_grid)

    # Apply the convolution for each z slice
    half_kernel_size = kernel_size // 2

    # Convolve along the XY plane for each Z slice
    for z in range(energy_grid.shape[2]):
        # Create 2D Gaussian kernel for xy-plane
        kernel_2d_xy = gaussian_kernel_2d(kernel_size, sigma_xy_array[z])
        
        # Convolve each slice along the x and y axes
        convolved_grid[:, :, z] = convolve(energy_grid[:, :, z], kernel_2d_xy, mode='same')

    # Convolve along the z-axis
    temp_grid = np.zeros_like(convolved_grid)
    for z in range(energy_grid.shape[2]):
        kernel_1d_z = gaussian_kernel_1d(kernel_size, sigma_z_array[z])
        for dz in range(-half_kernel_size, half_kernel_size + 1):
            kernel_index = dz + half_kernel_size
            
            if 0 <= z + dz < energy_grid.shape[2]:
                temp_grid[:, :, z] += convolved_grid[:, :, z + dz] * kernel_1d_z[kernel_index]

    return temp_grid
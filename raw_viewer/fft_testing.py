import numpy as np
import matplotlib.pylab as plt
from scipy.fft import fft2, rfft2, fftshift
import matplotlib.colors as colors

import raw_h5_file

#configure h5 file interface
raw_file = raw_h5_file.raw_h5_file(file_path='/mnt/analysis/e21072/gastest_h5_files/run_0368.h5',
                                     zscale=0.78125,
                                     flat_lookup_csv='./channel_mappings/flatlookup2cobos.csv')
raw_file.background_subtract_mode='fixed window'
raw_file.num_background_bins = (400, 500)
raw_file.remove_outliers = True
raw_file.data_select_mode = 'near peak'
raw_file.near_peak_window_width = 50
raw_file.require_peak_within = (-np.inf, np.inf)

#show event in euclidian space
image = raw_file.get_2d_image(5)
plt.figure()
plt.imshow(image)

#show event in fourier space
plt.figure()
fft_image = fft2(image)
plt.imshow(np.abs(fftshift(fft_image)))

def gaussian_fft(shape, sigmas):
    '''
    shape: shape of the array to fill
    sigmas: list of sigmas in the spatial domain (x,y,z, etc)
    '''
    gaussians = [] #1D gaussians
    for N, sigma_space in zip(shape, sigmas):
        ks = np.arange(N)
        ks[ks/N>0.5] -= N #fft shift
        sigma_freq = N/(2*np.pi*sigma_space)
        gaussians.append(np.exp(-ks**2/2/sigma_freq**2))
        gaussians[-1] /= np.sum(gaussians[-1])
    return np.prod(np.meshgrid(*gaussians), axis=0)

def sim_fft_image_2d(x0, y0, E0, theta, sigma):
    gaussian = gaussian_fft(np.shape(fft_image), (sigma, sigma))
    


'''
idea for algorithm:
1) take fft of event data cube
2) fit model below to fft taken in step 1, using magnitudes of each voxel only (ignoring phase information).
   This will be insensitive to translations, but sensitive to rotations and energy sharing between particles.
3) Fit model including phase information to get information about origin of decay

Maybe background subtract by just getting rid of 0 frequency component in both signals?
'''
import numpy as np
import matplotlib.pylab as plt
from scipy.fft import fft, fft2, rfft2, fftshift, ifft2, fftfreq
import matplotlib.colors as colors

import sys
#TODO: don't do this!!!
sys.path.append('/mnt/projects/e21072/OfflineAnalysis/analysis_scripts/alex/gadget_analysis/raw_viewer/')
import raw_h5_file

import srim_interface



#configure h5 file interface
raw_file = raw_h5_file.raw_h5_file(file_path='/mnt/analysis/e21072/gastest_h5_files/run_0368.h5',
                                    zscale=0.78125,
                                    flat_lookup_csv='../raw_viewer/channel_mappings/flatlookup2cobos.csv')
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
plt.show()

def gaussian_fft(shape, sigmas, point_spacings):
    '''
    shape: shape of the array to fill
    sigmas: list of sigmas in the spatial domain (x,y,z, etc)
    point_spacings: spacing between spacial points
    '''
    gaussians = [] #1D gaussians
    for N, spacing, sigma_space in zip(shape, point_spacings, sigmas):
        freq = fftfreq(N, spacing)
        sigma_freq = N/(2*np.pi*sigma_space)
        gaussians.append(np.exp(-ks**2/2/sigma_freq**2))
        gaussians[-1] /= np.sum(gaussians[-1])
    return np.prod(np.meshgrid(*gaussians), axis=0)


def sim_fft_image_2d(x0, y0, E0, theta, sigma, stopping_table, shape, image_dxy):
    '''
    shape: TODO: right now this is assumed to be square by how the rotation is performed
    image_dxy: dx/y between adjacent image pixels, in mm
    theta=0 corresponds to particle traveling along x axis
    '''
    gaussian = gaussian_fft(shape, (sigma, sigma))
    #get fft of brag curve in 1d
    track_length = stopping_table.get_stopping_distance(E0)
    dists = np.linspace(0, track_length, 200)
    stopping_powers = stopping_table.get_stopping_power_after_distances(E0, dists)
    stopping_power_fft = fft(stopping_powers)
    stopping_power_dx = dists[1] - dists[0]
    stopping_power_k = 
    #take stopping power to 2d
    kx,ky = np.meshgrid(*(np.arange(N) for N in shape))
    kx[kx/shape[0]>0.5] -= shape[0]
    ky[ky/shape[0]>0.5] -= shape[1]
    kxp = np.cos(theta)*kx - np.sin(theta)*ky
    stopping_power_2d =  np.interp(kxp, stopping_power_ks[stopping_power_ks.argsort()], stopping_powers_fft[stopping_power_ks.argsort()])
    translation = np.exp(-2*np.pi**(0+j)*(kx*x0/shape[0] + ky*y0/shape[1]))
    #return stopping_power_2d*translation*gaussian
    return stopping_power_2d,translation,gaussian

alpha_table = srim_interface.SRIM_Table(data_path='H_in_P10.txt', material_density=1.52)
sim_image_fft = sim_fft_image_2d(10, 10, 0.8, np.radians(45), 5, alpha_table,np.shape(fft_image), 2.2)



'''
idea for algorithm:
1) take fft of event data cube
2) fit model below to fft taken in step 1, using magnitudes of each voxel only (ignoring phase information).
   This will be insensitive to translations, but sensitive to rotations and energy sharing between particles.
3) Fit model including phase information to get information about origin of decay

Maybe background subtract by just getting rid of 0 frequency component in both signals?
'''
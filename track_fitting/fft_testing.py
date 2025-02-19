import numpy as np
import matplotlib.pylab as plt
import scipy.fft as fft
import matplotlib.colors as colors

import sys
#TODO: don't do this!!!
#sys.path.append('/mnt/projects/e21072/OfflineAnalysis/analysis_scripts/alex/gadget_analysis/raw_viewer/')
#import raw_h5_file

import srim_interface



#configure h5 file interface
if True:
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
        sigma_freq = N/(2*np.pi*sigma_space/spacing)
        gaussians.append(np.exp(-freq**2/2/sigma_freq**2))
    return np.prod(np.meshgrid(*gaussians), axis=0)


def sim_fft_image_2d(x0, y0, E0, theta, sigma, stopping_table, shape, image_dxy):
    '''
    shape: TODO: right now this is assumed to be square by how the rotation is performed
    image_dxy: dx/y between adjacent image pixels, in mm
    theta=0 corresponds to particle traveling along x axis
    '''
    gaussian = gaussian_fft(shape, (sigma, sigma), (image_dxy,image_dxy))
    #get fft of brag curve in 1d
    track_length = stopping_table.get_stopping_distance(E0)
    max_dist = np.sqrt(np.prod(shape))*image_dxy
    dists = np.linspace(-max_dist, max_dist, 5000)
    stopping_powers = stopping_table.get_stopping_power_after_distances(E0, dists)
    stopping_power_fft = fft(stopping_powers)
    stopping_power_dx = dists[1] - dists[0]
    stopping_power_ks = fftfreq(len(dists), stopping_power_dx)
    #take stopping power to 2d
    kx,ky = np.meshgrid(*(fftfreq(N, image_dxy) for N in shape))
    kxp = np.cos(theta)*kx - np.sin(theta)*ky
    plt.plot(dists,stopping_powers)
    stopping_power_2d =  np.interp(kxp*image_dxy/stopping_power_dx, stopping_power_ks[stopping_power_ks.argsort()], stopping_power_fft[stopping_power_ks.argsort()])
    translation = np.exp(-2*np.pi*(0+1j)*(kx*x0/shape[0] + ky*y0/shape[1]))
    #return stopping_power_2d*translation*gaussian
    return stopping_power_2d,translation,gaussian

p_table = srim_interface.SRIM_Table(data_path='H_in_P10.txt', material_density=1.64)
grid_spacing = 2.2
xs = np.arange(0,50, grid_spacing)
E0 = 1.1

track_length = p_table.get_stopping_distance(E0)
dists = np.linspace(0, 50, 100000)
#dists = np.arange(0,50, 1)
stopping_powers = p_table.get_stopping_power_after_distances(E0, dists)
stopping_powers_fft = fft.fft(stopping_powers)
k_stopping = fft.fftfreq(len(dists), dists[1]-dists[0])

ks = fft.fftfreq(len(xs), grid_spacing)
fspace = np.interp(ks,k_stopping[k_stopping.argsort()], stopping_powers_fft[k_stopping.argsort()])

#plt.scatter(k_stopping,stopping_powers_fft)
plt.scatter(ks,fspace)

plt.figure()
plt.plot(xs,fft.ifft(fspace))
#plt.plot(dists, stopping_powers)

plt.show()

#sim_image_fft = sim_fft_image_2d(10, 10, 0.8, np.radians(0), 10, alpha_table,(50, 50), 2.2)



'''
idea for algorithm:
1) take fft of event data cube
2) fit model below to fft taken in step 1, using magnitudes of each voxel only (ignoring phase information).
   This will be insensitive to translations, but sensitive to rotations and energy sharing between particles.
3) Fit model including phase information to get information about origin of decay

Maybe background subtract by just getting rid of 0 frequency component in both signals?
'''
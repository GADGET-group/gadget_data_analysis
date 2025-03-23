import numpy as np
import matplotlib.pylab as plt
import scipy.fft as fft
import matplotlib.colors as colors
import scipy.optimize as opt

import sys
#TODO: don't do this!!!
#sys.path.append('/mnt/projects/e21072/OfflineAnalysis/analysis_scripts/alex/gadget_analysis/raw_viewer/')
#import raw_h5_file

#import srim_interface
from track_fitting import build_sim

def gaussian_fft(shape, sigmas):
    '''
    shape: shape of the array to fill
    sigmas: list of sigmas in the spatial domain (x,y,z, etc)
    point_spacings: spacing between spacial points
    '''
    gaussians = [] #1D gaussians
    for N, sigma in zip(shape, sigmas):
        freq = fft.fftfreq(N)
        freq = freq/np.min(freq[freq>0])
        gaussians.append(np.exp(-freq**2/2/sigma**2)/np.sqrt(2*np.pi*sigma**2))
    return np.prod(np.meshgrid(*gaussians), axis=0)

def gaussian_w_cos(shape, sigma, kvec):
    '''
    shape: shape of the array to fill
    sigmas: list of sigmas in the spatial domain (x,y,z, etc)
    point_spacings: spacing between spacial points
    '''
    freqs =[fft.fftfreq(N) for N in shape]
    to_return = np.zeros(shape)
    for i in range(len(freqs[0])):
        for j in range(len(freqs[1])):
            to_return[i,j] = np.exp(-(freqs[0][i]**2 + freqs[1][j]**2)/2/sigma)*np.abs(np.cos(np.dot(kvec, (freqs[0][i], freqs[1][j]))))
    return to_return



raw_file = build_sim.get_rawh5_object('e21072', 124)

#show event in euclidian space
pads, traces = raw_file.get_pad_traces(108, include_veto_pads=False)
data = {pad:np.sum(trace) for pad, trace in zip(pads, traces)}
image = raw_file.get_2d_image(data)

plt.figure()
plt.imshow(image)
plt.colorbar()

#show event in fourier space
plt.figure()
fft_image = fft.fft2(image)
plt.imshow(np.abs(fft.fftshift(fft_image)))
plt.colorbar()


def to_minimize(params):
    A, sigma, k1, k2 = params
    residuals = A*gaussian_w_cos(fft_image.shape, sigma, (k1, k2)) - np.abs(fft_image)
    return np.sum(residuals*residuals)

res = opt.minimize(to_minimize, (1,0.006, 20, 20))

fit_im = res.x[0]*gaussian_w_cos(fft_image.shape, res.x[1], (res.x[2], res.x[3]))
plt.figure()
plt.imshow(fft.fftshift(fit_im))
plt.colorbar()

plt.figure()
plt.imshow(fft.fftshift(fit_im - np.abs(fft_image)))
plt.colorbar()

plt.show(block=False)





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

if False:
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



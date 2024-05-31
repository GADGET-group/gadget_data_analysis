import numpy as np
import matplotlib.pylab as plt
import scipy.integrate as integrate
import scipy.ndimage
import scipy.optimize

input_base_path = 'ruchi_event_107_%s.npy'

dists = np.load(input_base_path%'dist')
es = np.load(input_base_path%'e')

bins = 30
counts_per_MeV = 18600
#make histogram from projected track
fig, axs = plt.subplots(2,1)
hist, bin_edges=np.histogram(dists, bins=bins, weights=es)
bin_width = bin_edges[1] - bin_edges[0]
hist, bin_edges, patches = axs[0].hist(dists, bins=bins, weights=es/counts_per_MeV/bin_width)
axs[0].set_xlabel('position along track (mm)')
axs[0].set_ylabel('energy deposition (MeV/mm)')
fig.tight_layout()
#make bragg curve
dEdx_table = np.load('../p10_alpha_850torr.npy')
#dEdx_table =  np.genfromtxt("../P10_760torr_srim.csv", delimiter=",")

def bragg_w_diffusion(xs, x0, E0, sigma, direction, pressure):
    '''
    returns energy deposition/mm at each of the requested positions
    x values are assumed to be in mm, and be equally spaced
    pressure should be in torr
    '''
    if direction == 'right':
        xs_for_int = np.concatenate([[x0], xs[xs>=x0]])
    else:
        xs_for_int = np.flip(np.concatenate([xs[xs<=x0], [x0]]))
    def dEdx(E):
        to_return = np.interp(E, dEdx_table[:,0], dEdx_table[:,1], left=0)
        if direction == 'right':
            to_return *= -1
        return to_return

    if len(xs_for_int) == 1:
        return np.zeros(len(xs))

    Es = np.squeeze(integrate.odeint(lambda E, x: dEdx(E)*pressure/850, E0, xs_for_int))

    to_return = np.zeros(len(xs))
    if direction == 'right':
        to_return[xs>=x0] = -dEdx(Es[1:])
    else:
        Es = np.flip(Es)
        to_return[xs<=x0] = dEdx(Es[1:])
    
    #convolve with gaussian
    dx = xs[1] - xs[0]
    sigma_in_bins = sigma / dx
    to_return = scipy.ndimage.gaussian_filter1d(to_return, sigma_in_bins, mode='constant', cval=0)
    return to_return

#fit bragg curve to get x0, sigma, and then plot it
bin_centers = (bin_edges[1:] + bin_edges[:-1])/2
to_fit = lambda xs, x0, sigma: bragg_w_diffusion(xs, x0=x0, E0=6.404, sigma=sigma, direction='right', pressure=800)
#theoretical = bragg_w_diffusion(bin_centers, x0=-29.7, E0=6.404, sigma=3.4, direction='right', pressure=800)
popt, pcov = scipy.optimize.curve_fit(to_fit, bin_centers, hist, (-29.7, 3.4))
x0, sigma = popt
theoretical = to_fit(bin_centers, x0, sigma)
print(x0, sigma)
axs[0].plot(bin_centers, theoretical)


#make residuals plot
axs[1].bar(bin_centers, hist - theoretical, width=bin_width)
plt.show()

#typical track is 55 mm (25 pads) long and 6 pads wide, but ~90% energy deposition is contained in 
#a track just 3 pads wide. Let's assume the typical track is at ~45 degrees. Then ~50 pads contribute meaningfully to the energy resolution. 
#Moshe's paper says "2.8% FWHM resolution at 6.288 MeV" for 220Rn, while Ruchi's says 5.4%.
#So we have: 5.4% = sqrt(2.7%^2 + 50(x/75)^2) where x is the unknown uncertainties due to not gain matching.
#x^2=(5.4^2 - 2.7^2)*50=>4.67% from not gain matching and 33% per pad
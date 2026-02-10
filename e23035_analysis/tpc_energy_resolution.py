'''
https://docs.google.com/presentation/d/1VJnyP1jOI0FRYPetZnRHlKmyIutY8To8djjePLyWsSY/edit?slide=id.g3bd75023101_0_224#slide=id.g3bd75023101_0_224

For fft6_res3.pkl
'''
import matplotlib.pylab as plt
import scipy.optimize as opt
import numpy as np

#note: 6288 data point used to create gain match is ommitted
#4 MeV alpha peak is excluded because we can't be sure if it is a single peak
energy = [913, 1063, 1376, 1817, 6050, 8784]
energy_resolution = [19.8, 19.2, 24.1, 26.1,  73.4, 115.4]
uncertainties = [2.2,2.8,0.1,3,4.8,9.8]

other_energies = [4000, 6288]
other_resolutions=[71.1, 73.3]
other_sigma = [13.8, 1.5]

f = lambda x,m,b: m*x + b
popt, pcov = opt.curve_fit(f,energy, energy_resolution, sigma=uncertainties, absolute_sigma=True)

plt.errorbar(energy, energy_resolution, uncertainties, fmt='.', label='data used in fit')
plt.errorbar(other_energies, other_resolutions, other_sigma, fmt='.', label='other datapoints')
plt.plot(energy, f(np.array(energy), *popt))
plt.title('energy resolution = %f * energy + %f keV'%tuple(popt))
plt.legend()
plt.xlabel('energy (keV)')
plt.ylabel('energy resolution (keV)')
plt.show()

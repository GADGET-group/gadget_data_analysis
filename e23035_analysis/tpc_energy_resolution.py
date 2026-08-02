
import matplotlib.pylab as plt
import scipy.optimize as opt
import scipy.stats as stats
import numpy as np

from e23035_analysis import e23035_runs, spectrum_fitter
from raw_viewer import ddas_interface

energies = []
energy_resolutions = []
uncertainties = []

#fit lowest energy proton peak
experiment = 'e23035'
num_workers = 200
proton_binning = (4000//5, 0, 4000)
ddas_runs_low_energy_protons = e23035_runs.get_ddas_60_Ga_runs(good_gamma=False, final_beam_settings=True, good_low_energy_tpc=True, good_long_tracks_tpc=False)
pspec_low_energy = ddas_interface.get_histogram(experiment, ddas_runs_low_energy_protons, proton_binning, "proton_spectrum_low_energy", "proton_spectrum_low_energy", "tpc_energy", "tpc_particle_id==1", num_workers=num_workers)
f_le_protons = spectrum_fitter.spectrum_fitter(pspec_low_energy, 'bg_shift_gaus')
f_le_protons.peaks_to_fit = [([725],500,760)]
f_le_protons.shared_sigma = False
f_le_protons.location_wiggle = 20
f_le_protons.fit_peaks()
energies.append(f_le_protons.get_fit_param_for_peak(0, 'mu')[0])
sigma, sigma_err = f_le_protons.get_fit_param_for_peak(0, 'sigma')
energy_resolutions.append(sigma)
uncertainties.append(sigma_err)


ddas_runs_high_energy_protons = e23035_runs.get_ddas_60_Ga_runs(good_gamma=False, final_beam_settings=True, good_low_energy_tpc=False, good_long_tracks_tpc=True)
pspec_high_energy = ddas_interface.get_histogram(experiment, ddas_runs_high_energy_protons, proton_binning, 'proton_spectrum_high_energy', 'proton_spectrum_high_energy', 'tpc_energy', 'tpc_particle_id==1', num_workers=num_workers)
f_he_protons = spectrum_fitter.spectrum_fitter(pspec_high_energy, 'bg_shift_gaus')
f_he_protons.peaks_to_fit = [([3120],2900,3400)]
f_he_protons.param_bound_functions={'bg_slope':lambda E: (0,0)}
f_he_protons.location_wiggle = 50
f_he_protons.fit_peaks()
energies.append(f_he_protons.get_fit_param_for_peak(0,'mu')[0])
sigma, sigma_err = f_he_protons.get_fit_param_for_peak(0, 'sigma')
energy_resolutions.append(sigma)
uncertainties.append(sigma_err)

ddas_runs_zn_protons = e23035_runs.get_ddas_59_Zn_runs(good_gamma=False, final_beam_settings=True, good_low_energy_tpc=True, good_long_tracks_tpc=True)
pspec_zn_protons = ddas_interface.get_histogram(experiment, ddas_runs_zn_protons, proton_binning, 'proton_spectrum_zn_le_protons', 'proton_spectrum_zn_le_protons', 'tpc_energy', 'tpc_particle_id==1', num_workers=num_workers)
f_zn_protons = spectrum_fitter.spectrum_fitter(pspec_zn_protons, 'bg_shift_gaus')
f_zn_protons.peaks_to_fit = [([814, 913],700,940),  ([1778, 1817],1700,1900)]
f_zn_protons.shared_sigma = False
f_zn_protons.location_wiggle = 20
f_zn_protons.fit_peaks()
for i in range(2):
    energies.append(f_zn_protons.get_fit_param_for_peak(0, 'mu_%d'%i)[0])
    sigma, sigma_err = f_zn_protons.get_fit_param_for_peak(0, 'sigma_%d'%i)
    energy_resolutions.append(sigma)
    uncertainties.append(sigma_err)


f_zn_protons_double_peaks = spectrum_fitter.spectrum_fitter(pspec_zn_protons, 'bg_shift_gaus')
f_zn_protons_double_peaks.peaks_to_fit = [([1330,1376], 1300, 1500),([1778, 1817, 1857],1600,1925)]
f_zn_protons_double_peaks.param_bound_functions = {'bg_slope':lambda E: (0,0)}
f_zn_protons_double_peaks.location_wiggle=30
f_zn_protons_double_peaks.fit_peaks()

#dont include 1700-1857 peaks since the fit doesn't recover whats previously known about these peaks
# energies.append((1778+1817+1857)/3.)
# sigma, sigma_err = f_zn_protons_double_peaks.get_fit_param_for_peak(1, 'sigma')
# energy_resolutions.append(sigma)
# uncertainties.append(sigma_err)
#~1380 peaks
# energies.append((1330+1376)/2.)
# sigma, sigma_err = f_zn_protons_double_peaks.get_fit_param_for_peak(0, 'sigma')
# energy_resolutions.append(sigma)
# uncertainties.append(sigma_err)


alpha_binning = (7000//10, 2000, 9000)
ddas_runs_alphas = e23035_runs.get_ddas_60_Ga_runs(good_gamma=False, final_beam_settings=True, good_low_energy_tpc=False, good_long_tracks_tpc=False)
aspec = ddas_interface.get_histogram(experiment, ddas_runs_alphas, alpha_binning, 'alpha_spectrum', 'alpha spectrum', 'tpc_energy', 'tpc_particle_id==2', num_workers=num_workers)
f_alpha = spectrum_fitter.spectrum_fitter(aspec, 'bg_shift_gaus')
f_alpha.param_bound_functions={'bg_slope':lambda E: (0,0), 'sigma':lambda E: (0, 1000)}
f_alpha.peaks_to_fit = [([6600],6450,7500),([8520], 8000, 9000)]
f_alpha.location_wiggle = 100
f_alpha.fit_peaks()
for i in range(2):
    energies.append(f_alpha.get_fit_param_for_peak(i, 'mu')[0])
    sigma, sigma_err = f_alpha.get_fit_param_for_peak(i, 'sigma')
    energy_resolutions.append(sigma)
    uncertainties.append(sigma_err)

'''
https://docs.google.com/presentation/d/1VJnyP1jOI0FRYPetZnRHlKmyIutY8To8djjePLyWsSY/edit?slide=id.g3bd75023101_0_224#slide=id.g3bd75023101_0_224

For fft6_res3.pkl
'''
#note: 6288 data point used to create gain match is ommitted
#4 MeV alpha peak is excluded because we can't be sure if it is a single peak
energies, energy_resolutions, uncertainties = np.array(energies), np.array(energy_resolutions), np.array(uncertainties)

# energies = np.array([913, 1063, 1376, 1817, 6050, 8784])
# energy_resolutions = np.array([19.8, 19.2, 24.1, 26.1,  73.4, 115.4])
# uncertainties = np.array([2.2,2.8,0.1,3,4.8,9.8])

other_energies = [4000, 6288]
other_resolutions=[71.1, 73.3]
other_sigma = [13.8, 1.5]

f_lin = lambda x, m, b: m*x + b
f_sqrt = lambda x, m, b: np.sqrt(m*x + b)

popt_lin, _ = opt.curve_fit(f_lin, energies, energy_resolutions, sigma=uncertainties, absolute_sigma=True)
popt_sqrt, _ = opt.curve_fit(f_sqrt, energies, energy_resolutions, sigma=uncertainties, absolute_sigma=True)
popt_sqrt_posb, _ = opt.curve_fit(f_sqrt, energies, energy_resolutions, sigma=uncertainties, absolute_sigma=True, bounds=([-np.inf,0],[np.inf,np.inf]))

plt.errorbar(energies, energy_resolutions, uncertainties, fmt='.', label='data used in fit')
plt.errorbar(other_energies, other_resolutions, other_sigma, fmt='.', label='other datapoints')
xs = np.linspace(min(energies), max(energies), num=100)

plt.plot(xs, f_lin(xs, *popt_lin), label=f'Linear: {popt_lin[0]:.4f}*E + {popt_lin[1]:.4f}')
plt.plot(xs, f_sqrt(xs, *popt_sqrt), label=f'Sqrt: sqrt({popt_sqrt[0]:.4f}*E + {popt_sqrt[1]:.4f})')
plt.plot(xs, f_sqrt(xs, *popt_sqrt_posb), label=f'Sqrt: sqrt({popt_sqrt_posb[0]:.4f}*E + {popt_sqrt_posb[1]:.4f})')
plt.title('Energy Resolution Fits')
plt.legend()
plt.xlabel('energy (keV)')
plt.ylabel('energy resolution (keV)')

ndf = len(energies) - 2

chi2_lin = np.sum((f_lin(energies, *popt_lin) - energy_resolutions)**2/uncertainties**2)
print('--- Linear Fit ---')
print('X^2 = ', chi2_lin)
print('Ndf = ', ndf)
print('p-value = ', 1-stats.chi2.cdf(chi2_lin, ndf))

chi2_sqrt = np.sum((f_sqrt(energies, *popt_sqrt) - energy_resolutions)**2/uncertainties**2)
print('\n--- Sqrt Fit ---')
print('X^2 = ', chi2_sqrt)
print('Ndf = ', ndf)
print('p-value = ', 1-stats.chi2.cdf(chi2_sqrt, ndf))

plt.show(block=False)

#try using bounds to see if we recover known Zn peaks
sigma_func = lambda E: (f_sqrt(E,*popt_sqrt), f_sqrt(E,*popt_sqrt))
f_zn_test = spectrum_fitter.spectrum_fitter(pspec_zn_protons, 'bg_shift_gaus')
f_zn_test.peaks_to_fit = [([821,913], 600,980), ([1061, 1116, 1179,1263,1369,1465], 1040, 1500),
             ([1061, 1116, 1179,1263,1369,1465,1778,1817,1857, 2025, 2089, 2182, 2197, 2250, 2390, 2435], 1040, 2500)]
f_zn_test.parameterizations = {
    'sigma': {
        'formula': '[sigma_c] + [sigma_m]*({mu})',
        #'formula': 'sqrt([sigma_c] + [sigma_m]*({mu}))',
        'params': ['sigma_c', 'sigma_m'],
        'guesses': [0., 0.01],
        #'bounds': [(-1000, 1000), (0, 10)]
        'bounds': [(-100, 100), (0.0001, 0.1)]
    }
}
f_zn_test.param_bound_functions = {'bg_slope':lambda E: (0,0) if E>1000 else (-1,1)}
f_zn_test.shared_sigma = False
f_zn_test.location_wiggle=10
f_zn_test.fit_peaks()
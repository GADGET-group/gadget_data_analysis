from tqdm import tqdm
import numpy as np
import matplotlib.pylab as plt

from raw_viewer.field_dist.polynomial_correction import PolynomialCorrection
from raw_viewer import ddas_interface
from raw_viewer import process_runs

experiment = 'e23035'
get_runs = range(145, 150+1)
ddas_runs = range(126, 131+1)

poly_correction = PolynomialCorrection()



print('loading GET data')
uncorrected_lengths = process_runs.get_lengths(experiment, get_runs)
uncorrected_widths = process_runs.get_quantity('charge_width', experiment, get_runs)
uncorrected_angles = process_runs.get_angle(experiment, get_runs)
print('loading times from DDAS')
times_since_beam_off = []
for ddas_run in tqdm(ddas_runs):
    #exclude last event since it seems to be dumped
    times_since_beam_off.append(ddas_interface.get_time_since_beam_off(experiment, ddas_run)[:-1])
times_since_beam_off = np.concatenate(times_since_beam_off)


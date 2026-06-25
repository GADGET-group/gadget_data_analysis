import numpy as np
import ROOT

from raw_viewer import ddas_interface

experiment = 'e23035'
ddas_run = 262
num_workers = 10

tof_axis_length = np.abs(-6.2252e-7 + 6.18653e-7)
de_axis_length = 6546-6185

de_dict = {'60Ga':}

pid_hist = ddas_interface.get_histogram(experiment, ddas_run, (1000,-0.63e-6,-0.6e-6,1000,4000,8000), 'pid', 'pid', 
                'msx100_e:(cross_scint_b2_t - db_5_scint_t)', 
                selection='cross_scint_b2_m==1 && db_5_scint_m==1 &&msx100_m==1', num_workers=num_workers)
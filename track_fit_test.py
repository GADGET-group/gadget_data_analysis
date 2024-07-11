from track_fitting import TraceFunction
from raw_viewer import raw_h5_file

run_h5_path = '/mnt/analysis/e21072/gastest_h5_files/run_0368.h5'
event_num = 5

init_position_guess = (-7, 10, 100)
E_gues = 6.4

adc_scale = 376646/6.288 #counts/MeV, from fitting events with range 40-43 in run 0368 with p10_default

#use theoretical zscale
clock_freq = 50e6 #Hz
drift_speed = 54.4*1e6 #mm/s, from ruchi's paper
zscale = drift_speed/clock_freq


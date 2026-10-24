import numpy as np

from raw_viewer import process_runs

def get_cobos_present(experiment, run, num_events_to_check = 100):
    num_asads_expected = 4 #terminate once 4 asads have been seen
    h5 = process_runs.get_h5_file(experiment, run)
    h5.background_subtract_mode = 'none' #disable background subtrction since only care about channels
    cobos = []
    evt_bounds = h5.get_event_num_bounds()
    for i in range(evt_bounds[0], min(evt_bounds[0] + num_events_to_check, evt_bounds[1])):
        l = np.unique(h5.get_data(i)[:,0])
        for j in l:
            if j not in cobos:
                cobos.append(j)
        if len(cobos) >= num_asads_expected:
            return np.sort(cobos)
    return np.sort(cobos)
'''
 for i in range(1, 229):
     try:
         cobos = get_cobos_present('e25058', i)
         if len(cobos) != 4:
             print('run ', i, ' only has cobos ', cobos)
     except Exception as e:
         print(e)

'''
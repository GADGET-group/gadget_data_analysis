import os
import pathlib

import numpy as np
import matplotlib.pylab as plt

from raw_viewer import process_runs

def save_images(experiment, run):
    num_to_save = 10
    h5file = process_runs.get_h5_file(experiment, run)
    #save x-y projections for first, last, and random num_to_save images
    first, last = h5file.get_event_num_bounds()
    first_events = np.arange(first, first+num_to_save)
    last_events = np.arange(last-num_to_save+1, last+1)
    random_events = np.random.randint(first, last, num_to_save)
    events_to_save = np.concat([first_events, random_events, last_events])
    
    im_save_path = process_runs.get_save_path(experiment).replace('proc_pkl', 'quick_look')
    im_save_path = os.path.join(im_save_path, 'run%d'%run)
    pathlib.Path.mkdir(im_save_path)
    for event_num in events_to_save:
        h5file.show_2d_projection(event_num, block=False)
        plt.savefig(os.path.join(im_save_path, 'event%d_xy.png'%event_num))
        plt.close()

        #set baseline subtraction to fixed window before saving traces
        old_baseline_mode = h5file.background_subtract_mode
        old_background_bounds = h5file.num_background_bins
        h5file.background_subtract_mode = 'fixed window'
        h5file.num_background_bins = (10,20)
        h5file.plot_traces(event_num, block=False)
        plt.savefig(os.path.join(im_save_path, 'event%d_traces.png'%event_num))
        plt.close()
        h5file.background_subtract_mode = old_baseline_mode
        h5file.num_background_bins = old_background_bounds

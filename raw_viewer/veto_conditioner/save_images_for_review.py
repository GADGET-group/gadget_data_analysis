import os

import matplotlib.pyplot as plt

from raw_viewer import process_runs


def save_images(experiment, runs, events_per_run=100):
    for r in runs:
        h5file = process_runs.get_h5_file(experiment, r)
        print('saving images from run '+str(r))
        fig_name = 'xyproj'
        bounds = h5file.get_event_num_bounds()
        plt.figure(fig_name)
        veto_counts = process_runs.get_max_veto_counts(experiment, [r])
        ring_counts = process_runs.get_outer_ring_max_counts(experiment, [r])
        save_path = os.path.join(os.path.split(__file__)[0], f'{experiment}', f'run_{r}')
        os.makedirs(save_path, exist_ok=True)
        for event_num in range(bounds[0], min(bounds[0]+events_per_run, bounds[1])):
            h5file.show_2d_projection(event_number=event_num, block=False, fig_name=fig_name)
            plt.title('event %d, veto %f, ring %f'%(event_num, veto_counts[event_num-bounds[0]], ring_counts[event_num-bounds[0]]))
            plt.savefig(os.path.join(save_path, f'event{event_num}.png'))
            plt.clf()

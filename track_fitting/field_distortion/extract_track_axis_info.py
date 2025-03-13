import os
import pickle

from tqdm import tqdm
import numpy as np

from track_fitting import build_sim 

def get_track_info(experiment, run_number):
    '''
    Get information about track direction, width, and charge per pad, which isn't normally stored when processing runs.
    Only redoes processing if a pickled version of this information isn't available.
    '''
    package_directory = os.path.dirname(os.path.abspath(__file__))
    fname = os.path.join(package_directory, '%s_run%d.pkl'%(experiment, run_number))
    if os.path.exists(fname):
        print('run previously processed, loading previous results')
        with open(fname, 'rb') as file:
            return pickle.load(file)
    else:
        h5file = build_sim.get_rawh5_object(experiment, run_number)
        print('calculating quantities not saved by GUI process run')
        first_event, last_event = h5file.get_event_num_bounds()
        track_centers, principle_axes,variances_along_axes, pad_charges, track_endpoints = [],[],[],[],[]
        for evt in tqdm(range(first_event, last_event + 1)):
            center, uu,dd,vv = h5file.get_track_axis(evt, return_all_svd_results=True, threshold=h5file.length_counts_threshold)
            xs, ys, zs, es = h5file.get_xyze(evt, threshold=h5file.length_counts_threshold)
            principle_axes.append(vv)
            variances_along_axes.append(dd**2/(len(xs)-1))
            track_centers.append(center)
            pad_counts = np.zeros(1024)
            for pad, trace in zip(*h5file.get_pad_traces(evt)):
                pad_counts[pad] = np.sum(trace)
            pad_charges.append(pad_counts)
            #get track end points
            if len(variances_along_axes[-1])==3:
                points = np.concatenate((xs[:, np.newaxis], 
                        ys[:, np.newaxis], 
                        zs[:, np.newaxis]), 
                        axis=1)
                rbar = points - center
                track_direction = vv[0]/np.sqrt(np.sum(vv[0]*vv[0]))
                rdotv = np.dot(rbar, track_direction)
                first_point = points[np.argmin(rdotv)]
                last_point = points[np.argmax(rdotv)]
                track_endpoints.append([first_point, last_point])
            else:
                track_endpoints.append([(0,0,0), (0,0,0)])
        track_centers = np.array(track_centers)
        pad_charges = np.array(pad_charges)
        to_return={'track_center':track_centers, 'principle_axes':principle_axes, 'variance_along_axes': variances_along_axes,
                   'pad_charge': pad_charges, 'endpoints':track_endpoints}
        print('pickling')
        with open(fname, 'wb') as file:
            pickle.dump(to_return, file)
        return to_return
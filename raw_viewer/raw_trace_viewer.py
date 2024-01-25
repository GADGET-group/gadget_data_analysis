'''
File containing functions for visualizing data from the GADGET II TPC.
The raw_h4_file module contains functions for extracting and manipulating data from the h5
files generated when merging GRAW files.
'''

import os
import re

import matplotlib.pylab as plt
from matplotlib.colors import LinearSegmentedColormap
import h5py
import numpy as np
from raw_h5_file import VETO_PADS







def get_counts_array(file):
    '''
    returns an array, where each element is the total number of counts in an image
    '''
    to_return = []
    get = file['get']
    for entry in get:
        if '_data' in entry: #these are data entries. The others are meta data
            to_return.append(np.sum(file['get'][entry][:,5:]))
    return to_return




def show_2d_projection(file, event_number, block=True):
    cdict={'red':  ((0.0, 0.0, 0.0),
                    (0.25, 0.0, 0.0),
                    (0.5, 0.8, 1.0),
                    (0.75, 1.0, 1.0),
                    (1.0, 0.4, 1.0)),

            'green': ((0.0, 0.0, 0.0),
                    (0.25, 0.0, 0.0),
                    (0.5, 0.9, 0.9),
                    (0.75, 0.0, 0.0),
                    (1.0, 0.0, 0.0)),

            'blue':  ((0.0, 0.0, 0.4),
                    (0.25, 1.0, 1.0),
                    (0.5, 1.0, 0.8),
                    (0.75, 0.0, 0.0),
                    (1.0, 0.0, 0.0))
            }
    cmap = LinearSegmentedColormap('test',cdict)

    event = file['get']['evt%d_data'%event_number]
    dirname = os.path.dirname(__file__)
    padxy = np.loadtxt(os.path.join(dirname, 'padxy.txt'), delimiter=',')
    image = np.ones((72,72))*-1
    for pad_data in event:
        time_data = pad_data[5:]
        pad = pad_data[4]
        if pad < len(padxy):
            x, y = padxy[pad]
            x = int(x/1.1+36)
            y = int(y/1.1+36)
            image[x][y]=np.sum(time_data)
    image = np.ma.array(image, mask=(image==-1))#don't mess up the color map with pixels that didn't fire
    fig = plt.figure()
    plt.imshow(image, cmap=cmap)
    cbar = plt.colorbar()
    plt.title('event %d, total counts=%d'%(event_number, get_counts_in_event(file, event_number)))
    plt.show(block=block)


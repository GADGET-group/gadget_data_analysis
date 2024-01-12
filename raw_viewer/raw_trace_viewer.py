import os
import re

import matplotlib.pylab as plt
from matplotlib.colors import LinearSegmentedColormap
import h5py
import numpy as np


def load_run(run_number):
    h5_dir = '/mnt/analysis/e21072/h5test/'
    file_path = h5_dir + 'run_' + ('%4d'%run_number).replace(' ', '0') + '.h5'
    return h5py.File(file_path, 'r')

def plot_traces(file, event_number, block=True):
    event = file['get']['evt%d_data'%event_number]
    '''
    For each event, the the first 5 columns are CoBo, AsAd, AGET, channel and pad number.
    The following 512 columns are the time buckets.
    '''
    plt.figure()
    for data in event:
        pad = data[4]
        plt.plot(data[5:], label='%d'%pad)
    plt.legend()
    plt.show(block=block)

def get_first_good_event_number(file):
    return int(re.search('\d+', list(file['get'].keys())[0]).group(0))

def get_pads_fired(file, event_number):
    event = file['get']['evt%d_data'%event_number]
    dirname = os.path.dirname(__file__)
    padxy = np.loadtxt(os.path.join(dirname, 'padxy.txt'), delimiter=',')
    to_return = 0
    for pad_data in event:
        pad = pad_data[4]
        if pad < len(padxy):
            to_return += 1
    return to_return

def plot_3d_traces(file, event_number, threshold=0, block=True):
    event = file['get']['evt%d_data'%event_number]
    dirname = os.path.dirname(__file__)
    padxy = np.loadtxt(os.path.join(dirname, 'padxy.txt'), delimiter=',')
    xs, ys, zs, es = [], [], [], []
    for pad_data in event:
        time_data = pad_data[5:]
        pad = pad_data[4]
        if pad < len(padxy):
            x, y = padxy[pad]
            zscale = 1.45#from GadgetRunH5
            zs.append(np.arange(len(time_data))*zscale)
            xs.append(np.ones(len(time_data))*x)
            ys.append(np.ones(len(time_data))*y)
            es.append(time_data)
    es = np.array(es).flatten()
    xs = np.array(xs).flatten()[es>threshold]
    ys = np.array(ys).flatten()[es>threshold]
    zs = np.array(zs).flatten()[es>threshold]
    zs -= np.min(zs)    
    es = es[es>threshold]

    fig = plt.figure(figsize=(6,6))
    ax = plt.axes(projection='3d')
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_xlim3d(-35, 35)
    ax.set_ylim3d(-35, 35)
    ax.set_zlim3d(0, 400)

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
    cdict['alpha'] = ((0.0, 0.0, 0.0),
                      (0.3,0.2, 0.2),
                      (0.8,1.0, 1.0),
                     (1.0, 1.0, 1.0))
    cmap = LinearSegmentedColormap('test',cdict)
    ax.view_init(elev=90, azim=0)
    ax.scatter(xs, ys, zs, c=es, cmap=cmap)
    cbar = fig.colorbar(ax.get_children()[0])
    plt.show(block=block)

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
    plt.show(block=block)

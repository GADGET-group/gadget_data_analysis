import os

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
            zscale = 1.#400./512 #TODO: correct this
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
                    # (0.3,0.05, 0.05),
                    # (0.8,1.0, 1.0),
                    (1.0, 1.0, 1.0))
    cmap = LinearSegmentedColormap('test',cdict)

    ax.scatter(xs, ys, zs, c=es, cmap=cmap)
    cbar = fig.colorbar(ax.get_children()[0])
    plt.show(block=block)

if __name__ == '__main__':
    eventid=2007
    print(eventid)
    plot_3d_traces(load_run(316), eventid, threshold=500)
    #plot_traces(load_run(316), 235)
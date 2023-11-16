import matplotlib.pylab as plt
import h5py
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

def load_run(run_number):
    h5_dir = '/mnt/analysis/e21072/h5test/'
    file_path = h5_dir + 'run_0%d.h5'%run_number
    return h5py.File(file_path, 'r')

def plot_traces(file, event_number):
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
    plt.show()

def plot_3d_traces(file, event_number, threshold=0):
    event = file['get']['evt%d_data'%event_number]
    padxy = np.loadtxt('padxy.txt', delimiter=',')
    xs, ys, zs, es = [], [], [], []
    for pad_data in event:
        time_data = pad_data[5:]
        pad = pad_data[4]
        if pad < len(padxy):
            x, y = padxy[pad]
            zscale = 400./512 #TODO: correct this
            i = 0
            while i < len(time_data):
                if time_data[i]>threshold:
                    xs.append(x)
                    ys.append(y)
                    zs.append(i*zscale-40)
                    es.append(time_data[i])
                i += 1
    fig = plt.figure(figsize=(6,6))
    ax = plt.axes(projection='3d')
    ax.set_xlim3d(-35, 35)
    ax.set_ylim3d(-35, 35)
    ax.set_zlim3d(0, 35)

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

    ax.scatter(xs, ys, zs, c=es, cmap=cmap, depthshade=0)
    cbar = fig.colorbar(ax.get_children()[0])
    plt.show(block=True)

if __name__ == '__main__':
    eventid=57968
    print(eventid)
    plot_3d_traces(load_run(316), eventid, threshold=1000)
    #plot_traces(load_run(316), 235)
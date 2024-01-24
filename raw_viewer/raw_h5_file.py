import os

import numpy as np
import h5py
import matplotlib.pylab as plt
from matplotlib.colors import LinearSegmentedColormap


class raw_h5_file:
    '''
    Things this class should be able to do:
        Return x,y,z,e arrays
        Perform background subtraction based on 
        Return

    '''
    def __init__(self, file_path, zscale = 400./512, flat_lookup_csv=None):
        self.h5_file = h5py.File(file_path, 'r')
        self.padxy = np.loadtxt('padxy.txt', delimiter=',')
        

        if flat_lookup_csv == None: #figure out COBO configuration from meta data
            if  len(self.h5_file['meta']) == 9: #2 COBO configuration
                self.flat_lookup = np.loadtxt('flatlookup2cobos.csv', delimiter=',', dtype=int)
            else:
                assert False
        else:
            self.flat_lookup = np.loadtxt(flat_lookup_csv, delimiter=',', dtype=int)
        
        pad_plane = np.genfromtxt('PadPlane.csv',delimiter=',', filling_values=-1) #used for mapping pad numbers to a 2D grid
        self.pad_to_xy_index = {} #maps tuples of (asad, aget, channel) to (x_index,y_index)
        for y in range(len(pad_plane)):
            for x in range(len(pad_plane[0])):
                pad = pad_plane[x,y]
                if pad != -1:
                    self.pad_to_xy_index[int(pad)] = (x,y)

        self.chnls_to_pad = {} #maps tuples of (asad, aget, channel) to pad number
        self.chnls_to_xy_coord = {} #maps tuples of (asad, aget, channel) to (x,y) coordinates in mm
        self.chnls_to_xy_index = {}
        for line in self.flat_lookup:
            chnls = tuple(line[0:4])
            pad = line[4]
            self.chnls_to_pad[chnls] = pad
            self.chnls_to_xy_coord[chnls] = self.padxy[pad]
            self.chnls_to_xy_index[chnls] = self.pad_to_xy_index[pad]
        
        self.zscale = zscale #conversion factor from time bin to mm


    def get_xyte(self, event_number):
        '''
        Returns: xs, ys, ts, es
                 Where each of these is an array s.t. each "pixel" in the in the raw TPC data is represented.
                 eg, (xs[i], ys[i], ts[i]) gives the position of a pad and time bin number,
                 and es[i] gives the charge that arrived at that pad at the given time.
        '''
        xs, ys, es = [], [], []
        event_data =  self.h5_file['get']['evt%d_data'%event_number]
        #after this look, xs=[x1, x2, ...], same for ys, es=[[1st pad data], [2nd pad data], ...]
        for pad_data in event_data:
            chnl_info = tuple(pad_data[0:4])
            x,y = self.chnls_to_xy_coord[chnl_info]
            xs.append(x)
            ys.append(y)
            es.append(pad_data[5:])
        #reshape as needed to get to final format for x,y,e
        xs = np.repeat(xs, 512)
        ys = np.repeat(ys, 512)
        es = np.array(es).flatten()
        #make time bins data
        ts = np.tile(np.arange(0, 512), int(len(xs)/512))
        return xs, ys, ts, es
    
    def get_xyze(self, event_number):
        '''
        Same as xyte, but scales time bins to get z coordinate
        '''
        x,y,t,e = self.get_xyte(event_number)
        return x,y, t*self.zscale ,e
    
    
def plot_3d_traces(xs, ys, zs, es, threshold=0, block=True):

    fig = plt.figure(figsize=(6,6))
    ax = plt.axes(projection='3d')
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_xlim3d(-200, 200)
    ax.set_ylim3d(-200, 200)
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
    ax.view_init(elev=45, azim=45)
    ax.scatter(xs, ys, zs, c=es, cmap=cmap)
    cbar = fig.colorbar(ax.get_children()[0])
    #plt.title('event %d, total counts=%d'%(event_number, get_counts_in_event(file, event_number)))
    plt.show(block=block)

        



'''
import h5py
file = h5py.File('/mnt/analysis/e21072/gastest_h5_files/run_0032.h5', 'r')
a=file['get']['evt3057_data']

import raw_h5_file
file = raw_h5_file.raw_h5_file('/mnt/analysis/e21072/gastest_h5_files/run_0032.h5')
raw_h5_file.plot_3d_traces(*file.get_xyze(3057))
'''



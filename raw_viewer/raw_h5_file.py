import os

import numpy as np
import h5py
import matplotlib.pylab as plt
from matplotlib.colors import LinearSegmentedColormap

'''
Notes for tomorrow:
--make main_gui.py work with recent updates
--background subtraction, outlier removal, PCA, RvE plot
--are some pads more noisy than other? Are they all on one cobo or ASAD board or AGET chip?
'''

VETO_PADS = (253, 254, 508, 509, 763, 764, 1018, 1019)

class raw_h5_file:
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
        self.pad_to_xy_index = {} #maps pad number to (x_index,y_index)
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
    
    def get_counts_in_event(self, event_number):
        event = self.h5_file['get']['evt%d_data'%event_number]
        return np.sum(event[:,5:])
    
    def get_event_num_bounds(self):
        #returns first event number, last event number
        return int(self.h5_file['meta']['meta'][0]), int(self.h5_file['meta']['meta'][2])
    
    def get_pad_traces(self, event_number):
        '''
        returns [pads which fired], [[time series data for first pad], [time series data for 2nd pad], ...]
        Pad numbers are determined from AGET, COBO, and channel number, rather than the pad number written during
        the merging process.
        '''
        pads, pad_datas = [], []
        event_data =  self.h5_file['get']['evt%d_data'%event_number]
        for pad_data in event_data:
            chnl_info = tuple(pad_data[0:4])
            pads.append(self.chnls_to_pad[chnl_info])
            pad_datas.append(pad_data[5:])
        return pads, pad_datas
    
    def get_num_pads_fired(self, event_number):
        event = self.h5_file['get']['evt%d_data'%event_number]
        return len(event)
    




'''
import h5py
file = h5py.File('/mnt/analysis/e21072/gastest_h5_files/run_0032.h5', 'r')
a=file['get']['evt3057_data']

import raw_h5_file
file = raw_h5_file.raw_h5_file('/mnt/analysis/e21072/gastest_h5_files/run_0032.h5')
raw_h5_file.plot_3d_traces(*file.get_xyze(3057))
'''



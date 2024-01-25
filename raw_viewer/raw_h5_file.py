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
        
        self.pad_plane = np.genfromtxt('PadPlane.csv',delimiter=',', filling_values=-1) #used for mapping pad numbers to a 2D grid
        self.pad_to_xy_index = {} #maps pad number to (x_index,y_index)
        for y in range(len(self.pad_plane)):
            for x in range(len(self.pad_plane[0])):
                pad = self.pad_plane[x,y]
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

        self.pad_backgrounds = None #initialize with determine_pad_backgrounds

        #color map for plotting
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
        # cdict['alpha'] = ((0.0, 0.0, 0.0),
        #                 (0.3,0.2, 0.2),
        #                 (0.8,1.0, 1.0),
        #                 (1.0, 1.0, 1.0))
        self.cmap = LinearSegmentedColormap('test',cdict)


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
    
    def determine_pad_backgrounds(self, num_background_bins=200):
        '''
        Assume the first num_background_bins of each pad's data only include background.
        Determine average value of this pad and stddev across all events in which the pad fired.
        Store this information in a dictionairy member variable.

        Pad background will be stored in self.pad_backgrounds, which is a dictionairy indexed by pad
        number which stores (background average, background standard deviation) pairs.
        '''
        first, last = self.get_event_num_bounds()

        #compute average
        running_averages = {}
        for event_num in range(first, last+1):
            event_data = self.h5_file['get']['evt%d_data'%event_num]
            for line in event_data:
                chnl_info = tuple(line[0:4])
                pad = self.chnls_to_pad[chnl_info]
                if pad not in running_averages:
                    running_averages[pad] = (0,0) #running everage, events processed
                ave_this = np.average(line[5:num_background_bins+5])
                ave_last, n = running_averages[pad]
                running_averages[pad] = ((n*ave_last + ave_this)/(n+1), n+1)
        #compute standard deviation
        running_stddev = {}
        for event_num in range(first, last+1):
            event_data = self.h5_file['get']['evt%d_data'%event_num]
            for line in event_data:
                chnl_info = tuple(line[0:4])
                pad = self.chnls_to_pad[chnl_info]
                if pad not in running_stddev:
                    running_stddev[pad] = (0,0)
                std_this = np.std(line[5:num_background_bins+5])
                std_last, n = running_stddev[pad]
                running_stddev[pad] = ((n*std_last + std_this)/(n+1), n+1)
        self.pad_backgrounds = {}
        for pad in running_averages:
            self.pad_backgrounds[pad] = (running_averages[pad][0], running_stddev[pad][0])

    def show_pad_backgrounds(self, fig_name=None, block=True):
        ave_image = np.zeros(np.shape(self.pad_plane))
        std_image = np.zeros(np.shape(self.pad_plane))
        for pad in self.pad_backgrounds:
            x,y = self.pad_to_xy_index[pad]
            ave, std = self.pad_backgrounds[pad]
            ave_image[x,y] = ave
            std_image[x,y] = std

        fig=plt.figure(fig_name)
        plt.clf()
        ave_ax = plt.subplot(1,2,1)
        ave_ax.set_title('average counts')
        ave_shown = ave_ax.imshow(ave_image, cmap=self.cmap)
        fig.colorbar(ave_shown, ax=ave_ax)

        std_ax = plt.subplot(1,2,2)
        std_ax.set_title('standard deviation')
        std_shown=std_ax.imshow(std_image, cmap=self.cmap)
        fig.colorbar(std_shown, ax=std_ax)
        #plt.colorbar(ax=std_plot)
        #plt.colorbar())
        plt.show(block=block)

    def plot_traces(self, event_num, block=True, fig_name=None):
        '''
        Note: veto pads are plotted as dotted lines
        '''
        plt.figure(fig_name)
        plt.clf()
        pads, pad_data = self.get_pad_traces(event_num)
        for pad, data in zip(pads, pad_data):
            pad = data[4]
            r = pad/1024
            g = (pad%512)/512
            b = (pad%256)/256
            if pad in VETO_PADS:
                plt.plot(data[5:], '--', color=(r,g,b), label='%d'%pad)
            else:
                plt.plot(data[5:], color=(r,g,b), label='%d'%pad)
        plt.legend()
        plt.show(block=block)

    def plot_3d_traces(self, event_num, threshold=0, block=True, fig_name=None):
        fig = plt.figure(fig_name, figsize=(6,6))
        plt.clf()
        ax = plt.axes(projection='3d')
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        ax.set_xlim3d(-200, 200)
        ax.set_ylim3d(-200, 200)
        ax.set_zlim3d(0, 400)

        xs, ys, zs, es = self.get_xyze(event_num)
        xs = xs[es>threshold]
        ys = ys[es>threshold]
        zs = zs[es>threshold]
        es = es[es>threshold]

        ax.view_init(elev=45, azim=45)
        ax.scatter(xs, ys, zs, c=es, cmap=self.cmap)
        cbar = fig.colorbar(ax.get_children()[0])
        #TODO
        #plt.title('event %d, total counts=%d'%(event_num, get_counts_in_event(file, event_number)))
        plt.show(block=block)
    

        


'''
import h5py
file = h5py.File('/mnt/analysis/e21072/gastest_h5_files/run_0032.h5', 'r')
a=file['get']['evt3057_data']

import raw_h5_file
file = raw_h5_file.raw_h5_file('/mnt/analysis/e21072/gastest_h5_files/run_0032.h5')
raw_h5_file.plot_3d_traces(*file.get_xyze(3057))
'''



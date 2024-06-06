import os

import numpy as np
import scipy.spatial
import scipy.spatial.distance
import h5py
import matplotlib.pylab as plt
import matplotlib.colors as colors
from matplotlib.colors import LinearSegmentedColormap
import tqdm

import skimage.measure

'''
Notes for tomorrow:
--make main_gui.py work with recent updates
--background subtraction, outlier removal, PCA, RvE plot
--are some pads more noisy than other? Are they all on one cobo or ASAD board or AGET chip?
'''

VETO_PADS = (253, 254, 508, 509, 763, 764, 1018, 1019)
FIRST_DATA_BIN = 6 #first time bin is dumped, because it is junk
NUM_TIME_BINS = 512+5-FIRST_DATA_BIN

class raw_h5_file:
    def __init__(self, file_path, zscale = 400./512, flat_lookup_csv=None):
        self.file_path = file_path
        self.h5_file = h5py.File(file_path, 'r')
        self.padxy = np.loadtxt(os.path.join(os.path.dirname(__file__), 'padxy.txt'), delimiter=',')
        
        self.flat_lookup_file_path = flat_lookup_csv        
        self.flat_lookup = np.loadtxt(self.flat_lookup_file_path, delimiter=',', dtype=int)
        
        self.pad_plane = np.genfromtxt(os.path.join(os.path.dirname(__file__),'PadPlane.csv'),delimiter=',', filling_values=-1) #used for mapping pad numbers to a 2D grid
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
        #round xy to nearest 10nths place to avoid issues with different floating point formats
        self.xy_to_pad = {tuple(np.round(self.padxy[pad], 1)):pad for pad in range(len(self.padxy))}
        self.xy_to_chnls = {tuple(np.round(self.chnls_to_xy_coord[chnls], 1)):chnls 
                            for chnls in self.chnls_to_xy_coord}
        
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

        self.background_subtract_mode = 'none' #none, fixed window, or convolution
        self.background_convolution_kernel = None#bin backgrounds are determined by convolving the trace with this array
        self.remove_outliers = False
        self.num_background_bins = (0,0) #number of time bins to use for per event background subtraction
        self.range_bounds = (0, 100)
        self.ic_bounds = (0, 1e6)
        self.angle_bounds = None #min/max angles in radians to use when calculating veto conditions
        self.length_counts_threshold = 300 #threshold to use when calculating range
        self.ic_counts_threshold = 100 #threshold to use when calculating energy
        self.veto_threshold = 100 #veto events for which any veto pad exceeds this value at some point
        #allowed modes = 'all data', 'peak only', or 'near peak'
        #all data mode will make use of the entire trace from each pad
        #'peak only' will make use of only the max value the trace reached on each pad
        #'near peak' will only use data within some window of the peak of each trace
        self.mode = 'all data' 
        self.near_peak_window_width = 100 #+/- time bins to include
        self.require_peak_within = (-np.inf, np.inf)#currentlt implemented for near peak mode only. Zero entire trace if peak is not within this window
        self.include_counts_on_veto_pads = False

        #cobo and asad selction. Can be "all" or a list of ints
        self.asads='all'
        self.cobos='all'
        self.pads='all'

    def get_pad_from_xy(self, xy):
        '''
        xy: tuple of (x,y) to lookup pad number for
        '''
        xy = tuple(np.round(xy, 1))
        return self.xy_to_pad[xy]
    
    def get_chnl_from_xy(self, xy):
        '''
        xy: tuple of (x,y) to lookup pad number for
        '''
        xy = tuple(np.round(xy, 1))
        return self.xy_to_chnls[xy]

    def get_data(self, event_number):
        '''
        Get data for event, with background subtraction and pad outlier removal applied as specified
        by member variables.

        Outlier removal:
        1. Populate a pad plane image with all zeros, except for pads which fired
        2. Use skimage.measure.label to determine to label pads based on connectivity
        3. Make a new datacube which just has the pads from the largest blob

        Veto pads should NOT be removed during outlier removal.
        Does NOT apply thresholding. However, this is applied in get_xyte and and get_xyze.
        '''
        data = self.h5_file['get']['evt%d_data'%event_number]

        
        if self.asads != 'all' or self.cobos != 'all' or self.pads != 'all':
            to_copy = []
            for i, line in enumerate(data):
                chnl_info =  tuple(line[0:4])
                cobo, asad, channel, *rest = chnl_info
                pad = self.chnls_to_pad[chnl_info]
                if (self.cobos == 'all' or cobo in self.cobos) and \
                        (self.asads == 'all' or asad in self.asads) and \
                        (self.pads == 'all' or pad in self.pads):
                    to_copy.append(i)
            data = np.array(data[to_copy], dtype=float)

        else:
            data = np.array(data, copy=True, dtype=float)

        if self.remove_outliers:
            pad_image = np.zeros(np.shape(self.pad_plane))


        #Loop over each pad, performing background subtraction and marking the pad in the pad image
        #which will be used for outlier removal.
        if self.background_subtract_mode!='none' or self.remove_outliers:
            for line in data:
                chnl_info = tuple(line[0:4])
                if chnl_info in self.chnls_to_pad:
                    pad = self.chnls_to_pad[chnl_info]
                else:
                    #print('warning: the following channel tripped but doesn\'t have  a pad mapping: '+str(chnl_info))
                    continue
                if self.background_subtract_mode!='none':
                    line[FIRST_DATA_BIN:] -= self.calculate_background(line[FIRST_DATA_BIN:])
                if self.remove_outliers:
                    x,y = self.pad_to_xy_index[pad]
                    pad_image[x,y]=1
        if self.remove_outliers:
            labeled_image = skimage.measure.label(pad_image, background=0)
            labels, counts = np.unique(labeled_image[labeled_image!=0], return_counts=True)
            bigest_label = labels[np.argmax(counts)]
            new_data = []
            for line in data: #only copy over pads in the bigest blob and veto pads
                chnl_info = tuple(line[0:4])
                if chnl_info in self.chnls_to_pad:
                    pad = self.chnls_to_pad[chnl_info]
                else:
                    #print('warning: the following channel tripped but doesn\'t have  a pad mapping: '+str(chnl_info))
                    continue
                x,y = self.pad_to_xy_index[pad]
                if labeled_image[x,y] == bigest_label or pad in VETO_PADS:
                    new_data.append(line)
            data = np.array(new_data)
        
        if self.mode == 'peak only': #zero everything but the peak bin
            for line in data:
                peak_index = np.argmax(line[FIRST_DATA_BIN:])
                line[FIRST_DATA_BIN:FIRST_DATA_BIN+peak_index] = 0
                if peak_index < len(line[FIRST_DATA_BIN:]):
                    line[FIRST_DATA_BIN+peak_index+1:] = 0
        elif self.mode == 'near peak': #zero everything outside the window
            for line in data:
                peak_index = np.argmax(line[FIRST_DATA_BIN:])
                if peak_index < self.require_peak_within[0] or peak_index > self.require_peak_within[1]:
                    line[FIRST_DATA_BIN:] = 0
                else:
                    if peak_index - self.near_peak_window_width > 0:
                        line[FIRST_DATA_BIN:FIRST_DATA_BIN+peak_index - self.near_peak_window_width] = 0
                    if peak_index + self.near_peak_window_width < len(line[FIRST_DATA_BIN:]):
                        line[FIRST_DATA_BIN+peak_index + self.near_peak_window_width:] = 0
        
        return data

    def calculate_background(self, trace):
        '''
        Return calculated background for each timebin.

        Trace should only contain data bins
        '''
        #apply consnant offset of average value of a pad within a time window
        if self.background_subtract_mode == 'fixed window':
            return np.average(trace[self.num_background_bins[0]:self.num_background_bins[1]])
        #rolling average to each side of a pad
        elif self.background_subtract_mode == 'convolution':
            return np.convolve(trace, self.background_convolution_kernel, mode='same')
        #in the case of none, just return an array of 0s
        elif self.background_subtract_mode == 'none':
            return np.zeros(len(trace))
        assert False #invalid mode

    def get_xyte(self, event_number, threshold=-np.inf, include_veto_pads=True):
        '''
        Returns: xs, ys, ts, es
                 Where each of these is an array s.t. each "pixel" in the in the raw TPC data is represented.
                 eg, (xs[i], ys[i], ts[i]) gives the position of a pad and time bin number,
                 and es[i] gives the charge that arrived at that pad at the given time.
                 Only data where the charge deposition is greater than the threshold is included.
        '''
        xs, ys, es = [], [], []
        event_data =  self.get_data(event_number)
        #after this look, xs=[x1, x2, ...], same for ys, es=[[1st pad data], [2nd pad data], ...]
        for pad_data in event_data:
            chnl_info = tuple(pad_data[0:4])
            if chnl_info not in self.chnls_to_xy_coord:
                #print('warning: the following channel tripped but doesn\'t have  a pad mapping: '+str(chnl_info))
                continue
            if not include_veto_pads:
                pad = self.chnls_to_pad[chnl_info]
                if pad in VETO_PADS:
                    continue
            x,y = self.chnls_to_xy_coord[chnl_info]
            xs.append(x)
            ys.append(y)
            es.append(pad_data[FIRST_DATA_BIN:])
        #reshape as needed to get to final format for x,y,e
        NUM_TIME_BINS = 512+5-FIRST_DATA_BIN
        xs = np.repeat(xs, NUM_TIME_BINS)
        ys = np.repeat(ys, NUM_TIME_BINS)
        es = np.array(es).flatten()

        #make time bins data
        ts = np.tile(np.arange(0, NUM_TIME_BINS), int(len(xs)/NUM_TIME_BINS))

        #apply thresholding
        if threshold != -np.inf:
            xs = xs[es>threshold]
            ys = ys[es>threshold]
            ts = ts[es>threshold]
            es = es[es>threshold]
        return xs, ys, ts, es
    
    def get_xyze(self, event_number, threshold=-np.inf, include_veto_pads=True):
        '''
        Same as xyte, but scales time bins to get z coordinate
        '''
        x,y,t,e = self.get_xyte(event_number, threshold=threshold, include_veto_pads=include_veto_pads)
        return x,y, t*self.zscale ,e
    
    def get_event_num_bounds(self):
        #returns first event number, last event number
        return int(self.h5_file['meta']['meta'][0]), int(self.h5_file['meta']['meta'][2])
        #return int(self.h5_file['meta']['meta'][0]), np.min((int(self.h5_file['meta']['meta'][2]), int(self.h5_file['meta']['meta'][0])+10000))#TODO


    def get_pad_traces(self, event_number):
        '''
        returns [pads which fired], [[time series data for first pad], [time series data for 2nd pad], ...]
        Pad numbers are determined from AGET, COBO, and channel number, rather than the pad number written during
        the merging process.
        '''
        pads, pad_datas = [], []
        event_data =  self.get_data(event_number)
        for line in event_data:
            chnl_info = tuple(line[0:4])
            if chnl_info in self.chnls_to_pad:
                pad = self.chnls_to_pad[chnl_info]
            else:
                #print('warning: the following channel tripped but doesn\'t have  a pad mapping: '+str(chnl_info))
                continue
            pads.append(pad)
            pad_datas.append(line[FIRST_DATA_BIN:])
        return pads, pad_datas
    
    def get_num_pads_fired(self, event_number):
        event = self.get_data(event_number)
        return len(event)
    
    def get_track_length_angle(self, event_number):
        '''
        1. Remove all points which are less than threshold sigma above background
        2. Find max distance between any of the two remaining points.
        Should replace this with something more robust in the future. This will NOT
        well work if outlier removal and background subtraction haven't been performed.

        Returns: length, angle from z-axis in radians
        '''
        xs, ys, zs, es = self.get_xyze(event_number, self.length_counts_threshold, include_veto_pads=False)
        if len(xs) == 0:
            return 0, 0
        points = np.vstack((xs, ys, zs)).T
        #print(points)
        #find max distance using this algorithm
        #https://stackoverflow.com/questions/31667070/max-distance-between-2-points-in-a-data-set-and-identifying-the-points
        try:
            hull = scipy.spatial.ConvexHull(points)
            points = points[hull.vertices,:]
        except: #qhull will fail on colinear points, so just brute force if that's the case
            pass
        #find the two most distant points
        hdist = scipy.spatial.distance.cdist(points, points, metric='euclidean')
        indices = np.unravel_index(np.argmax(hdist, axis=None), hdist.shape)
        p1 = points[indices[0]]
        p2 = points[indices[1]]
        #print(p1, p2)
        dist = hdist[indices]
        dr = p1-p2
        if dr[2] != 0:
            angle = np.abs(np.arctan(np.sqrt(dr[0]**2 + dr[1]**2)/dr[2]))
        else:
            angle = np.radians(90)
        return dist, angle

    
    def determine_pad_backgrounds(self, num_background_bins=200, mode='background'):
        '''
        Assume the first num_background_bins of each pad's data only include background.
        Determine average value of this pad and stddev across all events in which the pad fired.
        Store this information in a dictionairy member variable.

        Pad background will be stored in self.pad_backgrounds, which is a dictionairy indexed by pad
        number which stores (background average, background standard deviation) pairs.
        
        Mode = background: determine average number of counts in background region
        Mode = average: determine average number of counts above background (if background subtraction is turned on)
        '''
        first, last = self.get_event_num_bounds()

        #compute average
        running_averages = {}
        for event_num in range(first, last+1):
            if mode == 'background':
                self.h5_file['get']['evt%d_data'%event_num]
            elif mode == 'average':
                event_data = self.get_data(event_num)#
            for line in event_data:
                chnl_info = tuple(line[0:4])
                if chnl_info in self.chnls_to_pad:
                    pad = self.chnls_to_pad[chnl_info]
                else:
                    print('warning: the following channel tripped but doesn\'t have  a pad mapping: '+str(chnl_info))
                    continue
                if pad not in running_averages:
                    running_averages[pad] = (0,0) #running everage, events processed
                if mode == 'background':
                    ave_this = np.average(line[FIRST_DATA_BIN+self.num_background_bins[0]:self.num_background_bins[1]+FIRST_DATA_BIN])
                elif mode == 'average':
                    ave_this = np.average(line[FIRST_DATA_BIN:511+FIRST_DATA_BIN])
                ave_last, n = running_averages[pad]
                running_averages[pad] = ((n*ave_last + ave_this)/(n+1), n+1)
        #compute standard deviation
        running_stddev = {}
        for event_num in range(first, last+1):
            event_data = self.h5_file['get']['evt%d_data'%event_num]
            for line in event_data:
                chnl_info = tuple(line[0:4])
                if chnl_info in self.chnls_to_pad:
                    pad = self.chnls_to_pad[chnl_info]
                else:
                    print('warning: the following channel tripped but doesn\'t have  a pad mapping: '+str(chnl_info))
                    continue
                if pad not in running_stddev:
                    running_stddev[pad] = (0,0)
                std_this = np.std(line[FIRST_DATA_BIN:num_background_bins+FIRST_DATA_BIN])
                std_last, n = running_stddev[pad]
                running_stddev[pad] = ((n*std_last + std_this)/(n+1), n+1)
        self.pad_backgrounds = {}
        for pad in running_averages:
            self.pad_backgrounds[pad] = (running_averages[pad][0], running_stddev[pad][0])
    
    def get_histogram_arrays(self):
        range_hist = []
        counts_hist = []
        angle_hist = []
        pads_railed_list = []
        accepted_events = []
        for i in tqdm.tqdm(range(*self.get_event_num_bounds())):#TODO: is this missing the last event in the run?
            should_veto, length, energy, angle, pads_railed = self.process_event(i)
            if not should_veto:
                range_hist.append(length)
                counts_hist.append(energy)
                angle_hist.append(angle)
                pads_railed_list.append(pads_railed)
                accepted_events.append(i)

        return np.array(range_hist), np.array(counts_hist), np.array(angle_hist), pads_railed_list, accepted_events

    def process_event(self, event_num):
        '''
        Returns: should veto, range, energy, angle, pads_railed
        '''
        should_veto=False
        counts = 0
        pads_railed = []
        for pad, trace in zip(*self.get_pad_traces(event_num)):
            if pad in VETO_PADS:
                if np.any(trace>self.veto_threshold):
                    should_veto = True
            if self.include_counts_on_veto_pads or not pad in VETO_PADS: #don't inlcude veto pad energy
                counts += np.sum(trace[trace>self.ic_counts_threshold])
            if np.max(trace) >= 4095:
                pads_railed.append(pad)
        length, angle = self.get_track_length_angle(event_num)
        if self.range_bounds != None:
            if length > self.range_bounds[1] or length < self.range_bounds[0]:
                should_veto = True
        if self.ic_bounds != None:
            if counts < self.ic_bounds[0] or counts > self.ic_bounds[1]:
                should_veto = True
        if self.angle_bounds != None:
            if angle < self.angle_bounds[0] or angle > self.angle_bounds[1]:
                should_veto = True
        return should_veto, length, counts, angle, pads_railed

    def show_counts_histogram(self, num_bins, fig_name=None, block=True):
        ranges, counts, angles, pads_railed_list, accepted_events_list = self.get_histogram_arrays()
        plt.figure(fig_name)
        plt.hist(counts, bins=num_bins)
        plt.show(block=block)

    def show_rve_histogram(self, num_e_bins, num_range_bins, fig_name=None, block=True):
        ranges, counts, angles, pads_railed_list,accepted_events_list = self.get_histogram_arrays()

        #TODO: make generic, these are P10 values
        calib_point_1 = (0.806, 156745)
        calib_point_2 = (1.679, 320842)
        energy_1, channel_1 = calib_point_1
        energy_2, channel_2 = calib_point_2
        energy_scale_factor = (energy_2 - energy_1) / (channel_2 - channel_1)
        energy_offset = energy_1 - energy_scale_factor * channel_1
        counts = energy_scale_factor*counts + energy_offset

        plt.figure(fig_name)
        plt.hist2d(counts, ranges, 
                   bins=(num_e_bins, num_range_bins), norm=colors.LogNorm())
        plt.xlabel('energy (MeV)')
        plt.ylabel('range (mm)')
        plt.colorbar()
        plt.show(block=block)

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
            r = pad/1024*.8
            g = (pad%512)/512*.8
            b = (pad%256)/256*.8
            if pad in VETO_PADS:
                plt.plot(data, '--', color=(r,g,b), label='%d'%pad)
            else:
                plt.plot(data, color=(r,g,b), label='%d'%pad)
        plt.legend(loc='upper right')
        plt.show(block=block)

    def plot_3d_traces(self, event_num, threshold=-np.inf, block=True, fig_name=None):
        fig = plt.figure(fig_name, figsize=(6,6))
        plt.clf()
        ax = plt.axes(projection='3d')
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        ax.set_xlim3d(-200, 200)
        ax.set_ylim3d(-200, 200)
        ax.set_zlim3d(0, 400)

        xs, ys, zs, es = self.get_xyze(event_num, threshold=threshold)

        #TODO: make generic, these are P10 values
        calib_point_1 = (0.806, 156745)
        calib_point_2 = (1.679, 320842)
        energy_1, channel_1 = calib_point_1
        energy_2, channel_2 = calib_point_2
        energy_scale_factor = (energy_2 - energy_1) / (channel_2 - channel_1)
        energy_offset = energy_1 - energy_scale_factor * channel_1

        ax.view_init(elev=45, azim=45)
        ax.scatter(xs, ys, zs, c=es, cmap=self.cmap)
        cbar = fig.colorbar(ax.get_children()[0])
        should_veto, length, energy, angle, pads_railed = self.process_event(event_num)
        plt.title('event %d, total counts=%d / %f MeV\n length=%f mm, angle=%f deg\n veto=%d\n # pads railed=%d'%(event_num, energy, 
                                                                                               energy*energy_scale_factor + energy_offset, length,
                                                                                               np.degrees(angle), should_veto, len(pads_railed)))
        plt.show(block=block)
    
    def show_2d_projection(self, event_number, block=True, fig_name=None):
        data = self.get_data(event_number)
        image = np.zeros(np.shape(self.pad_plane))
        for line in data:
            chnl_info = tuple(line[0:4])
            if chnl_info not in self.chnls_to_pad:
                print('warning: the following channel tripped but doesn\'t have  a pad mapping: '+str(chnl_info))
                continue
            pad = self.chnls_to_pad[chnl_info]
            x,y = self.pad_to_xy_index[pad]
            image[x,y] = np.sum(line[FIRST_DATA_BIN:])
        image[image<0]=0
        trace = np.sum(data[:,FIRST_DATA_BIN:],0)
        

        fig = plt.figure(fig_name, figsize=(6,6))
        plt.clf()
        should_veto, length, energy, angle, pads_railed_list = self.process_event(event_number)
        plt.title('event %d, total counts=%d, length=%f mm, angle=%f, veto=%d'%(event_number, energy, length, np.degrees(angle), should_veto))
        plt.subplot(2,1,1)
        plt.imshow(image, norm=colors.LogNorm())
        plt.colorbar()
        plt.subplot(2,1,2)
        plt.plot(trace)
        plt.show(block=block)

        


'''
import h5py
file = h5py.File('/mnt/analysis/e21072/gastest_h5_files/run_0032.h5', 'r')
a=file['get']['evt3057_data']

import raw_h5_file
from importlib import reload
reload(raw_h5_file)
file = raw_h5_file.raw_h5_file('/mnt/analysis/e21072/gastest_h5_files/run_0037.h5')
file.remove_outliers=True
file.get_data(553)
'''



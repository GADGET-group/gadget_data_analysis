import os

import numpy as np
import scipy.spatial
import scipy.spatial.distance
import h5py
import matplotlib.pylab as plt
import matplotlib.colors as colors
from matplotlib.colors import LinearSegmentedColormap
import tqdm

import cupy as cp
import cupyx.scipy.special as cpspecial



import skimage.measure

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
        self.pad_to_chnl = {}
        for line in self.flat_lookup:
            chnls = tuple(line[0:4])
            pad = line[4]
            self.chnls_to_pad[chnls] = pad
            self.pad_to_chnl[pad] = chnls
            self.chnls_to_xy_coord[chnls] = self.padxy[pad]
            self.chnls_to_xy_index[chnls] = self.pad_to_xy_index[pad]
        #round xy to nearest 10nths place to avoid issues with different floating point formats
        self.xy_to_pad = {tuple(np.round(self.padxy[pad], 1)):pad for pad in range(len(self.padxy))}
        self.xy_to_chnls = {tuple(np.round(self.chnls_to_xy_coord[chnls], 1)):chnls 
                            for chnls in self.chnls_to_xy_coord}
        self.xy_index_to_pad = {self.pad_to_xy_index[pad]:pad for pad in self.pad_to_xy_index}
        
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

        self.background_subtract_mode = 'none' #none, fixed window, smart, or convolution
        self.background_convolution_kernel = None#bin backgrounds are determined by convolving the trace with this array
        self.remove_outliers = False
        self.num_background_bins = (0,0) #number of time bins to use for per event background subtraction
        self.length_counts_threshold = 300 #threshold to use when calculating range
        self.ic_counts_threshold = 100 #threshold to use when calculating energy
        #allowed modes = 'all data', 'peak only', or 'near peak'
        #all data mode will make use of the entire trace from each pad
        #'peak only' will make use of only the max value the trace reached on each pad
        #'near peak' will only use data within some window of the peak of each trace
        self.data_select_mode = 'all data' 
        self.near_peak_window_width = 100 #+/- time bins to include
        self.require_peak_within = (-np.inf, np.inf)#currentlt implemented for near peak mode only. Zero entire trace if peak is not within this window
        self.include_counts_on_veto_pads = False #if counts on veto pads should be included for energy calibraiton

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

    def get_timestamp(self, event_number):
        '''
        Returns timestamp in seconds.
        '''
        #timestamps are stored in units of 10ns
        return self.h5_file['get']['evt%d_header'%event_number][1]/1e8 

    def get_timestamps_array(self):
        '''
        Returns an array with the timestamps of each event, in seconds
        '''
        first, last = self.get_event_num_bounds()
        num_events = last - first + 1
        to_return = np.zeros(num_events)
        for i, evt in tqdm.tqdm(enumerate(range(first, last+1))):
            to_return[i] = self.get_timestamp(evt)
        return to_return

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
        
        if self.data_select_mode == 'peak only': #zero everything but the peak bin
            for line in data:
                peak_index = np.argmax(line[FIRST_DATA_BIN:])
                line[FIRST_DATA_BIN:FIRST_DATA_BIN+peak_index] = 0
                if peak_index < len(line[FIRST_DATA_BIN:]):
                    line[FIRST_DATA_BIN+peak_index+1:] = 0
        elif self.data_select_mode == 'near peak': #zero everything outside the window
            for line in data:
                peak_index = np.argmax(line[FIRST_DATA_BIN:])
                if peak_index < self.require_peak_within[0] or peak_index > self.require_peak_within[1]:
                    line[FIRST_DATA_BIN:] = 0
                else:
                    if peak_index - self.near_peak_window_width > 0:
                        line[FIRST_DATA_BIN:FIRST_DATA_BIN+peak_index - self.near_peak_window_width] = 0
                    if peak_index + self.near_peak_window_width < len(line[FIRST_DATA_BIN:]):
                        line[FIRST_DATA_BIN+peak_index + self.near_peak_window_width:] = 0
        # To save with a delimiter (e.g., comma)
        np.savetxt('data_comma.txt', data, delimiter=',')

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
        elif self.background_subtract_mode == 'smart':
            peak_index = np.argmax(trace)
            #find  start of the peak, defined as where the a bin near_peak_window_width away
            #is no longer at least ic_counts_threshold below the current bin
            i = peak_index
            while i>0:
                j = max(0, i - self.near_peak_window_width)
                if trace[i] < trace[j] + self.ic_counts_threshold:
                    break
                i -= 1
            peak_start = i
            #find end
            i = peak_index
            while i < len(trace) - 1:
                j = min(len(trace) - 1, i + self.near_peak_window_width)
                if trace[i] < trace[j] + self.ic_counts_threshold:
                    break
                i += 1
            peak_end = i
            #fit a line through the points just outside the peak region
            xs = np.concatenate([np.arange(max(0, peak_start - self.near_peak_window_width), peak_start),
                                           np.arange(peak_end, min(peak_end + self.near_peak_window_width, len(trace)))])
            ys = trace[xs]
            slope, offset = np.polyfit(xs, ys, 1)
            #baseline will be the trace except in the peak region,
            #so that everything away from the peak is zero'd out
            baseline = np.array(trace, copy=True)
            for i in range(peak_start, peak_end+1):
                baseline[i] = slope*i + offset
            return baseline
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
    
    def get_xyze(self, event_number, threshold=-np.inf, include_veto_pads=False):
        '''
        Same as xyte, but scales time bins to get z coordinate
        '''
        x,y,t,e = self.get_xyte(event_number, threshold=threshold, include_veto_pads=include_veto_pads)
        return x,y, t*self.zscale ,e
    
    def get_track_axis(self, event=None, threshold=None, return_all_svd_results=False, xyze=None, return_np=True):
        '''
        Uses SVD on all points above some threshold to get track direction.
        Returns point, unit vector in track direction.
        If threshold==None, uses track length ic threshold.
        
        Either event number must be passed, or xyze which contans tuple of cupy arrays of 
        xyze points. This can be useful to avoid transfering data to GPU twice

        return_np specifies if data should be moved back to cpu from gpu 
        '''
        if threshold == None:
            threshold = self.length_counts_threshold
        if type(event) != type(None):
            x,y,z,e = self.get_xyze(event, threshold, False)
            x, y,z,e = cp.array(x), cp.array(y), cp.array(z), cp.array(e)
        else:
            x, y, z, e = xyze
        points = cp.concatenate((x[:, cp.newaxis], y[:, cp.newaxis], z[:, cp.newaxis]), axis=1)
        points_mean = points.mean(axis=0)
        uu, dd, vv = cp.linalg.svd(points - points_mean)
        if return_np:
            points_mean = cp.asnumpy(points_mean)
            uu, dd, vv = cp.asnumpy(uu), cp.asnumpy(dd), cp.asnumpy(vv)
        if return_all_svd_results:
            return points_mean, uu, dd, vv
        else:
            return points_mean, vv #vv[0] holds direction vector of 1st priciple component, etc

    
    def get_event_num_bounds(self):
        #returns first event number, last event number
        return int(self.h5_file['meta']['meta'][0]), int(self.h5_file['meta']['meta'][2])
        #return int(self.h5_file['meta']['meta'][0]), np.min((int(self.h5_file['meta']['meta'][2]), int(self.h5_file['meta']['meta'][0])+10000))#TODO


    def get_pad_traces(self, event_number, include_veto_pads=True):
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
            if include_veto_pads or pad not in VETO_PADS:
                pads.append(pad)
                pad_datas.append(line[FIRST_DATA_BIN:])
        return pads, pad_datas
    
    def get_num_pads_fired(self, event_number):
        event = self.get_data(event_number)
        return len(event)
    
    def get_track_length_angle(self, event_number, return_np=True):
        '''
        1. Determin track principle axis, and use this to get azimuthal angle
        2. Find point furthest along this axis, in each direction

        Returns: length, angle from z-axis in radians
        '''
        xs, ys, zs, es = self.get_xyze(event_number, self.length_counts_threshold, include_veto_pads=False)
        if len(xs) < 2:
            return 0, 0, 0
        #move points to GPU
        xs, ys, zs, es = cp.array(xs), cp.array(ys), cp.array(zs), cp.array(es)
        track_center, vv = self.get_track_axis(xyze=(xs, ys, zs, es), return_np=False)
        track_direction = vv[0]/cp.sqrt(np.sum(vv[0]*vv[0]))
        angle = cp.arctan2(np.sqrt(track_direction[0]**2 + track_direction[1]**2),cp.abs(track_direction[2]))

        points = cp.concatenate((xs[:, cp.newaxis], 
                       ys[:, cp.newaxis], 
                       zs[:, cp.newaxis]), 
                      axis=1)
        rbar = points - track_center
        rdotv = cp.dot(rbar, track_direction)
        first_point = points[cp.argmin(rdotv)]
        last_point = points[cp.argmax(rdotv)]
        dr = last_point - first_point
        dxy, dz = cp.sqrt(dr[0]**2 + dr[1]**2), dr[2]
        if return_np:
            dxy, dz, angle = cp.asnumpy(dxy), cp.asnumpy(dz), cp.asnumpy(angle)
        return dxy, dz, angle
    
   
    def get_histogram_arrays(self):
        max_veto_counts_list, pads_railed_list, angle_hist, counts_hist, dxy_hist, dz_hist = [], [], [], [], [], []
        first, last = self.get_event_num_bounds()
        for i in tqdm.tqdm(range(first, last+1)):
            max_veto_counts, dxy, dz, energy, angle, pads_railed = self.process_event(i)
            dxy_hist.append(dxy)
            dz_hist.append(dz)
            counts_hist.append(energy)
            angle_hist.append(angle)
            pads_railed_list.append(pads_railed)
            max_veto_counts_list.append(max_veto_counts)

        return np.array(max_veto_counts_list), np.array(dxy_hist), np.array(dz_hist), np.array(counts_hist), np.array(angle_hist), pads_railed_list

    def process_event(self, event_num):
        '''
        Returns: max veto counts, dxy, dz, energy, angle, pads_railed
        max veto counts is max counts in any single time bin on a single veto pad
        dxy is the track length in the pad plane, dz is the other component of track length
        '''
        should_veto=False
        counts = 0
        pads_railed = []
        max_veto_pad_counts = -np.inf
        for pad, trace in zip(*self.get_pad_traces(event_num)):
            trace_max = np.max(trace)
            if pad in VETO_PADS:
                if  trace_max > max_veto_pad_counts:
                    max_veto_pad_counts = trace_max
            if self.include_counts_on_veto_pads or not pad in VETO_PADS: #don't inlcude veto pad energy
                counts += np.sum(trace[trace>self.ic_counts_threshold])
            if trace_max >= 4095:
                pads_railed.append(pad)
        dxy, dz, angle = self.get_track_length_angle(event_num)
        return max_veto_pad_counts, dxy, dz, counts, angle, pads_railed

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
        max_veto_counts, dxy, dz, energy, angle, pads_railed = self.process_event(event_num)
        length = np.sqrt(dxy**2 + dz**2)
        plt.title('event %d, total counts=%d / %f MeV\n length=%f mm, angle=%f deg\n # pads railed=%d'%(event_num, energy, 
                                                                                               energy*energy_scale_factor + energy_offset, length,
                                                                                               np.degrees(angle), len(pads_railed)))
        plt.show(block=block)
    
    def get_2d_image(self, data):
        image = np.zeros(np.shape(self.pad_plane))
        for pad in data:
            x,y = self.pad_to_xy_index[pad]
            image[y,x] = data[pad]
        #image[image<0]=0
        return image

    def show_padplane_image(self, data, trace_dict=None, block=False, fig_name=None, title=''):
        '''
        Shows a figure with pad plane displaying "data", above the sum of all "traces" (if not None).
        Clicking on a pad will show the corresponding trace 
        data: Dictionary of pixel brightnesses indexed by pad number
        traces: optional. If passed, sould contain trace to dispay when a pad is clicked. Summed version of this 
                will be shown below the pad plane.
        '''
        image = self.get_2d_image(data)

        fig = plt.figure(fig_name, figsize=(6,6))
        plt.clf()
        plt.title(title)
        plt.subplot(2,1,1)
        plt.imshow(image, norm=colors.LogNorm())
        plt.colorbar()
        plt.subplot(2,1,2)
        plt.plot(np.sum([trace_dict[pad] for pad in trace_dict], axis=0))
        def onclick(event):
            x, y = int(np.round(event.xdata)), int(np.round(event.ydata))
            pad = self.xy_index_to_pad[(x,y)]
            if pad in trace_dict:
                plt.figure()
                plt.title('cobo %d, asad %d, aget %d, chnl %d, pad %d'%(*self.pad_to_chnl[pad], pad))
                plt.plot(trace_dict[pad])
                plt.show(block=False)

        fig.canvas.mpl_connect('button_press_event', onclick)
        plt.show(block=block)


    def show_2d_projection(self, event_number, block=True, fig_name=None):
        pads, traces = self.get_pad_traces(event_number)
        trace_dict = {pad: trace for pad, trace in zip(pads, traces)}
        data = {pad:np.sum(trace_dict[pad]) for pad in trace_dict}
        should_veto, dxy, dz, energy, angle, pads_railed_list = self.process_event(event_number)
        length = np.sqrt(dxy**2 + dz**2)
        title='event %d, total counts=%d, length=%f mm, angle=%f, veto=%d'%(event_number, energy, length, np.degrees(angle), should_veto)
        self.show_padplane_image(data, trace_dict=trace_dict, block=block, fig_name=fig_name, title=title)
        

    def show_traces_w_baseline_estimate(self, event_num, block=True, fig_name=None):
        '''
        plots traces without background subtraction, with backgrounds shown as ... lines
        '''
        plt.figure(fig_name)
        plt.clf()
        old_background_mode = self.background_subtract_mode
        self.background_subtract_mode = 'none' #will set back after drawing traces
        old_mode = self.data_select_mode
        self.data_select_mode = 'all data'
        pads, pad_data = self.get_pad_traces(event_num)
        for pad, data in zip(pads, pad_data):
            r = pad/1024*.8
            g = (pad%512)/512*.8
            b = (pad%256)/256*.8
            if pad in VETO_PADS:
                plt.plot(data, '--', color=(r,g,b), label='%d'%pad)
            else:
                plt.plot(data, color=(r,g,b), label='%d'%pad)
        self.background_subtract_mode = old_background_mode
        for pad, data in zip(pads, pad_data):
            r = pad/1024*.8
            g = (pad%512)/512*.8
            b = (pad%256)/256*.8
            plt.plot(self.calculate_background(data), '.', color=(r,g,b), label='%d baseline'%pad)
        self.data_select_mode = old_mode
        plt.legend(loc='upper right')
        plt.show(block=block)

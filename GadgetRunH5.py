'''
This file contains the following
-- Class which handles accessing information from a single run. Per-event
   quantities come from raw_viewer.process_runs (one ROOT file per run);
   per-event images are built from the raw h5 file.
'''
import os
import io
import math
import pickle

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.path
from tqdm import tqdm
from scipy.signal import savgol_filter

from raw_viewer import process_runs
from raw_viewer import raw_h5_file

VETO_PADS = raw_h5_file.VETO_PADS

def get_default_pad_gains(experiment='e25058'):
    '''
    Pad gains used by the e25058 analysis (see raw_viewer/plots/rve.py): the
    combined-run FFT gain match with the veto pads zeroed. Units are MeV/ADC,
    so process_runs.get_gm_ic returns energy in MeV with these gains.
    '''
    if experiment != 'e25058':
        raise ValueError(f'no default pad gains for {experiment!r}; pass gains explicitly')
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        'raw_viewer', 'plots', 'fft6_res3.pkl')
    with open(path, 'rb') as f:
        return np.asarray(pickle.load(f).pad_gains, dtype=np.float64)

def smooth_trace(trace, window_length=15, polyorder=3):
        smoothed_trace = savgol_filter(trace, window_length, polyorder)
        return smoothed_trace

def remove_noise(trace, threshold_ratio=0.1):
        threshold = threshold_ratio * np.max(np.abs(trace))
        trace[np.abs(trace) < threshold] = 0

        # Remove negative values
        trace[trace < 0] = 0

        # Find the index of the maximum value in the trace
        max_idx = np.argmax(trace)

        # Zero out bins to the left of the maximum value if a zero bin is encountered
        for i in range(max_idx - 1, -1, -1):
            if trace[i] == 0:
                trace[:i] = 0
                break

        # Zero out bins to the right of the maximum value if a zero bin is encountered
        for i in range(max_idx + 1, len(trace)):
            if trace[i] == 0:
                trace[i:] = 0
                break

        return trace

class GadgetRunH5:
    def __init__(self, run_num, folder_path, experiment='e25058', gains=None, require_processed=True):
        '''
        run_num: run number
        folder_path: directory in which cut-image folders are written
            (RvE_Frame.save_cut_files, prev_cut_select_window)
        experiment: experiment name understood by raw_viewer.process_runs
        gains: per-pad gains passed to process_runs.get_gm_ic. None uses
            get_default_pad_gains(experiment), which gives energy in MeV. Pass
            e.g. np.ones(1024) with the veto pads zeroed to get raw ADC counts.
        require_processed: if True, raise FileNotFoundError when the run's ROOT
            file does not exist, instead of letting process_runs process the
            run (hours) inside a GUI callback.

        Note: process_runs locates the pad map relative to the working
        directory, so run the GUI from the repository root.
        '''
        self.run_num = int(run_num)
        self.experiment = experiment
        self.folder_path = folder_path

        root_file = os.path.join(process_runs.get_save_path(experiment),
                                 f'{experiment}_run{self.run_num}.root')
        if require_processed and not os.path.exists(root_file):
            raise FileNotFoundError(f'{root_file} does not exist: process the run first, e.g. '
                                    f'process_runs.process_tpc_run({experiment!r}, {self.run_num})')

        # raw h5 file, configured (zscale, thresholds, background subtraction) by process_runs
        self.h5_file = process_runs.get_h5_file(experiment, self.run_num)
        self.file_path = self.h5_file.file_path
        self.first_event, self.last_event = self.h5_file.get_event_num_bounds()
        self.event_nums = np.arange(self.first_event, self.last_event + 1)

        # per-event quantities from the processed ROOT file. Every array covers
        # all events first_event..last_event, in order; see get_index/get_event_num.
        runs = [self.run_num]
        if gains is None:
            self.gains = get_default_pad_gains(experiment)
            self.energy_units = 'MeV'
        else:
            self.gains = np.asarray(gains, dtype=np.float64)
            self.energy_units = 'adc counts'
        self.counts = process_runs.get_gm_ic(experiment, runs, self.gains)
        self.ranges = process_runs.get_lengths(experiment, runs)              # mm, 3D track length
        self.angles = np.degrees(process_runs.get_angle(experiment, runs))    # degrees from the beam axis
        self.max_veto_counts = process_runs.get_max_veto_counts(experiment, runs)
        self.timestamps = process_runs.get_quantity('timestamps', experiment, runs)
        self.time_since_beam_off = process_runs.get_time_since_beam_off(experiment, runs)
        self.num_railed_pads = np.array([len(pads) for pads in
                                         process_runs.get_quantity('railed_pads', experiment, runs)])
        num_events = len(self.event_nums)
        for name in ('counts', 'ranges', 'angles', 'max_veto_counts', 'timestamps',
                     'time_since_beam_off', 'num_railed_pads'):
            if len(getattr(self, name)) != num_events:
                raise ValueError(f'{name} has {len(getattr(self, name))} entries '
                                 f'but run {self.run_num} has {num_events} events')

        # names used by older frames (EnergySpectrumFrame, RvE_Frame.show_event)
        self.total_energy = self.counts
        self.total_energy_MeV = self.counts
        self.len_list = self.ranges

        # threshold on counts for the hits used in the images (get_xyze)
        self.image_hit_threshold = 20

        #TODO: decide how to store calibration information with runs
        # legacy e21072 two-point calibration; only the image energy bar uses it (to_MeV)
        calib_point_1 = (0.806, 156745)
        calib_point_2 = (1.679, 320842)
        energy_1, channel_1 = calib_point_1
        energy_2, channel_2 = calib_point_2
        self.energy_scale_factor = (energy_2 - energy_1) / (channel_2 - channel_1)
        self.energy_offset = energy_1 - self.energy_scale_factor * channel_1

    def get_event_num_bounds(self):
        #returns first event number, last event number
        return self.h5_file.get_event_num_bounds()
        
    def to_MeV(self, counts):
        return counts*self.energy_scale_factor + self.energy_offset
    
    def to_counts(self, MeV):
        return (MeV - self.energy_offset)/self.energy_scale_factor

    def get_index(self, event_num):
        '''
        Index into the per-event arrays (counts, ranges, ...) of an event number.
        Accepts a scalar or an array of event numbers.
        '''
        index = np.asarray(event_num, dtype=np.int64) - self.first_event
        if np.any(index < 0) or np.any(index >= len(self.event_nums)):
            raise IndexError(f'event number {event_num} outside [{self.first_event}, {self.last_event}]')
        return index if np.ndim(index) else int(index)

    def get_event_num(self, index):
        '''
        Event number of an index into the per-event arrays (inverse of get_index).
        Accepts a scalar or an array of indices, e.g. the output of get_RvE_cut_indexes.
        '''
        event_num = self.event_nums[index]
        return event_num if np.ndim(event_num) else int(event_num)

    def get_hit_lists(self, event_num):
        '''
        x, y, z positions (mm) and counts of the hits of an event, from the raw h5 file.
        '''
        return self.h5_file.get_xyze(event_num, threshold=self.image_hit_threshold, include_veto_pads=False)

    def make_image(self, event_num, save_path=None, show=False, smoothen=False):
        '''
        Make datafused image of event "event_num" from the raw h5 data. Image will
        be saved to "save_path" if not None. 
        '''
        #helper funcitons to populate pad plane grid and energy bar 
        def make_grid():
            """
            "Create Training Data.ipynb"eate grid matrix of MM outline and energy bar, see spreadsheet below
            https://docs.google.com/spreadsheets/d/1_bbg6svfEph_g_Z002rmzTLu8yjQzuj_p50wqs7mMrI/edit?usp=sharing
            """
            row = np.array([63, 47, 39, 31, 27, 23, 19, 15, 15, 11, 11, 7, 7, 7, 7, 3, 3, 3, 3, 3, 3, 3, 7, 7, 
                    7, 7, 11, 11, 15, 15, 19, 23, 27, 31, 39, 47, 63]) 

            to_row = np.array([87, 103, 111, 119, 123, 127, 131, 135, 135, 139, 139, 143, 143, 143, 143, 147, 
                        147, 147, 147, 147, 147, 148, 143, 143, 143, 144, 139, 140, 135, 136, 132, 128,
                        124, 120, 112, 104, 88]) 

            col = np.array([100, 84, 76, 68, 64, 60, 56, 52, 52, 48, 48, 44, 44, 44, 44, 40, 40, 40, 40, 40, 
                    40, 40, 44, 44, 44, 44, 48, 48, 52, 52, 56, 60, 64, 68, 76, 84, 100])

            to_col = np.array([124, 140, 148, 156, 160, 164, 168, 172, 172, 176, 176, 180, 180, 180, 180, 184, 
                        184, 184, 184, 184, 184, 184, 180, 180, 180, 180, 176, 176, 172, 172, 168, 164, 
                        160, 156, 148, 140, 124]) 

            all_row = np.array([i for i in range(3, 148, 4)])
            all_col = np.array([i for i in range(40, 185, 4)]) 

            full_image_size_width = 224
            full_image_size_length = 151
            mm_grid = np.zeros((full_image_size_length, full_image_size_width, 3))    
            mm_grid.fill(255)                                                     

            #TODO: replace these loops with numpy indexing
            for i in range(len(row)): 
                # draw grid columns, 0 = black
                mm_grid[row[i]:to_row[i], all_col[i], :] = 0
        
            for i in range(len(col)):
                # draw grid rows
                mm_grid[all_row[i], col[i]:to_col[i], :] = 0

            # Ensure that matrix is of integers
            mm_grid = mm_grid.astype(int) 

            # Draw engergy bar box
            mm_grid = make_box(mm_grid)

            return mm_grid
        
        def make_box(mm_grid):
            """
            Draws the box for the energy bar
            """
            box_row = np.array([4, 4])
            to_box_row = np.array([145, 146])
            for_box_col = np.array([7, 17])

            box_col = np.array([7, 7])
            to_box_col = np.array([17, 17])
            for_box_row = np.array([4, 145])

            # Draw vertical lines of energy bar box
            for i in range(len(box_row)):
                mm_grid[box_row[i]:to_box_row[i], for_box_col[i], :] = 0
                mm_grid[for_box_row[i], box_col[i]:to_box_col[i], :] = 0

            return mm_grid

        # def fill_padplane(xset, yset, eset, tot_energy):
        def fill_padplane(xset, yset, eset, tot_energy):
            """
            Fills the 2D pad plane grid for image creation
            """
            pad_plane = make_grid()

            xset = np.array(xset)
            yset = np.array(yset)
            eset = np.array(eset)

            # pad plane mapping
            x = (35 + xset) * 2 + 42    # col value
            y = 145 - (35 + yset) * 2   # row value

            # create a dictionary to store (x,y) as keys and e as values
            d = {}
            for i in range(len(x)):
                key = (x[i], y[i])
                if key in d:
                    d[key] += eset[i]
                else:
                    d[key] = eset[i]

            # convert the dictionary back to arrays
            x = np.zeros(len(d))
            y = np.zeros(len(d))
            eset = np.zeros(len(d))
            for i, key in enumerate(d):
                x[i] = key[0]
                y[i] = key[1]
                eset[i] = d[key]

            # Find max E value and normalize
            energy = eset
            max_energy = np.max(energy)
            norm_energy = energy / max_energy


            # Fill in pad plane   
            for k in range(len(x)):
            
                if y[k] < 9:
                    y[k] = y[k] + 4

                if x[k] < 50:
                    x[k] = x[k] + 4

                if x[k] > 174:
                    x[k] = x[k] - 4

                if y[k] > 53:
                    y[k] = y[k] - 4

                if x[k] > 134:
                    x[k] = x[k] - 4

                if y[k] > 93:
                    y[k] = y[k] - 4

                if y[k] > 133:
                    y[k] = y[k] - 4	

                if x[k] < 90:
                    x[k] = x[k] + 4


                pad_plane[int(y[k])-1:int(y[k])+2, int(x[k])-1:int(x[k])+2, 0] = norm_energy[k] * 205

                pad_plane[int(y[k])-1:int(y[k])+2, int(x[k])-1:int(x[k])+2, 1] = norm_energy[k] * 240
            
            pad_plane = fill_energy_bar(pad_plane, tot_energy)

            return pad_plane


        def trace_image(padplane_image, trace):
            """
            Creates a 2D image from trace data
            """
            # Save plot as jpeg (only want RGB channels, not an alpha channel)
            # Need to take monitor dpi into account to get correct pixel size
            # Plot should have a pixel size of 73x224

            my_dpi = 96
            fig, ax = plt.subplots(figsize=(224/my_dpi, 73/my_dpi))

            x = np.linspace(0, len(trace)-1, len(trace))
            
            ax.tick_params(top=False, bottom=False, left=False, right=False, labelleft=False, labelbottom=False)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.fill_between(x, trace, color='b', alpha=1)
            # render the plot to memory and load it back as an RGB array so that
            # it can be appended to the pad plane plot
            buf = io.BytesIO()
            plt.savefig(buf, format='jpg', dpi=my_dpi)
            plt.close()
            buf.seek(0)
            img = plt.imread(buf, format='jpg')
            buf.close()
            rows,cols,colors = img.shape # gives dimensions for RGB array
            img_size = rows*cols*colors
            img_1D_vector = img.reshape(img_size)

            # you can recover the orginal image with:
            trace_image = img_1D_vector.reshape(rows,cols,colors)

            # append pad plane image with trace image
            complete_image = np.append(padplane_image, trace_image, axis=0)

            return complete_image

        def fill_energy_bar(pad_plane, tot_energy):
            """
            Fills the energy bar where the amount of pixels fired and the color corresponds to the energy of the track
            Max pixel_range should be 28 (7 rows for each color), so need to adjust accordingly.
            """

            def blue_range(pad_plane, rows):

                start_row = 140
                low_color = 0
                high_color = 35

                for i in range(rows):
                    pad_plane[start_row:start_row+5, 8:17, 0] = low_color
                    pad_plane[start_row:start_row+5, 8:17, 1] = high_color
                    start_row = start_row - 5 
                    low_color = low_color + 35
                    high_color = high_color + 35

                return pad_plane


            def yellow_range(pad_plane, rows):

                start_row = 105
                color = 220

                for i in range(rows):
                    pad_plane[start_row:start_row+5, 8:17, 2] = color
                    start_row = start_row - 5 
                    color = color - 15

                return pad_plane


            def orange_range(pad_plane, rows):

                start_row = 70
                color = 210
                for i in range(rows):
                    pad_plane[start_row:start_row+5, 8:17, 1] = color - 15
                    pad_plane[start_row:start_row+5, 8:17, 2] = color
                    start_row = start_row - 5 
                    color = color - 15

                return pad_plane


            def red_range(pad_plane, rows):

                    start_row = 35
                    color = 250

                    for i in range(rows):
                        pad_plane[start_row:start_row+5, 8:17, 0] = color
                        pad_plane[start_row:start_row+5, 8:17, 1] = 50
                        pad_plane[start_row:start_row+5, 8:17, 2] = 50
                        start_row = start_row - 5 
                        color = color - 15

                    return pad_plane

            # Calculate the energy in MeV
            energy_mev = self.to_MeV(tot_energy)
            # Calculate the proportion of the energy bar that should be filled
            proportion_filled = energy_mev / 3

            # Calculate how many rows should be filled
            total_rows = math.floor(proportion_filled * 28)

            # Fill the energy bar one row at a time
            if total_rows > 0:
                pad_plane = blue_range(pad_plane, rows=min(total_rows, 7))
            if total_rows > 7:
                pad_plane = yellow_range(pad_plane, rows=min(total_rows-7, 7))
            if total_rows > 14:
                pad_plane = orange_range(pad_plane, rows=min(total_rows-14, 7))
            if total_rows > 21:
                pad_plane = red_range(pad_plane, rows=min(total_rows-21, 7))

            return pad_plane


        def pt_shift(xset, yset): #TODO: is this a faithful representations?
            """
            Shifts all points to the center of nearest pad for pad mapping
            """
            
            def pos_odd_even(event_value):
                """
                Makes correction to positive points if they are odd or even
                """
                if event_value % 2 == 0:
                    event_value = event_value + 1
                    return event_value

                else:
                    return event_value


            def neg_odd_even(event_value):
                """
                Makes correction to negative points if they are odd or even
                """
                if event_value % 2 == 0:
                    event_value = event_value - 1
                    return event_value

                else:
                    return event_value

            for j in range(len(xset)):

                if xset[j] > 0:
                    xset[j] = math.floor(xset[j])
                    pos_adj_valx = pos_odd_even(xset[j])
                    xset[j] = pos_adj_valx

                elif xset[j] < 0:
                    xset[j] = math.ceil(xset[j])
                    neg_adj_valx = neg_odd_even(xset[j])
                    xset[j] = neg_adj_valx

                if yset[j] > 0:
                    yset[j] = math.floor(yset[j])
                    pos_adj_valy = pos_odd_even(yset[j])
                    yset[j] = pos_adj_valy

                elif yset[j] < 0:
                    yset[j] = math.ceil(yset[j])
                    neg_adj_valy = neg_odd_even(yset[j])
                    yset[j] = neg_adj_valy

            return xset, yset

        
        file = self.h5_file

        xHit, yHit, zHit, eHit = file.get_xyze(event_num,threshold=self.image_hit_threshold,include_veto_pads=False)
        energy = np.sum(eHit)
        pads,pad_data = file.get_pad_traces(event_num)
        pads = np.array(pads)
        pad_data = np.array(pad_data)
        is_not_veto = ~np.isin(pads, VETO_PADS)
        pad_data = pad_data[is_not_veto]
        trace = np.sum(pad_data, axis=0)
        max_val = np.argmax(trace)
        low_bound = max_val - 75
        if low_bound < 0:
            low_bound = 5
        upper_bound = max_val + 75
        if upper_bound > 512:
            upper_bound = 506
        trace = trace[low_bound:upper_bound]
        
        if smoothen:
            trace = smooth_trace(trace)
        trace = remove_noise(trace)

        mm_grid = make_grid()
        pad_plane = np.repeat(mm_grid[np.newaxis, :, :], 1, axis=0)
        new_pad_plane = np.repeat(mm_grid[np.newaxis, :, :], 1, axis=0)

            
        # Call pt_shift function to move all 2D pts to pad centers
        dset_0_copyx, dset_0_copyy = pt_shift(xHit, yHit)
            
        # Call fill_padplane to create 2D pad plane image
        pad_plane = np.append(pad_plane, new_pad_plane, axis=0)
        pad_plane[0] = fill_padplane(dset_0_copyx, dset_0_copyy, eHit, energy)
        
        # Call trace_image() to append trace to pad plane image
        complete_image = (trace_image(pad_plane[0], trace))

        title = "Particle Track"
        plt.rcParams['figure.figsize'] = [7, 7]
        plt.title(f' Image {event_num} of {title} Event:', fontdict = {'fontsize' : 20})
        plt.tick_params(top=False, bottom=False, left=False, right=False, labelleft=False, labelbottom=False)
        plt.imshow(complete_image)
        if save_path != None:
            plt.savefig(save_path)
        if show:
            plt.show(block=False)
        else:
            plt.close()

    def save_cutImages(self, cut_indices):
        '''
        Build the pad-plane image and trace of every event in cut_indices
        (indices into the per-event arrays, e.g. from get_RvE_cut_indexes).
        Returns a list of (pad_plane, trace, title, filename) tuples.
        '''

        def make_box(mm_grid):
            """
            Draws the box for the energy bar
            """
            box_row = np.array([4, 4])
            to_box_row = np.array([145, 146])
            for_box_col = np.array([7, 17])

            box_col = np.array([7, 7])
            to_box_col = np.array([17, 17])
            for_box_row = np.array([4, 145])

            # Draw vertical lines of energy bar box
            for i in range(len(box_row)):
                mm_grid[box_row[i]:to_box_row[i], for_box_col[i], :] = 0
                mm_grid[for_box_row[i], box_col[i]:to_box_col[i], :] = 0

            return mm_grid


        def make_grid():
            """
            "Create Training Data.ipynb"eate grid matrix of MM outline and energy bar, see spreadsheet below
            https://docs.google.com/spreadsheets/d/1_bbg6svfEph_g_Z002rmzTLu8yjQzuj_p50wqs7mMrI/edit?usp=sharing
            """
            row = np.array([63, 47, 39, 31, 27, 23, 19, 15, 15, 11, 11, 7, 7, 7, 7, 3, 3, 3, 3, 3, 3, 3, 7, 7, 
                    7, 7, 11, 11, 15, 15, 19, 23, 27, 31, 39, 47, 63]) 

            to_row = np.array([87, 103, 111, 119, 123, 127, 131, 135, 135, 139, 139, 143, 143, 143, 143, 147, 
                        147, 147, 147, 147, 147, 148, 143, 143, 143, 144, 139, 140, 135, 136, 132, 128,
                        124, 120, 112, 104, 88]) 

            col = np.array([100, 84, 76, 68, 64, 60, 56, 52, 52, 48, 48, 44, 44, 44, 44, 40, 40, 40, 40, 40, 
                    40, 40, 44, 44, 44, 44, 48, 48, 52, 52, 56, 60, 64, 68, 76, 84, 100])

            to_col = np.array([124, 140, 148, 156, 160, 164, 168, 172, 172, 176, 176, 180, 180, 180, 180, 184, 
                        184, 184, 184, 184, 184, 184, 180, 180, 180, 180, 176, 176, 172, 172, 168, 164, 
                        160, 156, 148, 140, 124]) 

            all_row = np.array([i for i in range(3, 148, 4)])
            all_col = np.array([i for i in range(40, 185, 4)]) 

            full_image_size_width = 224
            full_image_size_length = 151
            mm_grid = np.zeros((full_image_size_length, full_image_size_width, 3))    
            mm_grid.fill(255)                                                     

            for i in range(len(row)):
                # draw grid columns, 0 = black
                mm_grid[row[i]:to_row[i], all_col[i], :] = 0
        
            for i in range(len(col)):
                # draw grid rows
                mm_grid[all_row[i], col[i]:to_col[i], :] = 0

            # Ensure that matrix is of integers
            mm_grid = mm_grid.astype(int) 

            # Draw engergy bar box
            mm_grid = make_box(mm_grid)

            return mm_grid

        # Precompute the grid once and reuse it
        global_grid = make_grid()


        def blue_range(pad_plane, rows):

            start_row = 140
            low_color = 0
            high_color = 35

            for i in range(rows):
                pad_plane[start_row:start_row+5, 8:17, 0] = low_color
                pad_plane[start_row:start_row+5, 8:17, 1] = high_color
                start_row = start_row - 5 
                low_color = low_color + 35
                high_color = high_color + 35

            return pad_plane


        def yellow_range(pad_plane, rows):

            start_row = 105
            color = 220

            for i in range(rows):
                pad_plane[start_row:start_row+5, 8:17, 2] = color
                start_row = start_row - 5 
                color = color - 15

            return pad_plane


        def orange_range(pad_plane, rows):

            start_row = 70
            color = 210
            for i in range(rows):
                pad_plane[start_row:start_row+5, 8:17, 1] = color - 15
                pad_plane[start_row:start_row+5, 8:17, 2] = color
                start_row = start_row - 5 
                color = color - 15

            return pad_plane


        def red_range(pad_plane, rows):

            start_row = 35
            color = 250

            for i in range(rows):
                pad_plane[start_row:start_row+5, 8:17, 0] = color
                pad_plane[start_row:start_row+5, 8:17, 1] = 50
                pad_plane[start_row:start_row+5, 8:17, 2] = 50
                start_row = start_row - 5 
                color = color - 15

            return pad_plane


        def tot_energy_to_mev(tot_energy):
            """Convert tot_energy to energy in MeV using the line of best fit."""
            # Calculate the gradient and y-intercept for the line of best fit
            x1, y1 = 156745, 0.806 # Calibration values
            x2, y2 = 320842, 1.679
            gradient = (y2 - y1) / (x2 - x1)
            y_intercept = y1 - gradient * x1
            return gradient * tot_energy + y_intercept


        def fill_energy_bar(pad_plane, tot_energy):
            """
            Fills the energy bar where the amount of pixels fired and the color corresponds to the energy of the track
            Max pixel_range should be 28 (7 rows for each color), so need to adjust accordingly.
            """
            # Calculate the energy in MeV
            energy_mev = tot_energy_to_mev(tot_energy)

            # Calculate the proportion of the energy bar that should be filled
            proportion_filled = energy_mev / 3

            # Calculate how many rows should be filled
            total_rows = math.floor(proportion_filled * 28)

            # Fill the energy bar one row at a time
            if total_rows > 0:
                pad_plane = blue_range(pad_plane, rows=min(total_rows, 7))
            if total_rows > 7:
                pad_plane = yellow_range(pad_plane, rows=min(total_rows-7, 7))
            if total_rows > 14:
                pad_plane = orange_range(pad_plane, rows=min(total_rows-14, 7))
            if total_rows > 21:
                pad_plane = red_range(pad_plane, rows=min(total_rows-21, 7))

            return pad_plane


        def pos_odd_even(event_value):
            """
            Makes correction to positive points if they are odd or even
            """
            if event_value % 2 == 0:
                event_value = event_value + 1
                return event_value

            else:
                return event_value


        def neg_odd_even(event_value):
            """
            Makes correction to negative points if they are odd or even
            """
            if event_value % 2 == 0:
                event_value = event_value - 1
                return event_value

            else:
                return event_value


        def pt_shift(xset, yset):
            """
            Shifts all points to the center of nearest pad for pad mapping
            """
            for j in range(len(xset)):

                if xset[j] > 0:
                    xset[j] = math.floor(xset[j])
                    pos_adj_valx = pos_odd_even(xset[j])
                    xset[j] = pos_adj_valx

                elif xset[j] < 0:
                    xset[j] = math.ceil(xset[j])
                    neg_adj_valx = neg_odd_even(xset[j])
                    xset[j] = neg_adj_valx

                if yset[j] > 0:
                    yset[j] = math.floor(yset[j])
                    pos_adj_valy = pos_odd_even(yset[j])
                    yset[j] = pos_adj_valy

                elif yset[j] < 0:
                    yset[j] = math.ceil(yset[j])
                    neg_adj_valy = neg_odd_even(yset[j])
                    yset[j] = neg_adj_valy

            return xset, yset


        def fill_padplane(xset, yset, eset, tot_energy, global_grid):
            """
            Fills the 2D pad plane grid for image creation
            """
            pad_plane = np.copy(global_grid)

            xset = np.array(xset)
            yset = np.array(yset)
            eset = np.array(eset)

            # pad plane mapping
            x = (35 + xset) * 2 + 42    # col value
            y = 145 - (35 + yset) * 2   # row value

            # create a dictionary to store (x,y) as keys and e as values
            d = {}
            for i in range(len(x)):
                key = (x[i], y[i])
                if key in d:
                    d[key] += eset[i]
                else:
                    d[key] = eset[i]

            # convert the dictionary back to arrays
            x = np.zeros(len(d))
            y = np.zeros(len(d))
            eset = np.zeros(len(d))
            for i, key in enumerate(d):
                x[i] = key[0]
                y[i] = key[1]
                eset[i] = d[key]

            # Find max E value and normalize
            energy = eset
            max_energy = np.max(energy)
            norm_energy = energy / max_energy


            # Fill in pad plane   
            for k in range(len(x)):
            
                if y[k] < 9:
                    y[k] = y[k] + 4

                if x[k] < 50:
                    x[k] = x[k] + 4

                if x[k] > 174:
                    x[k] = x[k] - 4

                if y[k] > 53:
                    y[k] = y[k] - 4

                if x[k] > 134:
                    x[k] = x[k] - 4

                if y[k] > 93:
                    y[k] = y[k] - 4

                if y[k] > 133:
                    y[k] = y[k] - 4	

                if x[k] < 90:
                    x[k] = x[k] + 4


                pad_plane[int(y[k])-1:int(y[k])+2, int(x[k])-1:int(x[k])+2, 0] = norm_energy[k] * 205

                pad_plane[int(y[k])-1:int(y[k])+2, int(x[k])-1:int(x[k])+2, 1] = norm_energy[k] * 240
            
            pad_plane = fill_energy_bar(pad_plane, tot_energy)

            return pad_plane


        def plot_track(cut_indices):
            all_image_data = []  # List to store the results
            pbar = tqdm(total=len(cut_indices))

            for index in cut_indices:
                event_num = self.get_event_num(index)
                file = self.h5_file

                xHit, yHit, zHit, eHit = file.get_xyze(event_num,threshold=self.image_hit_threshold,include_veto_pads=False)
                if len(eHit) == 0:
                    pbar.update(n=1)
                    continue
                energy = np.sum(eHit)
                pads,pad_data = file.get_pad_traces(event_num)
                pads = np.array(pads)
                pad_data = np.array(pad_data)
                is_not_veto = ~np.isin(pads, VETO_PADS)
                pad_data = pad_data[is_not_veto]
                trace = np.sum(pad_data, axis=0)
                max_val = np.argmax(trace)
                low_bound = max_val - 75
                if low_bound < 0:
                    low_bound = 5
                upper_bound = max_val + 75
                if upper_bound > 512:
                    upper_bound = 506
                trace = trace[low_bound:upper_bound]
                trace = smooth_trace(trace)
                trace = remove_noise(trace)

                # Call pt_shift function to move all 2D pts to pad centers
                dset_0_copyx, dset_0_copyy = pt_shift(xHit, yHit)

                # Call fill_padplane to create 2D pad plane image
                pad_plane = fill_padplane(dset_0_copyx, dset_0_copyy, eHit, energy, global_grid)

                # Prepare the data necessary for plotting
                image_title = f' Image {event_num} of Particle Track Event'
                image_filename = f"run{self.run_num}_image_{event_num}.png"

                all_image_data.append((pad_plane, trace, image_title, image_filename))  # Append the result to the list

                pbar.update(n=1)

            return all_image_data  # Return the list of all results after the loop


        result = plot_track(cut_indices)
        return result

    def get_RvE_cut_indexes(self, verticies):
        '''
        verticies: list of (energy, range) tuples defining a cut in RvE.
        Energy in the units of self.counts (self.energy_units), range in mm.
        Returns the indices (into the per-event arrays) of the events inside the cut;
        use get_event_num to turn them into event numbers.
        '''
        selected_rve_path = matplotlib.path.Path(verticies)
        rve_points = np.vstack((self.counts, self.ranges)).transpose()
        self.rve_cut_select_mask = selected_rve_path.contains_points(rve_points)
        print(sum(self.rve_cut_select_mask), 'events selected in cut')
        return np.where(self.rve_cut_select_mask)[0]

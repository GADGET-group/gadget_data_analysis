'''
This file contains the following
-- Class which handles accessing information from a single run
-- A function for generating files subject to veto conditions from h5 files.
-- Functions for getting the defulat directory from run id
'''
import os
import numpy as np
import random
import socket

#imports for reading files
import h5py
import math
from sklearn.decomposition import PCA
from pca import pca
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.path
from tqdm import tqdm
import sys
from BaselineRemoval import BaselineRemoval
from sklearn.cluster import DBSCAN
from scipy.signal import savgol_filter
from skspatial.objects import Line
from raw_viewer.raw_h5_file import raw_h5_file
from raw_viewer.raw_h5_file import raw_h5_file

zscale = 1.45

def run_num_to_str(run_num):
    run_num = int(run_num)
    return  ('%4d'%run_num).replace(' ', '0')

def get_h5_path():
    if socket.gethostname() == 'tpcgpu':
        return "/egr/research-tpc/shared/Run_Data/"
    else:
        return "/mnt/analysis/e21072/h5test/"

def get_default_path(run_id):    
    run_str = run_num_to_str(run_id)
    return get_h5_path() + f'run_{run_str}'
    

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
    def __init__(self, run_num, folder_path):
        self.run_num = run_num
        self.folder_path = folder_path
        self.file_path = get_h5_path() + ('run_%04d.h5'%run_num)
    
        self.h5_file = raw_h5_file(self.file_path, flat_lookup_csv='./raw_viewer/channel_mappings/flatlookup4cobos.csv', zscale=zscale)
        self.h5_file.background_subtract_mode='fixed window'
        self.h5_file.data_select_mode='near peak'
        self.h5_file.remove_outliers=True
        self.h5_file.near_peak_window_width = 50
        self.h5_file.require_peak_within= (-np.inf, np.inf)
        self.h5_file.num_background_bins=(160, 250)
        self.h5_file.zscale = zscale
        self.h5_filelength_counts_threshold = 100
        self.h5_file.ic_counts_threshold = 25
        self.h5_file.include_counts_on_veto_pads = False


        #TODO: decide how to store calibration information with runs
        calib_point_1 = (0.806, 156745)
        calib_point_2 = (1.679, 320842)
        energy_1, channel_1 = calib_point_1
        energy_2, channel_2 = calib_point_2
        self.energy_scale_factor = (energy_2 - energy_1) / (channel_2 - channel_1)
        self.energy_offset = energy_1 - self.energy_scale_factor * channel_1
        
    def to_MeV(self, counts):
        return counts*self.energy_scale_factor + self.energy_offset
    
    def to_counts(self, MeV):
        return (MeV - self.energy_offset)/self.energy_scale_factor

    def get_index(self, event_num):
        '''
        Gets the index at which an event number can be found in the data
        '''
        return event_num

    def get_hit_lists(self, event_num):
        '''
        I think these are x,y,z positions of points in the point cloud, and 
        the energy associated with each point.
        '''
        index = self.get_index(event_num)
        return self.h5_file.get_xyze(index,threshold=20,include_veto_pads=False)

    def make_image(self, index, save_path=None, show=False, smoothen=False):
        '''
        Make datafused image of event at "index". Image will be saved to "save_path"
        if not None. 
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
            rand_num = random.randrange(0,1000000,1)
            temp_strg = f'/egr/research-tpc/shared/temp/energy_depo_{rand_num}.jpg'
            plt.savefig(temp_strg, dpi=my_dpi)
            plt.close()

            # Load png plot as a matrix so that it can be appended to pad plane plot
            img = plt.imread(temp_strg)
            os.remove(temp_strg)
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

        
        VETO_PADS = (253, 254, 508, 509, 763, 764, 1018, 1019)
        file = self.h5_file

        xHit, yHit, zHit, eHit = file.get_xyze(index,threshold=20,include_veto_pads=False)
        energy = np.sum(eHit)
        self.total_energy_MeV = eHit
        pads,pad_data = file.get_pad_traces(index)
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
        plt.title(f' Image {index} of {title} Event:', fontdict = {'fontsize' : 20})
        plt.tick_params(top=False, bottom=False, left=False, right=False, labelleft=False, labelbottom=False)
        str_event_num = f"run{self.run_num}_image_{index}.jpg"
        plt.imshow(complete_image)
        if save_path != None:
            plt.savefig(save_path)
        if show:
            plt.show(block=False)
        else:
            plt.close()

    def save_cutImages(self, cut_indices):

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
            import pickle
            import torch
            all_image_data = []  # List to store the results
            pbar = tqdm(total=len(cut_indices))

            for event_num in cut_indices:
                VETO_PADS = (253, 254, 508, 509, 763, 764, 1018, 1019)
                file = self.h5_file

                xHit, yHit, zHit, eHit = file.get_xyze(event_num,threshold=20,include_veto_pads=False)
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

    def get_RvE_cut_indexes(self, points):
        '''
        points: list of (energy, range) tuples defining a cut in RvE
        Energy is in MeV, range in mm
        '''
        path = matplotlib.path.Path(points)
        to_return = []
        index = 0
        while index < len(self.good_events):
            this_point = (self.total_energy_MeV[index], self.len_list[index])
            if path.contains_point(this_point):
                to_return.append(index)
            index += 1
        return to_return
    


def generate_files(run_num, length, ic, pads, eps, samps, poly):
    run_num = run_num_to_str(run_num)
    #check if files already exist
    mypath = get_default_path(run_num)
    sub_mypath = mypath + f'/len{length}_ic{ic}_pads{pads}_eps{eps}_samps{samps}_poly{poly}'
    if os.path.isdir(mypath):
        if os.path.isdir(sub_mypath):
            # Give option to overwrite or cancel
            print("Files Already Exist.")
            overwrite = int(input('Would you like to overwrite (1=yes, 0=no): '))    
            if overwrite == True:
                print('Overwriting Existing Files')
            else:
                return
            
        else:
            os.makedirs(sub_mypath)
        
    else:
        os.makedirs(mypath)
        os.makedirs(sub_mypath)

    #start coppied from generate_files in The_GADGET_FUI.py
    # In[2]:
    class HiddenPrints:
        def __enter__(self):
            self._original_stdout = sys.stdout
            sys.stdout = open(os.devnull, 'w')

        def __exit__(self, exc_type, exc_val, exc_tb):
            sys.stdout.close()
            sys.stdout = self._original_stdout



    def remove_outliers(xset, yset, zset, eset, pads):
        """
        Uses DBSCAN to find and remove outliers in 3D data
        """

        data = np.array([xset.T, yset.T, zset.T]).T
        DBSCAN_cluster = DBSCAN(eps=eps, min_samples=samps).fit(data)
        del data
    
        if all(element == -1 for element in DBSCAN_cluster.labels_):
            veto = True
        else:
            # Identify largest clusters
            labels = DBSCAN_cluster.labels_
            unique_labels = set(labels)
            if -1 in unique_labels:
                unique_labels.remove(-1)

            # Find the two largest clusters
            largest_clusters = sorted(unique_labels, key=lambda x: np.sum(labels == x), reverse=True)[:2]

            # Relabel non-main-cluster points as outliers
            for cluster_label in unique_labels:
                if cluster_label not in largest_clusters:
                    labels[labels == cluster_label] = -1

            # Remove outlier points
            out_of_cluster_index = np.where(labels == -1)
            rev = out_of_cluster_index[0][::-1]
            for i in rev:
                xset = np.delete(xset, i)
                yset = np.delete(yset, i)
                zset = np.delete(zset, i)
                eset = np.delete(eset, i)

            if len(xset) <= pads:
                veto = True
            else:
                veto = False

        return xset, yset, zset, eset, veto

    def track_len(xset, yset, zset):
        """
        Uses PCA to find the length of a track
        """
        veto_on_length = False

        # Form data matrix
        data = np.concatenate((xset[:, np.newaxis], 
               yset[:, np.newaxis], 
               zset[:, np.newaxis]), 
               axis=1)

        # Use PCA to find track length
        pca = PCA(n_components=2)
        principalComponents = pca.fit(data)
        principalComponents = pca.transform(data)
        principalDf = pd.DataFrame(data = principalComponents
         , columns = ['principal component 1', 'principal component 2'])
        calibration_factor = 1.4 

        # Call track_angle to get the angle of the track
        angle_deg = track_angle(xset, yset, zset)

        # Calculate the scale factor based on the angle
        #scale_factor = get_scale_factor(angle_deg)
        #scale_factor_trace = scale_factor * 1.5 

        # Apply the scale factor to the track length
        # track_len = scale_factor * calibration_factor * 2.35 * principalDf.std()[0]
        track_len = calibration_factor * 2.35 * principalDf.std()[0]

        if track_len > length:
            veto_on_length = True

        return track_len, veto_on_length, angle_deg



    def track_angle(xset, yset, zset):
        """
        Fits 3D track, and determines angle wrt pad plane
        """
        # Form data matrix
        data = np.concatenate((xset[:, np.newaxis], 
                   yset[:, np.newaxis], 
                   zset[:, np.newaxis]), 
                   axis=1)

        # Fit regression line
        line_fit = Line.best_fit(data)

        # Find angle between the vector of the fit line and a vector normal to the xy-plane (pad plane)
        v = np.array([line_fit.vector]).T   # fit line vector
        n = np.array(([[0, 0, 1]])).T       # Vector normal to xy-plane
        dot = np.dot(n.T, v)[0][0]          # Note that both vectors already have a magnitude of 1

        # Clamp the dot variable to be within the valid range
        dot = max(-1.0, min(1.0, dot))

        theta = math.acos(dot)
        track_angle_rad = (math.pi/2 - theta) 
        track_angle_deg = track_angle_rad * (180 / np.pi)

        # Angle should always be less than 90 deg
        if track_angle_deg < 0:
            track_angle_deg = 180 + track_angle_deg 
        if track_angle_deg > 90:
            track_angle_deg = 180 - track_angle_deg

        return track_angle_deg

    def get_scale_factor(angle, angle_min=40, angle_max=90, scale_min=1, scale_max=1.3):
        if angle < angle_min:
            return scale_min
        elif angle > angle_max:
            return scale_max
        else:
            return scale_min + (scale_max - scale_min) * (angle - angle_min) / (angle_max - angle_min)
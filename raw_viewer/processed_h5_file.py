import h5py

import raw_h5_file

class processed_h5_file(raw_h5_file.raw_h5_file):
    def __init__(self, raw_file, results_file):
        '''
        raw_file: File containing results of merging GRAW files
        results_file: place to store results of processing raw_file
        '''
        super.__init__(raw_file)
        self.results_file = h5py.File(results_file, 'a')

    def get_settings_string(self):
        '''
        Returns a tuple representing the settings used to calculate veto, range, etc, of the super class
        '''
        return 'background=%d-%d, angle=%f-%f, ic bounds=%f-%f, range=%f-%f, background subtract=%d, outlier removal=%d, length, ic, veto threshold=%f, %f, %f' \
               %(self.num_background_bins[0], self.num_background_bins[1], 
                  self.angle_bounds[0], self.angle_bounds[1], 
                    self.ic_bounds[0], self.ic_bounds[1],
                    self.range_bounds[0], self.range_bounds[1],
                    self.apply_background_subtraction, self.remove_outliers,
                    self.length_counts_threshold, self.ic_counts_threshold,
                    self.veto_threshold
                )
    
    def process_all_events(self):
        pass

import raw_h5_file
import numpy as np
import matplotlib.pylab as plt
import matplotlib.colors as colors
'''
from importlib import reload
reload(raw_h5_file)
#file = raw_h5_file.raw_h5_file('/mnt/analysis/e21072/gastest_h5_files/run_0099.h5')
file = raw_h5_file.raw_h5_file('E:/run_0328.h5')
file.veto_threshold=500
file.apply_background_subtraction=True
file.remove_outliers=True
file.range_bounds=(0, np.inf)
file.ic_bounds=(0, np.inf)
file.num_background_bins=(400,500)
file.length_counts_threshold = 350
file.ic_counts_threshold = 20
file.zscale=1.45
ranges, counts, angles = file.get_histogram_arrays()
plt.hist2d(np.degrees(angles), counts, 100, norm=colors.LogNorm())
plt.xlabel('track angle (degrees)')
plt.ylabel('integrated charge (counts)')

np.save('range_run_328_1.npy', ranges)
np.save('angles_run_328_1.npy', angles)
np.save('counts_run_328_1.npy', counts)
'''

'''
#file = raw_h5_file.raw_h5_file('/mnt/analysis/e21072/gastest_h5_files/run_0099.h5')
#file = raw_h5_file.raw_h5_file('E:/gas_test_h5/gastestsrun_0099.h5')
file = raw_h5_file.raw_h5_file('C:/temp/run_0099.h5')
file.veto_threshold=500
file.apply_background_subtraction=True
file.remove_outliers=True
file.range_bounds=(1, np.inf)
file.ic_bounds=(1, np.inf)
file.num_background_bins=(400,500)
file.length_counts_threshold = 60
file.ic_counts_threshold = 20
file.zscale=1.45
ranges, counts, angles = file.get_histogram_arrays()
plt.hist2d(np.degrees(angles), counts, 100, norm=colors.LogNorm())
plt.xlabel('track angle (degrees)')
plt.ylabel('integrated charge (counts)')
'''


'''
file = raw_h5_file.raw_h5_file('E:/gas_test_h5/run_0088.h5')
file.veto_threshold=100
file.apply_background_subtraction=True
file.remove_outliers=True
file.range_bounds=(1, np.inf)
file.ic_bounds=(1, np.inf)
file.num_background_bins=(0,100)
file.zscale=0.25
file.length_counts_threshold = 50
file.ic_counts_threshold = 2
ranges_co2, counts_co2, angles_co2 = file.get_histogram_arrays()
plt.hist2d(np.degrees(angles_co2), counts_co2, 100, norm=colors.LogNorm())
plt.xlabel('track angle (degrees)')
plt.ylabel('integrated charge (counts)')
'''

'''
file = h5py.File('output_220Rn_alpha_12Nov_13500events.h5', 'r')

'''

'''
file = raw_h5_file.raw_h5_file('E:/gastestsrun_0127.h5')
file.veto_threshold=500
file.apply_background_subtraction=True
file.remove_outliers=False
file.range_bounds=(0, np.inf)
file.ic_bounds=(-np.inf, np.inf)
file.num_background_bins=(0,150)
file.zscale=1.45
file.length_counts_threshold = 50
file.ic_counts_threshold = -1000000
ranges, counts, angles = file.get_histogram_arrays()
'''

'''
#file = raw_h5_file.raw_h5_file('E:/gastestsrun_0127.h5')
file = raw_h5_file.raw_h5_file('C:/temp/gastestsrun_0127.h5')
file.veto_threshold=np.inf
file.apply_background_subtraction=True
file.remove_outliers=True
file.mode = 'near peak'
file.require_peak_within = (150,300)
file.near_peak_window_width = 50
file.range_bounds=(0, np.inf)
file.ic_bounds=(-np.inf, np.inf)
file.num_background_bins=(400,500)
file.zscale=1.45
file.length_counts_threshold = 100
file.ic_counts_threshold = 15
file.include_counts_on_veto_pads = True
ranges, counts, angles = file.get_histogram_arrays()
plt.hist(counts[counts>1e4], 100)'''
'''
file = raw_h5_file.raw_h5_file('E:/gastestsrun_0134.h5')
'''
'''
for run in range(163, 173):
    try:
        ranges = np.load('D:/raw_viewer/%d_ranges.npy'%run)
        counts = np.load('D:/raw_viewer/%d_counts.npy'%run)
        angles = np.load('D:/raw_viewer/%d_angles.npy'%run)
        
        mask = counts>1e5#np.logical_and(counts>1e5, angles>np.radians(20))
        plt.figure()
        plt.hist2d(counts[mask], ranges[mask],100)
        plt.colorbar()
        plt.xlabel('adc counts')
        plt.ylabel('range (mm)')
        plt.title('run %d'%run)
        plt.savefig('run_%d_RvE.png'%run)
    except:
        print('couldn\'t process run %d'%run)
'''      

'''
for run in range(163, 173):
    try:
        ranges = np.load('D:/raw_viewer/%d_ranges.npy'%run)
        counts = np.load('D:/raw_viewer/%d_counts.npy'%run)
        angles = np.load('D:/raw_viewer/%d_angles.npy'%run)
        
        plt.figure()
        plt.hist(counts,100)
        plt.xlabel('adc counts')
        plt.ylabel('# events')
        plt.title('run %d'%run)
        plt.savefig('run_%d_counts_hist.png'%run)
    except:
        print('couldn\'t process run %d'%run)
'''
'''
for run in range(274, 275):
    local=True
    if local:
        file=raw_h5_file.raw_h5_file('C:/temp/run_%04d.h5'%run, 
                                   flat_lookup_csv='flatlookup4cobos.csv')
    else:
        file = raw_h5_file.raw_h5_file('/mnt/analysis/e21072/h5test/run_%04d.h5'%run, 
                                   flat_lookup_csv='flatlookup4cobos.csv')
    file.veto_threshold=1500
    file.apply_background_subtraction=True
    file.remove_outliers=True
    file.mode = 'near peak'
    file.require_peak_within = (20,180)
    file.near_peak_window_width = 50
    file.range_bounds=(0, np.inf)
    file.ic_bounds=(-np.inf, np.inf)
    file.num_background_bins=(200,400)
    file.zscale=1.45
    file.length_counts_threshold = 100
    file.ic_counts_threshold = 75
    file.include_counts_on_veto_pads = False
    ranges, counts, angles = file.get_histogram_arrays()
    np.save('e21072_run%04d_ranges_veto_1500'%run, ranges)
    np.save('e21072_run%04d_counts_veto_1500'%run, counts)
    np.save('e21072_run%04d_angles_veto_1500'%run, angles)
'''
ranges = np.load('I:/projects/e21072/OfflineAnalysis/analysis_scripts/alex/gadget_analysis/raw_viewer/e21072_run274_ranges_veto_1500.npy')
counts = np.load('I:/projects/e21072/OfflineAnalysis/analysis_scripts/alex/gadget_analysis/raw_viewer/e21072_run274_counts_veto_1500.npy')
angles = np.load('I:/projects/e21072/OfflineAnalysis/analysis_scripts/alex/gadget_analysis/raw_viewer/e21072_run274_angles_veto_1500.npy')

plt.figure()
plt.hist(counts,1000)
plt.xlabel('adc counts')
plt.ylabel('# events')
plt.title('run %d'%274)

events_in_peak=np.where(np.logical_and(counts>1.4e5,counts<1.5e5))
pc_events=np.load('C:/temp/run_274_len90_ic600000_pads5_eps5_samps5_poly2/good_events.npy')

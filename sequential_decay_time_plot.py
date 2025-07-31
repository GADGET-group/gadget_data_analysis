import numpy as np
import h5py
import scipy
import pandas as pd
import raw_viewer.raw_h5_file as raw_h5_file
import matplotlib.pyplot as plt

event_type = 'RnPo Chain'  # Change to 'Accidental Coin' to look at random events

categorized_events_of_interest = pd.read_csv('./complete_categorized_events_of_interest.csv',encoding='utf-8-sig', skip_blank_lines = False, nrows = 36164, header=None)
array_of_categorized_events_of_interest = categorized_events_of_interest[0].to_numpy()

manual_time_diff = pd.read_csv('./manual_rnpo_time_diff.csv',encoding='utf-8-sig', skip_blank_lines = True, header=None)
manual_time_diff = manual_time_diff.to_numpy() * 20
print("Number of manual rnpo time differences:", len(manual_time_diff))
bin_heights, bin_borders, _ = plt.hist(manual_time_diff, bins=64, density=False, range=(0, 10240), color = 'yellow')
bin_centers = (bin_borders[:-1] + bin_borders[1:]) / 2
# plt.show()

def exponential_fit(x, a, b, c):
    return a * np.exp(-b * x) + c

popt, pcov = scipy.optimize.curve_fit(exponential_fit, bin_centers, bin_heights, p0=[45, 0.00031, 0.3], bounds=([-np.inf,-np.inf,0],[np.inf,np.inf, np.inf]))
x_fit = np.linspace(min(bin_borders), max(bin_borders), 500)
y_fit = exponential_fit(x_fit, *popt)

print(popt)
plt.clf()
plt.plot(x_fit, y_fit, color='red', label='Exponential Fit on Manual RnPo Chain Time Differences')
plt.hist(manual_time_diff, bins=64, range=(0, 10240), label='Manual RnPo Chain Time Differences', color='skyblue')
plt.title('Manual RnPo Chain Time Differences')
plt.xlabel('Time difference (ns)')
plt.ylabel('Counts')
# plt.show()

zscale = 0.65

# length = []
# total_counts = np.load('/egr/research-tpc/shared/Run_Data/run_%04d_raw_viewer/run_%04dsmart/counts.npy'%(run,run))
# total_veto = np.load('/egr/research-tpc/shared/Run_Data/run_%04d_raw_viewer/run_%04dsmart/veto.npy'%(run,run))
# dxy = np.load('/egr/research-tpc/shared/Run_Data/run_%04d_raw_viewer/run_%04dsmart/dxy.npy'%(run,run))
# dt = np.load('/egr/research-tpc/shared/Run_Data/run_%04d_raw_viewer/run_%04dsmart/dt.npy'%(run,run))
# for event in range(len(dxy)):
#     length.append((dxy[event]**2 + zscale*dt[event]**2)**0.5)
# total_ranges = length
    
h5file = raw_h5_file.raw_h5_file('/egr/research-tpc/dopferjo/interesting_events_without_run_number_in_event_name_without_event_447.h5', 
                                zscale = zscale, 
                                flat_lookup_csv = "/egr/research-tpc/dopferjo/gadget_analysis/raw_viewer/channel_mappings/flatlookup2cobos.csv")

f = h5py.File('/egr/research-tpc/dopferjo/interesting_events_without_run_number_in_event_name_without_event_447.h5', 'r')
first_event, last_event = int(f['meta']['meta'][0]), int(f['meta']['meta'][2])

print("First event: %d, Last event: %d"%(first_event, last_event))

# process settings are set the same as smart config settings
h5file.length_ic_threshold = 100
h5file.ic_counts_threshold = 9
h5file.view_threshold = 100
h5file.include_cobos = all
h5file.include_asads = all
h5file.include_pads = all
h5file.veto_threshold = 300
h5file.range_min = 1
h5file.range_max = np.inf
h5file.min_ic = 1
h5file.max_ic = np.inf
h5file.angle_min = 0
h5file.angle_max = 90
h5file.background_bin_start = 160
h5file.background_bin_stop = 250
h5file.zscale = .65
h5file.background_start_entry = 160
h5file.background_stop_entry = 250
h5file.exclude_width_entry = 20
h5file.include_width_entry = 40
h5file.near_peak_window_entry = 50
h5file.near_peak_window_width = 50
h5file.peak_first_allowed_bin_entry = -np.inf
h5file.peak_last_allowed_bin_entry = np.inf
h5file.peak_first_allowed_bin = -np.inf
h5file.peak_last_allowed_bin = np.inf
h5file.peak_mode = 'all data'
h5file.background_subtract_mode = 'none'
h5file.data_select_mode = 'all data'
h5file.remove_outliers = 1
h5file.num_background_bins = (160,250)

time_diffs = []
time_diff_diffs = []
counter = 0
for event_number in range(len(array_of_categorized_events_of_interest)):
    if array_of_categorized_events_of_interest[event_number] == event_type: # This can be be changed to 'Accidental Coin' to look at how the time difference of random events should looks
        counter += 1
        print(event_type, ' Event number:', event_number)
        # Find time difference between the two alphas
        pads, traces = h5file.get_pad_traces(event_number, include_veto_pads=False)
        # Print out all pad numbers that fired
        # print("Pads that fired:", pads)

        # For each pad, find two peaks and print their centroids
        pad_time_diff = []
        pad_time_diff_sum = []
        if event_type == 'Accidental Coin':
            # I am using the summed trace for accidental coincidences
            # This is because the two alphas are not necessarily on the same pads
            summed_trace = np.sum(traces, axis=0)
            peaks, properties = scipy.signal.find_peaks(summed_trace, width = (8,500), prominence=(7000, np.inf))
            peaks = peaks[properties['prominences'] > 7000]
            if len(peaks) >= 2:
                pad_time_diff_sum.append(np.abs(peaks[0] - peaks[-1]))
                #TODO: Finish implementing the accidental coincidence logic, and change hard coded 'RnPo Chain' and 'Accidental Coin' to event_type

        if event_type == 'RnPo Chain':
            for trace in traces:
                peaks, properties = scipy.signal.find_peaks(trace, width = (5,500), prominence=(40, 4000))
                # print("Peaks: ",peaks)
                # print("Prominences: ", properties['prominences'])
                # print("Widths: ", properties['widths'])
                # print(properties['prominences'] > 40)
                peaks = peaks[properties['prominences'] > 140]
                if len(peaks) >= 2:
                    if event_number == 23946:
                        # plot the trace with the prominences
                        contour_heights = trace[peaks] - properties['prominences']
                        plt.plot(trace)
                        plt.plot(peaks, trace[peaks], "x")
                        plt.vlines(x=peaks, ymin=contour_heights, ymax=trace[peaks], colors='orange', linestyles='dashed')
                        plt.show()
                    pad_time_diff.append(np.abs(peaks[0] - peaks[1]))
                    
            summed_trace = np.sum(traces, axis=0)
            peaks, properties = scipy.signal.find_peaks(summed_trace, width = (8,500), prominence=(7000, np.inf))
            peaks = peaks[properties['prominences'] > 7000]
            if len(peaks) >= 2:
                pad_time_diff_sum = np.abs(peaks[0] - peaks[-1])
            if not pad_time_diff_sum:
                pad_time_diff_sum = np.nan
            # Plot a histogram of the time differences

            # break
            
            # if len(pad_time_diff) < 3:
            #     print("Sketchy event number:", event_number)
            #     print(np.mean(pad_time_diff))

        time_diffs.append(np.mean(pad_time_diff))
        time_diff_diffs.append(np.float64(pad_time_diff_sum) - np.mean(pad_time_diff) )

print("Number of %s events processed: %d"%(event_type, counter))
# print("Time differences (in bins):", time_diffs)
# print("Time differences (in ns):", np.array(time_diffs) * 20)  # Assuming 20 ns per bin
time_diffs = np.array(time_diffs)
time_diff_diffs = np.array(time_diff_diffs)

# time_diffs = time_diffs[~np.isnan(time_diffs)]
plt.hist(np.array(time_diffs) * 20, bins=64, range=(0, 10240), label='Calculated Time Differences', alpha=0.6, color='salmon')
plt.xlabel('Time difference (ns)')
plt.ylabel('Counts')
plt.legend()
plt.show()

plt.clf()
plt.hist(np.array(time_diff_diffs) * 20, bins = 100)
plt.xlabel('Time difference of summed trace - Time difference average (ns)')
plt.ylabel('Counts')
plt.yscale('log')
plt.show()
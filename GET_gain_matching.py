# import numpy as np
# import scipy.signal as sig
# import matplotlib.pyplot as plt
# import matplotlib as mpl
# from tqdm import tqdm
# from scipy.optimize import curve_fit
# # removed run 367 because it has 2 events in it and thus we cannot easily find a peak to gain match our data
# # it is also missing runs 46, 76 because it is too small a run for the peak fitter to find the peak properly (TODO: fix the peak fitter smoother so the prominence can be lowered)
# production_runs = [6,7,8,10,11,12,13,14,15,16,17,31,32,33,34,35,36,37,38,39,40,41,42,43,44,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,73,74,75,77,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98,99,100,101,102,103,104,105,106,107,108,126,133,134,150,151,152,153,154,155,156,157,158,159,160,161,164,165,166,167,168,169,170,171,172,173,174,175,176,177,178,179,180,181,182,183,184,185,186,187,188,189,190,191,192,193,194,195,196,197,198,199,200,201,202,203,204,205,206,207,208,209,210,211,212,213,214,215,216,217,218,219,220,221,222,223,224,225,226,227,228,229,230,231,232,233,234,235,236,237,238,239,240,241,242,243,245,246,247,248,249,250,251,252,253,254,255,256,257,258,259,260,261,262,263,264,265,266,267,268,269,270,271,272,273,274,275,276,278,279,280,281,282,283,284,285,286,287,288,289,290,291,292,293,294,295,296,297,298,299,300,301,302,304,305,306,307,308,309,310,311,312,313,314,315,316,317,318,319,320,321,322,323,324,325,326,327,328,329,330,331,333,334,335,336,337,338,339,340,341,342,343,344,345,346,347,348,349,350,351,352,353,354,355,356,357,358,359,360,361,362,363,364,365,366,368,369,370,371,372,373,374,375,376,377,378,379,380,381,382,383,384,385,386,387,388,389,390,391,392,393,394,395,396,397,398,399,400,401,402,403,404,405,406,407,408,409,410,411,412,413,414,415,416,417,418,419,420,421,422,423,424,425,426,427,428,429,430,431,432,433,434,435,436,437,438,439,440,441,442,443,444,445,446,447,448,449,450,451,452,453,454,455,456,457,458,459,460,461,462,463,464,465,466,467,468,469,470,471,472,473,474,475,476,477,478,479,480,481,482,483,484,485,486,487,488,489,490,491,492,493,494,495,496,497,498,499,500,501,502,503,504,505,506,507,508,509,510,511,512,513,514,515,516,517,518,519,520,521,522,523,524,527,528,529,530,531,532,533,534,535,536,537,538,539,540,541,542,543,544,545,546,547,548,549,550,551,552,553,554,555]

# # production_runs = [6,7,8,10,11,12,13,14,15,16,17,31,32,33,34,35,36,37,38,39,40,41,42,43,44,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,73,74,75,77,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98,99,100,101,102,103,104,105,106,107,108,126,133,134,150,151,152,153,154,155,156,157,158]
# # production_runs = [55]
# total_counts = []
# total_ranges = []
# total_veto =[]
# dxy = []
# dt = []

# zscale = 0.65

# # Parameters used to find gain matching scaler
# # bins = 4000 # bins in the energy histogram; this is the good one for now
# bins = 2000
# energy_low = 0
# energy_high = 1e6
# length_low = 0
# length_high = 200
# smoothing_factor = 10 # averaging window size to ensure proper peak search results
# smoothing_factor_length = 8
# results = {"run": [], "counts_peak_loc": [], "range_peak_loc": [], "counts_gain_factor": [], "range_gain_factor": []}

# for run in production_runs:
#     if run > 5:
#         length = []
#         counts =[]
#         veto = []
#         dxy = []
#         dt = []

#         counts = np.load('/mnt/daqtesting/protondet2024/h5/run_%04d/run_%04dp10_2000torr/counts.npy'%(run,run))
#         veto = np.load('/mnt/daqtesting/protondet2024/h5/run_%04d/run_%04dp10_2000torr/veto.npy'%(run,run))

#         dxy = np.load('/mnt/daqtesting/protondet2024/h5/run_%04d/run_%04dp10_2000torr/dxy.npy'%(run,run))
#         dt = np.load('/mnt/daqtesting/protondet2024/h5/run_%04d/run_%04dp10_2000torr/dt.npy'%(run,run))
#         for event in range(len(dxy)):
#             length.append((dxy[event]**2 + zscale*dt[event]**2)**0.5)
#         length = np.array(length)

#         print('Events in run %d: '%run,len(counts))
#         counts_mask = np.logical_and.reduce((counts<9e5,
#                                     counts>5.5e5,
#                                     length>34))

#         #  Histogram the projection of energies
#         hist, bin_edges = np.histogram(counts[counts_mask], bins=bins, range=(energy_low, energy_high))
#         bin_width = (energy_high - energy_low) / bins
#         bin_centers = (bin_edges[:-1] + bin_edges[1:])/2
#         # length_mask = np.logical_and.reduce((length>32,
#         #                                      length<55))
#         # Now we do the same thing for ranges
#         hist_length, bin_edges_length = np.histogram(length[counts_mask], bins=bins, range=(length_low, length_high))
#         bin_width_length = (length_high - length_low) / bins
#         bin_centers_length = (bin_edges_length[:-1] + bin_edges_length[1:])/2

#         # # Perform a moving average smoothing for both energy and ranges
#         # window = np.arange(0, bins, 1)
#         # window_centered = window - (window[-1] + window[0]) / 2
#         # fil = np.fft.ifftshift(np.sinc(window_centered / smoothing_factor))  # Size of points taken for average is denominator
#         # transformed = np.fft.fft2(hist, axes=(0,))
#         # hist_smoothed = np.real(np.fft.ifft2(transformed * fil, axes=(0,)))
#         # plt.hist(counts[counts_mask], bins=200)
#         # plt.show()
#         # fil_length = np.fft.ifftshift(np.sinc(window_centered / smoothing_factor_length))
#         # transformed_length = np.fft.fft2(hist_length, axes=(0,))
#         # hist_length_smoothed = np.real(np.fft.ifft2(transformed_length * fil_length, axes=(0,)))
#         # plt.hist(hist_length_smoothed)
#         # plt.show()
#         # # Find largest peak in smoothed histograms
#         # pks, props = sig.find_peaks(hist_smoothed, distance=1, prominence=10, width=7, rel_height=0.95)
#         # pks_length, props_length = sig.find_peaks(hist_length_smoothed, distance=1, prominence=10, width=10, rel_height=0.95)
    
#         # Now we have the gated histogram, fit the peak and use that mean to gain match each run
#         def gaussian(x, height, mean, sigma):
#             return height * np.exp(-((x-mean) / (2*sigma))**2)
#         # plt.scatter(bin_centers,hist)
#         # plt.show()
#         guesses = [300,7e5,0.3e5]
#         popt, pcov = curve_fit(gaussian, bin_centers, hist,p0=guesses)
#         print(popt)
#         # plt.scatter(bin_centers_length,hist_length)
#         # plt.show()
#         guesses_length = [300,35,1]
#         popt_length, pcov_length = curve_fit(gaussian, bin_centers_length, hist_length, p0=guesses_length)
#         print(popt_length)
#         # plt.scatter(bin_centers_length,hist_length)
#         # plt.show()
#         max_peak_centroid = popt[1]
#         max_length_centroid = popt_length[1]
#         results["run"].append(run)
#         results["counts_gain_factor"].append(max_peak_centroid)
#         results["range_gain_factor"].append(max_length_centroid)

#         # print("Peaks in energy in run %d: "%run,pks)
#         # print("Peaks in range in run %d: "%run,pks_length)
#         # # plot 1d hist of ranges
#         # fig,ax = plt.subplots()
#         # ax.bar(bin_edges_length[:-1],hist_length,width=np.diff(bin_edges_length),edgecolor="black",align="edge")
#         # # plt.hist(total_counts, bins=bins)
#         # plt.plot(pks_length * bin_width_length + length_low, hist_length_smoothed[pks_length], "rx")
#         # plt.show()
#         total_counts = np.concatenate([total_counts,np.multiply(counts,644000/max_peak_centroid)])
#         total_ranges = np.concatenate([total_ranges,np.multiply(length,36.547/max_length_centroid)])
#         total_veto = np.concatenate([total_veto,veto])

# print(results)
# # fig,ax = plt.subplots()
# # ax.bar(bin_edges[:-1],hist_smoothed,width=np.diff(bin_edges),edgecolor="black",align="edge")
# # # plt.hist(total_counts, bins=bins)
# # plt.plot(pks * bin_width + energy_low, hist_smoothed[pks], "rx")
# # plt.show()

# # fig,ax = plt.subplots()
# # ax.bar(bin_edges_length[:-1],hist_length_smoothed,width=np.diff(bin_edges_length),edgecolor="black",align="edge")
# # # plt.hist(total_counts, bins=bins)
# # plt.plot(pks_length * bin_width_length + length_low, hist_length_smoothed[pks_length], "rx")
# # plt.show()

# plt.figure(0)
# plt.title('RvE for All Runs', fontsize=32)
# plt.hist2d(total_counts, total_ranges, 200, norm=mpl.colors.LogNorm(), range=[[0,2e6],[0,200]])
# # plt.hist(total_counts, bins=200)
# # plt.colorbar(labelsize=24)
# plt.colorbar()
# plt.axvline(x = 8.65e5, color = 'r', linestyle = 'solid')
# plt.xlabel('Energy (arb. units)', fontsize=24)
# plt.ylabel('Range (mm)', fontsize=24)
# plt.xticks(fontsize=24)
# plt.yticks(fontsize=24)
# plt.show()

# plt.title("Energy Histogram for All Runs")
# plt.hist(total_counts, 1000, range=[0,2e6])
# plt.xlabel('Energy (arb. units)')
# plt.show()

import numpy as np
import scipy.signal as sig
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import LogNorm
from scipy.optimize import curve_fit
from tqdm import tqdm

# config
zscale = 0.65
bins = 2000
energy_low, energy_high = 0, 1e6
length_low, length_high = 0, 200
production_runs = [6,7,8,10,11,12,13,14,15,16,17,31,32,33,34,35,36,37,38,39,40,41,42,43,44,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,73,74,75,77,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98,99,100,101,102,103,104,105,106,107,108,126,133,134,150,151,152,153,154,155,156,157,158,159,160,161,164,165,166,167,168,169,170,171,172,173,174,175,176,177,178,179,180,181,182,183,184,185,186,187,188,189,190,191,192,193,194,195,196,197,198,199,200,201,202,203,204,205,206,207,208,209,210,211,212,213,214,215,216,217,218,219,220,221,222,223,224,225,226,227,228,229,230,231,232,233,234,235,236,237,238,239,240,241,242,243,245,246,247,248,249,250,251,252,253,254,255,256,257,258,259,260,261,262,263,264,265,266,267,268,269,270,271,272,273,274,275,276,278,279,280,281,282,283,284,285,286,287,288,289,290,291,292,293,294,295,296,297,298,299,300,301,302,304,305,306,307,308,309,310,311,312,313,314,315,316,317,318,319,320,321,322,323,324,325,326,327,328,329,330,331,333,334,335,336,337,338,339,340,341,342,343,344,345,346,347,348,349,350,351,352,353,354,355,356,357,358,359,360,361,362,363,364,365,366,368,369,370,371,372,373,374,375,376,377,378,379,380,381,382,383,384,385,386,387,388,389,390,391,392,393,394,395,396,397,398,399,400,401,402,403,404,405,406,407,408,409,410,411,412,413,414,415,416,417,418,419,420,421,422,423,424,425,426,427,428,429,430,431,432,433,434,435,436,437,438,439,440,441,442,443,444,445,446,447,448,449,450,451,452,453,454,455,456,457,458,459,460,461,462,463,464,465,466,467,468,469,470,471,472,473,474,475,476,477,478,479,480,481,482,483,484,485,486,487,488,489,490,491,492,493,494,495,496,497,498,499,500,501,502,503,504,505,506,507,508,509,510,511,512,513,514,515,516,517,518,519,520,521,522,523,524,527,528,529,530,531,532,533,534,535,536,537,538,539,540,541,542,543,544,545,546,547,548,549,550,551,552,553,554,555]
# production_runs = [39] 
# Storage lists
all_counts = []
all_ranges = []
all_veto = []
low_gain_run = []
high_gain_run = []
p_run_low = []
p_run_high = []
run_range = []
corrected_centroids = []
corrected_range_centroids = []
results = {"run": [], "counts_gain_factor": [], "range_gain_factor": []}

def gaussian(x, height, mean, sigma):
    return height * np.exp(-((x - mean) / (2 * sigma)) ** 2)

for run in tqdm(production_runs):
    if run <= 5:
        continue

    base_path = f"/mnt/daqtesting/protondet2024/h5/run_{run:04d}/run_{run:04d}p10_2000torr"
    counts = np.load(f"{base_path}/counts.npy")
    veto = np.load(f"{base_path}/veto.npy")
    dxy = np.load(f"{base_path}/dxy.npy")
    dt = np.load(f"{base_path}/dt.npy")

    # Vectorized length calculation
    length = np.sqrt(dxy**2 + zscale * dt**2)

    # print(f"Events in run {run}: {len(counts)}")

    # Apply mask once
    counts_mask = (counts < 9e5) & (counts > 5.5e5) & (length > 34)
    if run < 48:
        counts_mask = (counts < 9e5) & (counts > 5.5e5) & (length > 43)


    # Histogram counts
    hist, bin_edges = np.histogram(counts[counts_mask], bins=bins, range=(energy_low, energy_high))
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Histogram lengths
    hist_length, bin_edges_length = np.histogram(length[counts_mask], bins=bins, range=(length_low, length_high))
    bin_centers_length = (bin_edges_length[:-1] + bin_edges_length[1:]) / 2

    # plt.figure(figsize=(10, 5))
    # plt.subplot(1, 2, 1)
    # if run > 47:
    #     plt.hist2d(counts[counts>0],length[counts>0], bins = 200, cmap='viridis', range=((6e3,1e6),(2,75)), norm=LogNorm())
    #     plt.colorbar(label='Counts in bin')
    #     plt.title('Run %04d Range versus Energy'%run)
    #     plt.xlabel('Energy (ADC Counts)')
    #     plt.ylabel('Range (mm)')
    #     plt.show()
    # Fit energy histogram
    guesses = [300, 6.5e5, 0.3e5]
    popt, _ = curve_fit(gaussian, bin_centers, hist, p0=guesses)

    # Fit length histogram
    guesses_length = [300, 35, 1]
    if run < 48:
        guesses_length = [300, 45, 1]
    popt_length, _ = curve_fit(gaussian, bin_centers_length, hist_length, p0=guesses_length)

    if popt_length[1] >= 39 or popt_length[1] <= 36.5: # if we have a high range, redo the fit with different counts_mask
           # Apply mask once
        counts_mask = (counts < 9e5) & (counts > 5.5e5) & (length > 42)

        # Histogram counts
        hist, bin_edges = np.histogram(counts[counts_mask], bins=bins, range=(energy_low, energy_high))
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        # Histogram lengths
        hist_length, bin_edges_length = np.histogram(length[counts_mask], bins=bins, range=(length_low, length_high))
        bin_centers_length = (bin_edges_length[:-1] + bin_edges_length[1:]) / 2

        # Fit energy histogram
        guesses = [300, 6.5e5, 0.3e5]
        popt, _ = curve_fit(gaussian, bin_centers, hist, p0=guesses)

        # Fit length histogram
        guesses_length = [300, 45, 1]
        popt_length, _ = curve_fit(gaussian, bin_centers_length, hist_length, p0=guesses_length)

    if popt_length[1] >= 39 or popt_length[1] < 36.5: # if we have a low range this run, redo the fit with different counts_mask
           # Apply mask once
        counts_mask = (counts < 9e5) & (counts > 5.5e5) & (length > 42)

        # Histogram counts
        hist, bin_edges = np.histogram(counts[counts_mask], bins=bins, range=(energy_low, energy_high))
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        # Histogram lengths
        hist_length, bin_edges_length = np.histogram(length[counts_mask], bins=bins, range=(length_low, length_high))
        bin_centers_length = (bin_edges_length[:-1] + bin_edges_length[1:]) / 2

        # Fit energy histogram
        guesses = [300, 6.5e5, 0.3e5]
        popt, _ = curve_fit(gaussian, bin_centers, hist, p0=guesses)

        # Fit length histogram
        guesses_length = [300, 43, 1]
        popt_length, _ = curve_fit(gaussian, bin_centers_length, hist_length, p0=guesses_length)

    max_peak_centroid = popt[1]
    max_length_centroid = popt_length[1]

    results["run"].append(run)
    results["counts_gain_factor"].append(max_peak_centroid)
    results["range_gain_factor"].append(max_length_centroid)
    if max_peak_centroid < 6.4e5:
        low_gain_run.append(max_peak_centroid)
        p_run_low.append(run)
    elif max_peak_centroid > 6.4e5:
        high_gain_run.append(max_peak_centroid)
        p_run_high.append(run)
    # run_range.append(max_length_centroid)
    # if max_length_centroid > 40:
    #     print("Large Range calc for Po-212 peak: Run %d"%run)
    corrected_counts = counts * (644000 / max_peak_centroid)
    corrected_ranges = length * (36.547 / max_length_centroid)
    # Do this one more time to see if it works!
    corrected_counts_mask = (corrected_counts < 9e5) & (corrected_counts > 5.5e5) & (corrected_ranges > 33)
    # Histogram counts
    hist, bin_edges = np.histogram(corrected_counts[corrected_counts_mask], bins=bins, range=(energy_low, energy_high))
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Histogram lengths
    hist_length, bin_edges_length = np.histogram(corrected_ranges[corrected_counts_mask], bins=bins, range=(length_low, length_high))
    bin_centers_length = (bin_edges_length[:-1] + bin_edges_length[1:]) / 2

    # Fit energy histogram
    guesses = [300, 6.5e5, 0.3e5]
    popt, _ = curve_fit(gaussian, bin_centers, hist, p0=guesses)

    # Fit length histogram
    guesses_length = [300, 43, 1]
    popt_length, _ = curve_fit(gaussian, bin_centers_length, hist_length, p0=guesses_length)
    corrected_centroids.append(popt[1])
    corrected_range_centroids.append(popt_length[1])

    # Append scaled data
    # if max_length_centroid > 38 or max_length_centroid < 36.5:
    if False:
        run_range.append(0)
        continue
    else:
        # print(max_length_centroid)
        all_counts.append(counts * (644000 / max_peak_centroid))
        all_ranges.append(length * (36.547 / max_length_centroid))
        all_veto.append(veto)
        run_range.append(max_length_centroid)


    # # using these for raw combined RvE and energy plots
    # all_counts.append(counts)
    # all_ranges.append(length)
    # all_veto.append(veto)
plt.rc('font', size=28)

# plt.plot(production_runs,corrected_centroids, 'x')
# plt.axvline(x=38.5, color='gray', linestyle='--')
# plt.axvline(x=125.5, color='gray', linestyle='--')
# plt.axvline(x=192.5, color='gray', linestyle='--')
# plt.axvline(x=239.5, color='gray', linestyle='--')
# plt.axvline(x=301.5, color='gray', linestyle='--')
# plt.axvline(x=366.5, color='gray', linestyle='--')
# plt.axvline(x=428.5, color='gray', linestyle='--')
# plt.axvline(x=492.5, color='gray', linestyle='--')
# plt.xlabel('Run Number')
# plt.ylabel('Corrected Centroids (ADC Counts)')
# plt.show()
# plt.plot(production_runs,corrected_range_centroids, 'x')
# plt.axvline(x=38.5, color='gray', linestyle='--')
# plt.axvline(x=125.5, color='gray', linestyle='--')
# plt.axvline(x=192.5, color='gray', linestyle='--')
# plt.axvline(x=239.5, color='gray', linestyle='--')
# plt.axvline(x=301.5, color='gray', linestyle='--')
# plt.axvline(x=366.5, color='gray', linestyle='--')
# plt.axvline(x=428.5, color='gray', linestyle='--')
# plt.axvline(x=492.5, color='gray', linestyle='--')
# plt.xlabel('Run Number')
# plt.ylabel('Corrected Range Centroids (mm)')
# plt.show()

# plt.plot(production_runs,run_range, 'x')
# plt.axvline(x=38.5, color='gray', linestyle='--')
# plt.axvline(x=125.5, color='gray', linestyle='--')
# plt.axvline(x=192.5, color='gray', linestyle='--')
# plt.axvline(x=239.5, color='gray', linestyle='--')
# plt.axvline(x=301.5, color='gray', linestyle='--')
# plt.axvline(x=366.5, color='gray', linestyle='--')
# plt.axvline(x=428.5, color='gray', linestyle='--')
# plt.axvline(x=492.5, color='gray', linestyle='--')
# plt.xlabel('Run Number')
# plt.ylabel('Range of the Po-212 Peak (mm)')
# plt.show()

# plt.plot(p_run_low,low_gain_run, 'gx', label = 'Low Gain')
# plt.plot(p_run_high,high_gain_run, 'ro', label = 'High Gain')
# plt.axvline(x=38.5, color='gray', linestyle='--')
# plt.axvline(x=125.5, color='gray', linestyle='--')
# plt.axvline(x=192.5, color='gray', linestyle='--')
# plt.axvline(x=239.5, color='gray', linestyle='--')
# plt.axvline(x=301.5, color='gray', linestyle='--')
# plt.axvline(x=366.5, color='gray', linestyle='--')
# plt.axvline(x=428.5, color='gray', linestyle='--')
# plt.axvline(x=492.5, color='gray', linestyle='--')
# plt.xlabel('Run Number')
# plt.ylabel('Mean of the Po-212 Peak (arb. units)')
# plt.legend()
# plt.show()


# Concatenation after loop
total_counts = np.concatenate(all_counts)
total_ranges = np.concatenate(all_ranges)
total_veto = np.concatenate(all_veto)

print("Number of events in energy spectrum: ",len(total_counts))
print("Number of non-vetoed events in energy spectrum: ",len(total_counts[total_veto<300]))

# plt.hist(results["counts_gain_factor"],bins=50,alpha=0.3)
# plt.show()
# plt.hist(results["range_gain_factor"],bins=50,alpha=0.3)
# plt.show()

# np.savetxt("real_data_veto_300_no_run-by-run_corrections.csv",total_counts[total_veto<300])
# print(results)

# Plotting
# 2D hist
plt.figure(0)
plt.title('RvE for All Runs (veto < 300)', fontsize=32)
plt.hist2d(total_counts[total_veto<300], total_ranges[total_veto<300], 150, norm=mpl.colors.LogNorm(), range=[[0, 1.5e6], [0, 100]])
plt.colorbar()
plt.axvline(x=8.65e5, color='r', linestyle='solid')
plt.xlabel('Energy (arb. units)', fontsize=24)
plt.ylabel('Range (mm)', fontsize=24)
plt.xticks(fontsize=24)
plt.yticks(fontsize=24)
plt.show()

# log scale
plt.title("Range Histogram for All Runs")
plt.hist(total_ranges, 1000, range=[0, 1.5e6], color="blue", label='no veto')
plt.hist(total_ranges[total_veto<300], 1000, range=[0, 1.5e6], color="red", label='veto < 300')
plt.yscale('log')
plt.legend()
plt.xlabel('Range (mm)')
plt.ylabel('Counts')
plt.show()

# lin scale
plt.title("Range Histogram for All Runs")
plt.hist(total_ranges, 1000, range=[0, 1.5e6], color="blue", label='no veto')
plt.hist(total_ranges[total_veto<300], 1000, range=[0, 1.5e6], color="red", label='veto < 300')
plt.legend()
plt.xlabel('Range (mm)')
plt.ylabel('Counts')
plt.show()

# log scale
plt.title("Energy Histogram for All Runs")
plt.hist(total_counts, 1000, range=[0, 1.5e6], color="blue", label='no veto')
plt.hist(total_counts[total_veto<300], 1000, range=[0, 1.5e6], color="red", label='veto < 300')
plt.yscale('log')
plt.legend()
plt.xlabel('Energy (arb. units)')
plt.ylabel('Counts')
plt.show()

# lin scale
plt.title("Energy Histogram for All Runs")
plt.hist(total_counts, 1000, range=[0, 1.5e6], color="blue", label='no veto')
plt.hist(total_counts[total_veto<300], 1000, range=[0, 1.5e6], color="red", label='veto < 300')
plt.legend()
plt.xlabel('Energy (arb. units)')
plt.ylabel('Counts')
plt.show()

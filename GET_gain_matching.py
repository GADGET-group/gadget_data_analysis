import numpy as np
import scipy.signal as sig
import matplotlib.pyplot as plt
import matplotlib as mpl
from tqdm import tqdm
# removed run 367 because it has 2 events in it and thus we cannot easily find a peak to gain match our data
# it is also missing runs 46, 76 because it is too small a run for the peak fitter to find the peak properly (TODO: fix the peak fitter smoother so the prominence can be lowered)
production_runs = [6,7,8,10,11,12,13,14,15,16,17,31,32,33,34,35,36,37,38,39,40,41,42,43,44,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,73,74,75,77,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98,99,100,101,102,103,104,105,106,107,108,126,133,134,150,151,152,153,154,155,156,157,158,159,160,161,164,165,166,167,168,169,170,171,172,173,174,175,176,177,178,179,180,181,182,183,184,185,186,187,188,189,190,191,192,193,194,195,196,197,198,199,200,201,202,203,204,205,206,207,208,209,210,211,212,213,214,215,216,217,218,219,220,221,222,223,224,225,226,227,228,229,230,231,232,233,234,235,236,237,238,239,240,241,242,243,245,246,247,248,249,250,251,252,253,254,255,256,257,258,259,260,261,262,263,264,265,266,267,268,269,270,271,272,273,274,275,276,278,279,280,281,282,283,284,285,286,287,288,289,290,291,292,293,294,295,296,297,298,299,300,301,302,304,305,306,307,308,309,310,311,312,313,314,315,316,317,318,319,320,321,322,323,324,325,326,327,328,329,330,331,333,334,335,336,337,338,339,340,341,342,343,344,345,346,347,348,349,350,351,352,353,354,355,356,357,358,359,360,361,362,363,364,365,366,368,369,370,371,372,373,374,375,376,377,378,379,380,381,382,383,384,385,386,387,388,389,390,391,392,393,394,395,396,397,398,399,400,401,402,403,404,405,406,407,408,409,410,411,412,413,414,415,416,417,418,419,420,421,422,423,424,425,426,427,428,429,430,431,432,433,434,435,436,437,438,439,440,441,442,443,444,445,446,447,448,449,450,451,452,453,454,455,456,457,458,459,460,461,462,463,464,465,466,467,468,469,470,471,472,473,474,475,476,477,478,479,480,481,482,483,484,485,486,487,488,489,490,491,492,493,494,495,496,497,498,499,500,501,502,503,504,505,506,507,508,509,510,511,512,513,514,515,516,517,518,519,520,521,522,523,524,527,528,529,530,531,532,533,534,535,536,537,538,539,540,541,542,543,544,545,546,547,548,549,550,551,552,553,554,555]

# production_runs = [6,7,8,10,11,12,13,14,15,16,17,31,32,33,34,35,36,37,38,39,40,41,42,43,44,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,73,74,75,77,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98,99,100,101,102,103,104,105,106,107,108,126,133,134,150,151,152,153,154,155,156,157,158]
# production_runs = [55]
total_counts = []
total_ranges = []
total_veto =[]
dxy = []
dt = []

zscale = 0.65

# Parameters used to find gain matching scaler
# bins = 4000 # bins in the energy histogram; this is the good one for now
bins = 2000
energy_low = 0
energy_high = 1e6
length_low = 0
length_high = 200
smoothing_factor = 100 # averaging window size to ensure proper peak search results
smoothing_factor_length = 80
results = {"run": [], "counts_peak_loc": [], "range_peak_loc": [], "counts_gain_factor": [], "range_gain_factor": []}

for run in production_runs:
    if run > 158:
        length = []
        counts =[]
        veto = []
        dxy = []
        dt = []

        counts = np.load('/mnt/daqtesting/protondet2024/h5/run_%04d/run_%04dp10_2000torr/counts.npy'%(run,run))
        veto = np.load('/mnt/daqtesting/protondet2024/h5/run_%04d/run_%04dp10_2000torr/veto.npy'%(run,run))

        dxy = np.load('/mnt/daqtesting/protondet2024/h5/run_%04d/run_%04dp10_2000torr/dxy.npy'%(run,run))
        dt = np.load('/mnt/daqtesting/protondet2024/h5/run_%04d/run_%04dp10_2000torr/dt.npy'%(run,run))
        for event in range(len(dxy)):
            length.append((dxy[event]**2 + zscale*dt[event]**2)**0.5)
        length = np.array(length)

        print('Events in run %d: '%run,len(counts))
        counts_mask = np.logical_and.reduce((counts<9e5,
                                    counts>5e5))
        #  Histogram the projection of energies
        hist, bin_edges = np.histogram(counts[counts_mask], bins=bins, range=(energy_low, energy_high))
        bin_width = (energy_high - energy_low) / bins
        # length_mask = np.logical_and.reduce((length>32,
        #                                      length<55))
        # Now we do the same thing for ranges
        hist_length, bin_edges_length = np.histogram(length, bins=bins, range=(length_low, length_high))
        bin_width_length = (length_high - length_low) / bins

        # Perform a moving average smoothing for both energy and ranges
        window = np.arange(0, bins, 1)
        window_centered = window - (window[-1] + window[0]) / 2
        fil = np.fft.ifftshift(np.sinc(window_centered / smoothing_factor))  # Size of points taken for average is denominator
        transformed = np.fft.fft2(hist, axes=(0,))
        hist_smoothed = np.real(np.fft.ifft2(transformed * fil, axes=(0,)))

        fil_length = np.fft.ifftshift(np.sinc(window_centered / smoothing_factor_length))
        transformed_length = np.fft.fft2(hist_length, axes=(0,))
        hist_length_smoothed = np.real(np.fft.ifft2(transformed_length * fil_length, axes=(0,)))

        # Find largest peak in smoothed histograms
        pks, props = sig.find_peaks(hist_smoothed, distance=1, prominence=10, width=7, rel_height=0.95)
        pks_length, props_length = sig.find_peaks(hist_length_smoothed, distance=1, prominence=10, width=10, rel_height=0.95)

        max_peak_centroid = pks[-1] * bin_width + energy_low

        max_length_centroid = pks_length[np.argmax(hist_length_smoothed[pks_length])]*bin_width_length + length_low
        
        results["run"].append(run)
        results["counts_gain_factor"].append(max_peak_centroid)
        results["range_gain_factor"].append(max_length_centroid)

        print("Peaks in energy in run %d: "%run,pks)
        print("Peaks in range in run %d: "%run,pks_length)
        # # plot 1d hist of ranges
        # fig,ax = plt.subplots()
        # ax.bar(bin_edges_length[:-1],hist_length,width=np.diff(bin_edges_length),edgecolor="black",align="edge")
        # # plt.hist(total_counts, bins=bins)
        # plt.plot(pks_length * bin_width_length + length_low, hist_length_smoothed[pks_length], "rx")
        # plt.show()
        total_counts = np.concatenate([total_counts,np.multiply(counts,498000/max_peak_centroid)])
        total_ranges = np.concatenate([total_ranges,np.multiply(length,40/max_length_centroid)])
        total_veto = np.concatenate([total_veto,veto])

print(results)
# fig,ax = plt.subplots()
# ax.bar(bin_edges[:-1],hist_smoothed,width=np.diff(bin_edges),edgecolor="black",align="edge")
# # plt.hist(total_counts, bins=bins)
# plt.plot(pks * bin_width + energy_low, hist_smoothed[pks], "rx")
# plt.show()

fig,ax = plt.subplots()
ax.bar(bin_edges_length[:-1],hist_length_smoothed,width=np.diff(bin_edges_length),edgecolor="black",align="edge")
# plt.hist(total_counts, bins=bins)
plt.plot(pks_length * bin_width_length + length_low, hist_length_smoothed[pks_length], "rx")
plt.show()

plt.figure(0)
plt.title('RvE for Runs 1-158', fontsize=32)
plt.hist2d(total_counts, total_ranges, 200, norm=mpl.colors.LogNorm(), range=[[0,2e6],[0,200]])
# plt.hist(total_counts, bins=200)
# plt.colorbar(labelsize=24)
plt.colorbar()
plt.axvline(x = 8.65e5, color = 'r', linestyle = 'solid')
plt.xlabel('Energy (arb. units)', fontsize=24)
plt.ylabel('Range (mm)', fontsize=24)
plt.xticks(fontsize=24)
plt.yticks(fontsize=24)
plt.show()

plt.title("Energy Histogram for Runs 1-158")
plt.hist(total_counts, 1000, range=[0,2e6])
plt.xlabel('Energy (arb. units)')
plt.show()
import h5py
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import scipy.optimize as opt
runs = [18,19,20,21,22,23,24,25,26,27]

production_runs = [6,7,8,10,11,12,13,14,15,16,17,31,32,33,34,35,36,37,38,39,40,41,42,43,44,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,73,74,75,77,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98,99,100,101,102,103,104,105,106,107,108,126,133,134,150,151,152,153,154,155,156,157,158]
timestamps = []
timestamp_diff = []
zscale = 0.65

for run in  tqdm(production_runs):
    path = "/Volumes/Extreme SSD/e24joe/"
    path = "/mnt/daqtesting/protondet2024/h5/"
    #base_path = path + 'run_%04d_raw_data_export/'%run
    base_path = path + 'run_%04d/run_%04dp10_2000torr/'%(run,run)
    run = 'run_%04d.h5'%run
    timestamps_list = []
    
    with h5py.File(path + run) as f:
        first_event, last_event = int(f['meta']['meta'][0]), int(f['meta']['meta'][2])
        first_timestamp = f['meta']['meta'][1]
        counts = np.load(base_path+'counts.npy')
        # dxy = np.load(base_path+'dxy.npy')
        # dt = np.load(base_path+'dt.npy')
        timestamps_list = np.load(base_path+'timestamps.npy')
        #ranges = np.load(base_path+'ranges.npy')
        # ranges = np.sqrt(dxy**2 + (dt*zscale)**2)
        # event_numbers = np.load(base_path+'event_numbers.npy')
        # Selecting on a particular energy event
        # event_select_mask = np.logical_and(ranges > 35, np.logical_and(ranges < 50, counts>6*578472/6.288))
        # indexes = np.where(event_select_mask == True)
        # It looks like the timestamps are in units of 10 ns, so we divide by 1e8 to get seconds
        # for i in range(len(counts)):
        #     timestamps_list.append((f['get']['evt%d_header'%i][1]))
        # timestamps = np.concatenate([timestamps, timestamps_list])
        timestamps = np.concatenate([timestamps, [t - timestamps_list[0] for t in timestamps_list]])
        timestamp_diff = np.concatenate([timestamp_diff, [timestamps_list[i] - timestamps_list[i-1] for i in range(1, len(timestamps_list))]])
# timestamps = np.loadtxt("./alpha_time_dist_all_angles.csv", delimiter=",", usecols=0)
print(len(timestamps))
print(np.shape(timestamp_diff))
# timestamp_diff = [timestamps[i] - timestamps[i-1] for i in range(1, len(timestamps))]
# hist, bin_edges = np.histogram(timestamp_diff, bins=100)
# hist, bin_edges = np.histogram((timestamps-timestamps[0]), bins=100)
hist, bin_edges = np.histogram((timestamp_diff), bins=20000)
#put everything in units of counts/time bin
sigma = np.sqrt(hist)/(bin_edges[1] - bin_edges[0])
hist = hist/(bin_edges[1] - bin_edges[0])

#plot
plt.stairs(hist, bin_edges)
# bin_centers = (bin_edges[0:-1] + bin_edges[1:])/2
# exp = lambda t, A, tau, C: A*np.exp(-t/tau) + C
# popt, pcov = opt.curve_fit(exp, bin_centers, hist, p0=(5, 10, 0), sigma=sigma, absolute_sigma=True)
# print('half life = %f +/- %f ns'%(popt[1]*np.log(2)*20, np.sqrt(pcov[1,1])*np.log(2)*20))
# plt.plot(bin_centers, exp(bin_centers, *popt))
# print('popt = ',popt)
# print('uncertainties: ', np.sqrt(np.diag(pcov)))
plt.yscale('log')
plt.xlabel('Time Difference Between Events (seconds)')
plt.ylabel('Counts per bin')
# plt.show(block=False)
# plt.savefig('no_flow_half_life.png')

# plt.figure()
# plt.plot(bin_centers, exp(bin_centers, *popt) - hist)
plt.show()
# plt.savefig('no_flow_half_life_residuals.png')

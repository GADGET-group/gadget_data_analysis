import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt

runs = [18,19,20,21,22,23,24,25,26,27,28,29,30]
all_timestamps = []
last_timestamp = -1

for run in runs:
    print("Last Timestamp: %d"%last_timestamp)
    timestamp = np.load('/mnt/analysis/e21072/gastest_h5_files/e24joe/run_%04d/run_%04dp10_2000torr/timestamps.npy'%(run,run))
    first_timestamp = timestamp[0]
    print("First Timestamp: %d"%first_timestamp)
    timestamp = [x-first_timestamp for x in timestamp]
    if last_timestamp != -1:
        timestamp = [x+last_timestamp+1 for x in timestamp]
        all_timestamps = np.concatenate([all_timestamps,timestamp])
        last_timestamp = timestamp[-1]
        continue
    all_timestamps = np.concatenate([all_timestamps,timestamp])
    last_timestamp = timestamp[-1]

print("Total Events: %d"%len(all_timestamps))

counts, bin_edges = np.histogram(all_timestamps/3600, bins = 100)
sigma = np.sqrt(counts)/(bin_edges[1]-bin_edges[0])/3600
hist = counts/(bin_edges[1] - bin_edges[0])/3600

bin_centers = (bin_edges[0:-1] + bin_edges[1:])/2
two_exps = lambda t, A1, A2, tau1, tau2, C: A1*np.exp(-t/tau1) + A2*np.exp(-t/tau2) + C
# two_exps = lambda t, A1, tau1, C: A1*np.exp(-t/tau1) + C
popt, pcov = opt.curve_fit(two_exps,bin_centers,hist,p0=(5,5,16,125,0), sigma=sigma, absolute_sigma=True)
print("half-lives = %f +/- %f hours and %f +/- %f hours"%(popt[2]*np.log(2),np.sqrt(pcov[2,2])*np.log(2),popt[3]*np.log(2),np.sqrt(pcov[3,3])*np.log(2)))
print("popt = A1, A2, tau1, tau2, C")
print("popt = ",popt)
print("uncertainties: ", np.sqrt(np.diag(pcov)))
plt.plot(bin_centers,two_exps(bin_centers, *popt))
plt.stairs(hist,bin_edges)
plt.show()
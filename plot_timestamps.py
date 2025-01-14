import matplotlib.pyplot as plt
import numpy as np

run = 'run_0430.h5'

# timestamps = np.loadtxt("timestamps_%s.txt"%run)

timestamps = np.loadtxt("./alpha_time_dist_all_angles.csv", delimiter=",", usecols=0)
plt.hist(timestamps,bins=20)
plt.show()

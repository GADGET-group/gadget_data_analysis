import pickle

import matplotlib.pylab as plt
import matplotlib.path
import matplotlib.colors

import numpy as np
from scipy import optimize

#load data
fname = 'e21072_121to128.pkl'
with open(fname, 'rb') as f:
    data = pickle.load(f)
lengths = data['length'] #track lengths in mm
MeV = data['energy'] #energy in MeV after applying gain match
veto_pad_counts = data['veto_pad_counts'] #integrated charge on all veto pads in adc counts

#apply veto condition and make proton and alpha cuts
veto_threshold = 10000
veto_mask = veto_pad_counts < veto_threshold

rve_points = np.vstack((MeV, lengths)).transpose()
verticies = [(0.728, 11.5), (0.843, 11),(0.849,15.9), (2.772,50.5), (3.5, 100), (3.1, 165), (2.401,100), (0.607,17.3)]
path = matplotlib.path.Path(verticies)
proton_cut_mask = path.contains_points(rve_points)&veto_mask
verticies = [(1.51,19.3),(9.17,86.9), (9.41,47.1), (1.51, 3)]
path = matplotlib.path.Path(verticies)
alpha_cut_mask = path.contains_points(rve_points)&veto_mask

#show proton and alpha cuts on RvE plot
plt_mask = veto_mask
rve_bins = 1000
plt.figure()
plt.title('events used in gain match')
plt.hist2d(MeV[plt_mask], lengths[plt_mask], bins=rve_bins, norm=matplotlib.colors.LogNorm())
plt.colorbar()
plt.scatter(MeV[proton_cut_mask], lengths[proton_cut_mask], marker='.', label='proton events')
plt.scatter(MeV[alpha_cut_mask], lengths[alpha_cut_mask], marker='.', label='alpha events')
plt.xlabel('Energy (MeV)')
plt.ylabel('range (mm)')

#show projections on energy axis
plt.figure()
plt.title('proton energy spectrum')
plt.hist(MeV[proton_cut_mask], 1000)

plt.figure()
plt.title('alpha energy spectrum')
plt.hist(MeV[alpha_cut_mask], 1000)
plt.show()


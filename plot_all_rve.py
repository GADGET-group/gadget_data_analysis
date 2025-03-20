import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

production_runs = []
# production_runs = [3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,31]
runs = 42
for i in range(runs):
    production_runs.append(i+1)

total_counts = []
total_ranges = []
total_veto =[]
dxy = []
dt = []

zscale = 0.65

# plot rve run by run
# for run in production_runs:
length = []
# total_counts = np.load('/mnt/daqtesting/protondet2024/h5/run_%04d/run_%04dp10_2000torr/counts.npy'%(run,run))
# total_veto = np.load('/mnt/daqtesting/protondet2024/h5/run_%04d/run_%04dp10_2000torr/veto.npy'%(run,run))

# dxy = np.load('/mnt/daqtesting/protondet2024/h5/run_%04d/run_%04dp10_2000torr/dxy.npy'%(run,run))
# dt = np.load('/mnt/daqtesting/protondet2024/h5/run_%04d/run_%04dp10_2000torr/dt.npy'%(run,run))
total_counts = np.load('/mnt/daqtesting/protondet2024/complete_runs_arraysp10_2000torr/counts.npy')
# total_veto = np.load('/mnt/daqtesting/protondet2024/complete_runs_arraysp10_2000torr/veto.npy')

dxy = np.load('/mnt/daqtesting/protondet2024/complete_runs_arraysp10_2000torr/dxy.npy')
dt = np.load('/mnt/daqtesting/protondet2024/complete_runs_arraysp10_2000torr/dt.npy')
for event in range(len(dxy)):
    length.append((dxy[event]**2 + zscale*dt[event]**2)**0.5)
total_ranges = np.array(length)
# mask = np.logical_and.reduce((total_veto<150,
#                             total_counts<np.inf,
#                             total_counts>-np.inf,
#                             total_ranges>-np.inf,
#                             total_ranges<np.inf
#                             ))
# print('Events in run %4d: '%run,len(total_counts[mask]))
plt.figure()
# plt.title('run_%d'%run)
# plt.hist2d(total_counts[mask], total_ranges[mask], 150, norm=mpl.colors.LogNorm(), range=[[0,1e6],[0,150]])
plt.hist2d(total_counts, total_ranges, 150, norm=mpl.colors.LogNorm(), range=[[0,1e6],[0,150]])
plt.colorbar()
plt.title('RvE All Runs')
plt.xlabel('Energy (ADC counts)')
plt.ylabel('Range (mm)')
plt.savefig('RvE_all_runs_no_veto_4mar25.png')
plt.close()

# Create 6 MeV mask for plotting
# mask = np.logical_and.reduce((total_veto<150,
#                               total_counts<6.34e5,
#                               total_counts>4.14e5,
#                               total_ranges>26,
#                               total_ranges<39
#                               ))
# 8 MeV Mask
# mask = np.logical_and.reduce((total_veto<150,
#                               total_counts<8.545e5,
#                               total_counts>5.756e5,
#                               total_ranges>40,
#                               total_ranges<50
#                               ))
# No mask




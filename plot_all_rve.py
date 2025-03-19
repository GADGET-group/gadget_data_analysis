import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd

total_counts = []
total_ranges = []
total_veto =[]
dxy = []
dt = []

zscale = 0.65

categorized_events_of_interest = pd.read_csv('/egr/research-tpc/dopferjo/categorized_events_of_interest.csv',encoding='utf-8-sig', skip_blank_lines = False, nrows = 36164, header=None)
# print(categorized_events_of_interest.head())
# print(categorized_events_of_interest.tail())
print(len(categorized_events_of_interest))
print(categorized_events_of_interest.columns)
# plot rve run by run
# for run in production_runs:
length = []
total_counts = np.load('/egr/research-tpc/dopferjo/interesting_events_without_run_number_in_event_name_without_event_447_raw_viewer/interesting_events_without_run_number_in_event_name_without_event_447p10_2000torr/counts.npy')
total_veto = np.load('/egr/research-tpc/dopferjo/interesting_events_without_run_number_in_event_name_without_event_447_raw_viewer/interesting_events_without_run_number_in_event_name_without_event_447p10_2000torr/veto.npy')
dxy = np.load('/egr/research-tpc/dopferjo/interesting_events_without_run_number_in_event_name_without_event_447_raw_viewer/interesting_events_without_run_number_in_event_name_without_event_447p10_2000torr/dxy.npy')
dt = np.load('/egr/research-tpc/dopferjo/interesting_events_without_run_number_in_event_name_without_event_447_raw_viewer/interesting_events_without_run_number_in_event_name_without_event_447p10_2000torr/dt.npy')
for event in range(len(dxy)):
    length.append((dxy[event]**2 + zscale*dt[event]**2)**0.5)
total_ranges = np.array(length)

print(len(total_ranges))
print(len(total_counts))

print(np.max(total_counts))
print(np.max(total_ranges))

# mask = np.logical_and.reduce((total_veto<150,
#                             total_counts<np.inf,
#                             total_counts>-np.inf,
#                             total_ranges>-np.inf,
#                             total_ranges<np.inf
#                             ))

# mask = categorized_events_of_interest[0] != 'Weird Oscillations' or categorized_events_of_interest[0] != 'flagged' or categorized_events_of_interest[0] != 'Noisy'  

array_of_categorized_events_of_interest = categorized_events_of_interest[0].to_numpy()

print(array_of_categorized_events_of_interest)

mask = []
for i in range(len(array_of_categorized_events_of_interest)):
    if array_of_categorized_events_of_interest[i] == "Weird Oscillations" or array_of_categorized_events_of_interest[i] == "Noisy" or array_of_categorized_events_of_interest[i] == "flagged":
        print(array_of_categorized_events_of_interest[i])
        mask.append(True)
    else:
        mask.append(False)

print(mask)
print(len(mask))

# print('Events in run %4d: '%run,len(total_counts[mask]))
plt.figure()
# plt.title('run_%d'%run)
# plt.hist2d(total_counts[mask], total_ranges[mask], 150, norm=mpl.colors.LogNorm(), range=[[0,1e6],[0,150]])
plt.hist2d(total_counts[mask], total_ranges[mask], 150, norm=mpl.colors.LogNorm(), range=[[0,3.03e7],[0,420]])
plt.colorbar()
plt.title('RvE_noisy_events_of_interest')
plt.xlabel('Energy (ADC counts)')
plt.ylabel('Range (mm)')
# plt.show()
plt.savefig('RvE_noisy_events_of_interest.png')
# plt.savefig('RvE_only_flagged_events_and_labelled_noise.png')
# plt.close()
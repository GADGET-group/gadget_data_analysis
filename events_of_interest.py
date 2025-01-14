import numpy as np
import csv
import matplotlib.pyplot as plt
import matplotlib as mpl
import h5py

# production_runs = [5,6,7,8,10,11,12,13,14,15,16,17,31,32,33,34,35,36,37,38,39,40,41,42,43,44,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,73,74,75,76,77,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98,99,100,101,102,103,104,108,126,127,129,133,134,150,151,152,153,154,155,156,157,158,159,160,161,162,164,165,166,167,168,169,170,171,172,173,174,175,176,177,178,179,180,181,182,183,184,185,186,187,188,189,190,191,192,193]

# RUN 47 HAS YET TO BE PROCESSED< AND SO HAS NOT BEEN INCLUDED IN THE RARE EVENT SEARCH YET

# production_runs = [194,195,196,197,198,199,200]
production_runs = [193]
def events_of_interest_search():
    counts = []
    length = []
    ranges = []
    angles = []
    zscale = 0.65

    for run in production_runs:
        counts = np.load('/mnt/daqtesting/protondet2024/h5/run_%04d/run_%04dp10_2000torr/counts.npy'%(run,run))
        angles = np.load('/mnt/daqtesting/protondet2024/h5/run_%04d/run_%04dp10_2000torr/angles.npy'%(run,run))
        dxy = np.load('/mnt/daqtesting/protondet2024/h5/run_%04d/run_%04dp10_2000torr/dxy.npy'%(run,run))
        dt = np.load('/mnt/daqtesting/protondet2024/h5/run_%04d/run_%04dp10_2000torr/dt.npy'%(run,run))
        for event in range(len(dxy)):
            length.append((dxy[event]**2 + zscale*dt[event]**2)**0.5)
        ranges = length
        i = 0
        for e, r in zip(counts, ranges):
            if e > 8.65e5:
                # here we do a quick check to see if the high charge producing event is noise or not
                # because of the way the trigger is set up, the first event should be triggered at around time bin 160
                # if there is a lot of charge showing up in the time bin range 20-80, then it comes from noisy capacitor switching

                print('Check Event %d in Run %d'%(i,run))
            i = i+1

def plot_angles_of_po():
    counts = []
    length = []
    ranges = []
    veto = []
    angles = []
    po_angles = []
    dxy = []
    dt = []
    zscale = 0.65

    for run in production_runs:
        counts = np.concatenate([counts,np.load('/mnt/daqtesting/protondet2024/h5/run_%04d/run_%04dp10_2000torr/counts.npy'%(run,run))])
        angles = np.concatenate([angles,np.load('/mnt/daqtesting/protondet2024/h5/run_%04d/run_%04dp10_2000torr/angles.npy'%(run,run))])
        veto = np.concatenate([veto,np.load('/mnt/daqtesting/protondet2024/h5/run_%04d/run_%04dp10_2000torr/veto.npy'%(run,run))])
        dxy = np.concatenate([dxy,np.load('/mnt/daqtesting/protondet2024/h5/run_%04d/run_%04dp10_2000torr/dxy.npy'%(run,run))])
        dt = np.concatenate([dt,np.load('/mnt/daqtesting/protondet2024/h5/run_%04d/run_%04dp10_2000torr/dt.npy'%(run,run))])
        for event in range(len(dxy)):
            length.append((dxy[event]**2 + zscale*dt[event]**2)**0.5)
        ranges = np.concatenate([ranges,length])
    for e, r, a, v in zip(counts, ranges, angles, veto):
        if e>5.756e5 and e<8.54e5 and r>40 and r<50 and v<150: # 8 Mev blob gate
        # if e>4.140e5 and e<6.34e5 and r>26 and r<39 and v<150: # 6 Mev blob gate
            po_angles.append(a)#a*180/np.pi)
    plt.figure()
    # plt.title('run_%d'%run)
    plt.hist(po_angles, bins=90, weights=1/np.sin(po_angles))
    plt.title('8 MeV blob angles')
    plt.xlabel('Angles (degrees)')
    plt.ylabel('Counts')
    plt.show()
    plt.close()


def add_events_to_h5():
    # event_info = np.loadtxt("./events_of_interest.csv", delimiter=",", usecols=0)
    runnum = []
    eventnum = []
    eventtype = []
    notes = []

    dest_h5 = h5py.File("/mnt/daqtesting/protondet2024/interesting_events_without_run_number_in_event_name_without_event_447.h5", "w")
    meta = dest_h5.create_group('meta')
    get = dest_h5.create_group('get')
    # THIS ARRAY NEEDS TO BE CHANGED MAUALLY TO MATCH THE NUMBER OF EVENTS TO BE WRITTEN TO THE h5 FILE
    arr = [0,0,36162,1e20]
    meta.create_dataset('meta',data=arr)
    with open('events_of_interest_no_header_no_event447.csv', mode='r', encoding='utf-8-sig') as fobj:
        for line in fobj:
            row = line.split(sep=',')
            runnum.append(float(row[0]))
            eventnum.append(float(row[1]))
            # eventtype.append(row[2])
            # notes.append(row[3])

    # with h5py.File("/mnt/daqtesting/protondet2024/h5/run_0001.h5","r") as temph5:
    #     get.create_dataset('run1_evt155318_header', data=temph5['get']['evt155318_header'])
    #     get.create_dataset('run1_evt155318_data', data=temph5['get']['evt155318_data'])

    prev_run = -1
    i = 0
    for run in runnum:
        # pull events of interest from the temph5 and add them to the interesting_events h5
        # add events here (TODO: change this from pseudocode to real code)
        # if prev_run != run:
        with h5py.File("/mnt/daqtesting/protondet2024/h5/run_%04d.h5"%run,"r") as temph5:
            get.create_dataset('evt%d_header'%(i), data=temph5['get']['evt%d_header'%eventnum[i]])
            get.create_dataset('evt%d_data'%(i), data=temph5['get']['evt%d_data'%eventnum[i]])
            # get.create_dataset('run%d_evt%d_header'%(runnum[i], eventnum[i]), data=temph5['get']['evt%d_header'%eventnum[i]])
            # get.create_dataset('run%d_evt%d_data'%(runnum[i], eventnum[i]), data=temph5['get']['evt%d_data'%eventnum[i]])
        print("Finished Adding Event %d from Run %04d"%(eventnum[i], runnum[i]))
        print(i)
        prev_run = run
        i = i+1

# import numpy as np
# import csv
# import pandas as pd

# # production_runs = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,31,32,33,34,35,36,37,38,39,40,41,42]
# production_runs = [1,2,3,4,5,7,8,9,10,11,12,13,14,15,16,17,31]

# counts = []
# df = pd.DataFrame(columns=['Run','Event'])
# df2 = pd.DataFrame(columns=['Run','Event'])

# for run in production_runs:
#     counts = np.load('/mnt/daqtesting/protondet2024/h5/run_%04d/run_%04dp10_2000torr/counts.npy'%(run,run))
#     for event in counts:
#         if event > 8.6e5:
#             df2.iloc[0,0] = run
#             df2.iloc[0,1] = event
#             print(df2)

# print(interesting_event)

add_events_to_h5()

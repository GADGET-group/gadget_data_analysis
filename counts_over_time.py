import numpy as np
import matplotlib.pyplot as plt
import h5py

files = []

pwd = '/mnt/analysis/e21072/gastest_h5_files/gastestsrun_'
h5 = '.h5'

runs = range(168, 172)

# Create a list of paths to each run we want to combine
for run in runs:
    files.append('%s%04d%s'% (pwd, run, h5))

# Combine runs into a single h5 file
with h5py.File('runs_merged.h5',mode='w') as h5fw:
    h5fw.create_group('get')
    h5fw.create_group('meta')
    for h5name in files:
        h5fr = h5py.File(h5name,'r') 
        dset0 = list(h5fr.keys())[0]
        dset1 = list(h5fr.keys())[1]
        arr_data_get = h5fr[dset0]
        arr_data_meta = h5fr[dset1]

        # print(arr_data_get)
        for items in arr_data_meta:
            # print(arr_data_meta[items]['cobo0asad0_files'][0])
            if items == 'meta':
                for i in range(len(items)):
                    print(arr_data_meta[items][i])
            print(items)
            print(arr_data_meta[items][0])
        # h5fw.copy(dset1,dset1)
        h5fr.close()

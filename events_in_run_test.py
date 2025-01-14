import numpy as np
import csv
import matplotlib.pyplot as plt
import matplotlib as mpl

runs = [196,200,201,202,203]

for run in runs:
    timestamps = np.load('/mnt/daqtesting/protondet2024/h5/run_%04d/run_%04dp10_2000torr/timestamps.npy'%(run,run))
    counts = np.load('/mnt/daqtesting/protondet2024/h5/run_%04d/run_%04dp10_2000torr/counts.npy'%(run,run))
    # angles = np.load('/mnt/daqtesting/protondet2024/h5/run_%04d/run_%04dp10_2000torr/angles.npy'%(run,run))
    # dt = np.load('/mnt/daqtesting/protondet2024/h5/run_%04d/run_%04dp10_2000torr/dt.npy'%(run,run))
    # dxy = np.load('/mnt/daqtesting/protondet2024/h5/run_%04d/run_%04dp10_2000torr/dxy.npy'%(run,run))
    # veto = np.load('/mnt/daqtesting/protondet2024/h5/run_%04d/run_%04dp10_2000torr/veto.npy'%(run,run))
    print("Run #",run)
    print(len(counts))
    # print(len(angles))
    # print(len(dt))
    # print(len(dxy))
    # print(len(timestamps))
    # print(len(veto))
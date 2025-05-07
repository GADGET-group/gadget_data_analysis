import numpy as np
import os
import matplotlib.pylab as plt
import matplotlib.colors as colors
from matplotlib.colors import LinearSegmentedColormap
import sys

np.set_printoptions(threshold=sys.maxsize)

data = np.load('/egr/research-tpc/dopferjo/gadget_analysis/padgain_noveto.npy')
print(np.shape(data))
data_with_vetos = np.insert(data,[253,253,506,506,759,759],0)
data_with_vetos = np.append(data_with_vetos,[0,0])
print(np.shape(data_with_vetos))

pad_plane = np.genfromtxt(os.path.join(os.path.dirname(__file__),'raw_viewer/PadPlane.csv'),delimiter=',', filling_values=-1)

# print(np.shape(data[data>0]))

pad_to_xy_index = {}
for y in range(len(pad_plane)):
    for x in range(len(pad_plane[0])):
        pad = pad_plane[x,y]
        if pad != -1:
            pad_to_xy_index[int(pad)] = (x,y)

image = np.zeros(np.shape(pad_plane))
pad = 0
for line in data_with_vetos:
    # chnl_info = tuple(line[0:4])
    # if chnl_info not in self.chnls_to_pad:
    #     print('warning: the following channel tripped but doesn\'t have  a pad mapping: '+str(chnl_info))
    #     continue
    x,y = pad_to_xy_index[pad]
    image[x,y] = line
    pad = pad + 1
image[image<0]=0
# trace = np.sum(data[:,FIRST_DATA_BIN:],0)
fig_name = None
fig = plt.figure(fig_name, figsize=(6,6))
plt.clf()
# should_veto, dxy, dz, energy, angle, pads_railed_list = self.process_event(event_number)
# length = np.sqrt(dxy**2 + dz**2)
plt.title('Results of Pad Gainmatching for First 100,000 Events in the Po-212 Peak')
# plt.subplot(2,1,1)
plt.imshow(image, norm=colors.LogNorm())
plt.colorbar()
# plt.subplot(2,1,2)
# plt.plot(trace)
plt.show(block=True)
from raw_viewer import raw_h5_file
import configparser
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

FIRST_DATA_BIN = 6 #first time bin is dumped, because it is junk
VETO_PADS = (253, 254, 508, 509, 763, 764, 1018, 1019)

production_runs = [6,7,8,10,11,12,13,14,15,16,17,31,32,33,34,35,36,37,38,39,40,41,42,43,44,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,73,74,75,77,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98,99,100,101,102,103,104,105,106,107,108,126,133,134,150,151,152,153,154,155,156,157,158,159,160,161,164,165,166,167,168,169,170,171,172,173,174,175,176,177,178,179,180,181,182,183,184,185,186,187,188,189,190,191,192,194,195,196,197,198,199,200,201,202,203,204,205,206,207,208,209,210,211,212,213,214,215,216,217,218,219,220,221,222,223,224,225,226,227,228,229,230,231,232,233,234,235,236,237,238,239,240,241,242,243,245,246,247,248,249,250,251,252,253,254,255,256,257,258,259,260,261,262,263,264,265,266,267,268,269,270,271,272,273,274,275,276,278,279,280,281,282,283,284,285,286,287,288,289,290,291,292,293,294,295,296,297,298,299,300,301,302,304,305,306,307,308,309,310,311,312,313,314,315,316,317,318,319,320,321,322,323,324,325,326,327,328,329,330,331,333,334,335,336,337,338,339,340,341,342,343,344,345,346,347,348,349,350,351,352,353,354,355,356,357,358,359,360,361,362,363,364,365,366,367,368,369,370,371,372,373,374,375,376,377,378,379,380,381,382,383,384,385,386,387,388,389,390,391,392,393,394,395,396,397,398,399,400,401,402,403,404,405,406,407,408,409,410,411,412,413,414,415,416,417,418,419,420,421,422,423,424,425,426,427,428,429,430,431,432,433,434,435,436,437,438,439,440,441,442,443,444,445,446,447,448,449,450,451,452,453,454,455,456,457,458,459,460,461,462,463,464,465,466,467,468,469,470,471,472,473,474,475,476,477,478,479,480,481,482,483,484,485,486,487,488,489,490,491,492,493,494,495,496,497,498,499,500,501,502,503,504,505,506,507,508,509,510,511,512,513,514,515,516,517,518,519,520,521,522,523,524,527,528,529,530,531,532,533,534,535,536,537,538,539,540,541,542,543,544,545,546,547,548,549,550,551,552,553,554,555]
# production_runs = [6,7,8,10,11,12,13,14,15,16,17,31,32,33,34,35,36]
# production_runs = [88]
xdist = []
ydist = []
xdist250 = []
ydist250 = []
xdist500 = []
ydist500 = []
total_counts = []
for run in tqdm(production_runs):
    prevxdist = []
    settings_path = "/mnt/daqtesting/protondet2024/h5/run_%04d/run_%04dp10_2000torr/config.gui_ini"%(run,run)
    data_file_path = "/mnt/daqtesting/protondet2024/h5/run_%04d.h5"%(run)
    flat_lookup_path = "/mnt/daqtesting/protondet2024/h5/run_%04d/run_%04dp10_2000torr/flatlookup2cobos.csv"%(run,run)
    
    counts = np.load("/mnt/daqtesting/protondet2024/h5/run_%04d/run_%04dp10_2000torr/counts.npy"%(run,run))
    angles = np.load("/mnt/daqtesting/protondet2024/h5/run_%04d/run_%04dp10_2000torr/angles.npy"%(run,run))
    vetos = np.load("/mnt/daqtesting/protondet2024/h5/run_%04d/run_%04dp10_2000torr/veto.npy"%(run,run))
    data = raw_h5_file.raw_h5_file(file_path=data_file_path, flat_lookup_csv=flat_lookup_path)

    config = configparser.ConfigParser()
    config.read(settings_path)
    data.length_ic_threshold = config.get('ttk.Entry','length_ic_threshold')
    data.energy_ic_threshold = config.get('ttk.Entry','energy_ic_threshold')
    data.view_threshold = config.get('ttk.Entry','view_threshold')
    data.include_cobos = config.get('ttk.Entry','include_cobos')
    data.include_asads = config.get('ttk.Entry','include_asads')
    data.include_pads = config.get('ttk.Entry','include_pads')
    data.veto_threshold = config.get('ttk.Entry','veto_threshold')
    data.range_min = config.get('ttk.Entry','range_min')
    data.range_max = config.get('ttk.Entry','range_max')
    data.min_ic = config.get('ttk.Entry','min_ic')
    data.max_ic = config.get('ttk.Entry','max_ic')
    data.angle_min = config.get('ttk.Entry','angle_min')
    data.angle_max = config.get('ttk.Entry','angle_max')
    data.background_bin_start = config.get('ttk.Entry','background_bin_start')
    data.background_bin_stop = config.get('ttk.Entry','background_bin_stop')
    data.zscale = float(config.get('ttk.Entry','zscale'))
    data.near_peak_window_width = config.get('ttk.Entry','near_peak_window_width')
    data.peak_first_allowed_bin = config.get('ttk.Entry','peak_first_allowed_bin')
    data.peak_last_allowed_bin = config.get('ttk.Entry','peak_last_allowed_bin')
    data.peak_mode = config.get('ttk.OptionMenu','peak_mode')
    data.background_mode = config.get('ttk.OptionMenu','background_mode')
    data.remove_outliers = config.get('ttk.CheckButton','remove_outliers')

    for event in range(len(angles)):
        total_counts.append(counts[event])
        if (angles[event] < 80 or vetos[event] > 300 or counts[event] < 4e5 or counts[event] > 6e5):
            continue
        if (len(prevxdist) >= 100):
            break
        else:
            evt_data = data.get_data(event)
            image = np.zeros(np.shape(data.pad_plane))
            for line in evt_data:
                chnl_info = tuple(line[0:4])
                if chnl_info not in data.chnls_to_pad:
                    print('warning: the following channel tripped but doesn\'t have  a pad mapping: '+str(chnl_info))
                    continue
                pad = data.chnls_to_pad[chnl_info]
                if pad not in VETO_PADS:
                    x,y = data.pad_to_xy_index[pad]
                    # x,y = data.chnls_to_xy_coord[chnl_info]
                    # print(data.chnls_to_xy_coord[chnl_info])
                    image[x,y] = np.sum(line[FIRST_DATA_BIN:])
            image[image<0]=0
            y_coords, x_coords = np.indices(image.shape)
            x_centroid = np.sum(x_coords * image) / np.sum(image)
            y_centroid = np.sum(y_coords * image) / np.sum(image)
            prevxdist.append(x_centroid)

            if run < 126:
                xdist250.append(x_centroid)
                ydist250.append(y_centroid)
            if run >= 126:
                xdist500.append(x_centroid)
                ydist500.append(y_centroid)
            xdist.append(x_centroid)
            ydist.append(y_centroid)
            fig, ax = plt.subplots()
            ax.imshow(image,cmap = 'gray')
            ax.plot(x_centroid, y_centroid, 'r+', markersize=15)
            ax.set_title(f'Image Centroid of Event {event}, Run{run}: ({x_centroid:.2f}, {y_centroid:.2f})\n{angles[event]} deg')
            print(f"Calculated Centroid: x={x_centroid}, y={y_centroid}")
            plt.show()

np.savetxt('xdist_nocathode_cm.csv',xdist)
np.savetxt('ydist_nocathode_cm.csv',ydist)

plt.hist(total_counts, 200)
plt.show()

plt.hist2d(xdist,ydist,50)
plt.xlabel("X-Value (mm)")
plt.ylabel("Y-Value (mm)")
plt.colorbar()
plt.show()

plt.hist2d(xdist250,ydist250,50)
plt.xlabel("X-Value (mm)")
plt.ylabel("Y-Value (mm)")
plt.colorbar()
plt.show()

plt.hist2d(xdist500,ydist500,50)
plt.xlabel("X-Value (mm)")
plt.ylabel("Y-Value (mm)")
plt.colorbar()
plt.show()


plt.hist(xdist, bins=100, edgecolor='black', alpha=0.7)
plt.xlabel("X-Centroid")
plt.ylabel("Counts")
plt.title("Histogram of X-Distribution")
plt.show()                
plt.hist(ydist, bins=100, edgecolor='black', alpha=0.7)
plt.xlabel("Y-Centroid")
plt.ylabel("Counts")
plt.title("Histogram of Y-Distribution")
plt.show()
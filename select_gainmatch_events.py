import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from tqdm import tqdm
import h5py
from raw_viewer import raw_h5_file

# production_runs = [6,7,8,10,11,12,13,14,15,16,17,31,32,33,34,35,36,37,38,39,40,41,42,43,44,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,73,74,75,77,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98,99,100,101,102,103,104,105,106,107,108,126,133,134,150,151,152,153,154,155,156,157,158,159,160,161,164,165,166,167,168,169,170,171,172,173,174,175,176,177,178,179,180,181,182,183,184,185,186,187,188,189,190,191,192,194,195,196,197,198,199,200,201,202,203,204,205,206,207,208,209,210,211,212,213,214,215,216,217,218,219,220,221,222,223,224,225,226,227,228,229,230,231,232,233,234,235,236,237,238,239,240,241,242,243,245,246,247,248,249,250,251,252,253,254,255,256,257,258,259,260,261,262,263,264,265,266,267,268,269,270,271,272,273,274,275,276,278,279,280,281,282,283,284,285,286,287,288,289,290,291,292,293,294,295,296,297,298,299,300,301,302,304,305,306,307,308,309,310,311,312,313,314,315,316,317,318,319,320,321,322,323,324,325,326,327,328,329,330,331,333,334,335,336,337,338,339,340,341,342,343,344,345,346,347,348,349,350,351,352,353,354,355,356,357,358,359,360,361,362,363,364,365,366,367,368,369,370,371,372,373,374,375,376,377,378,379,380,381,382,383,384,385,386,387,388,389,390,391,392,393,394,395,396,397,398,399,400,401,402,403,404,405,406,407,408,409,410,411,412,413,414,415,416,417,418,419,420,421,422,423,424,425,426,427,428,429,430,431,432,433,434,435,436,437,438,439,440,441,442,443,444,445,446,447,448,449,450,451,452,453,454,455,456,457,458,459,460,461,462,463,464,465,466,467,468,469,470,471,472,473,474,475,476,477,478,479,480,481,482,483,484,485,486,487,488,489,490,491,492,493,494,495,496,497,498,499,500,501,502,503,504,505,506,507,508,509,510,511,512,513,514,515,516,517,518,519,520,521,522,523,524,527,528,529,530,531,532,533,534,535,536,537,538,539,540,541,542,543,544,545,546,547,548,549,550,551,552,553,554,555]
production_runs = [121,122,123,124,125,126,127,128]

total_counts = []
total_ranges = []
total_veto =[]
dxy = []
dt = []

zscale = 0.65
zscale = 0.9  # This is the zscale used in the proton-alpha data set

events_added_to_pad_data_h5 = 0
total_events_to_add = 3007
total_events_to_add = 20000 # total events in the 700 keV proton peak in runs 121-128 (0.02422617557 * (112505+113693+116903+123833+115993+115272+87885+62684)) <- Percent of events in proton blob in run 124, times the total number of events in the runs 121-128, which is 112505+113693+116903+123833+115993+115272+87885+62684 = 848,768 events

# This commented out code is for the double alpha data set

# with h5py.File("/mnt/daqtesting/protondet2024/pad_gainmatch_events.h5", "w") as dest_h5:
#     meta = dest_h5.create_group('meta')
#     get = dest_h5.create_group('get')
#     # THIS ARRAY NEEDS TO BE CHANGED MAUALLY TO MATCH THE NUMBER OF EVENTS TO BE WRITTEN TO THE h5 FILE
#     arr = [0,0,total_events_to_add,1e20]
#     meta.create_dataset('meta',data=arr)
#     for run in production_runs:
#         # print("Currently on Run %d"%run)
#         length = []
#         total_counts = np.load('/mnt/daqtesting/protondet2024/h5/run_%04d/run_%04dp10_2000torr/counts.npy'%(run,run))
#         total_veto = np.load('/mnt/daqtesting/protondet2024/h5/run_%04d/run_%04dp10_2000torr/veto.npy'%(run,run))
#         dxy = np.load('/mnt/daqtesting/protondet2024/h5/run_%04d/run_%04dp10_2000torr/dxy.npy'%(run,run))
#         dt = np.load('/mnt/daqtesting/protondet2024/h5/run_%04d/run_%04dp10_2000torr/dt.npy'%(run,run))
#         for event in range(len(dxy)):
#             length.append((dxy[event]**2 + zscale*dt[event]**2)**0.5)
#         total_ranges = length
            
#         h5file = raw_h5_file.raw_h5_file("/mnt/daqtesting/protondet2024/h5/run_%04d.h5"%run, 
#                                         zscale = 0.65, 
#                                         flat_lookup_csv = "/mnt/projects/e21072/OfflineAnalysis/analysis_scripts/joe/gadget_analysis/raw_viewer/channel_mappings/flatlookup2cobos.csv")
#         # process settings are set the same as p10_2000torr config settings
#         h5file.length_ic_threshold = 100
#         h5file.ic_counts_threshold = 25
#         h5file.view_threshold = 100
#         h5file.include_cobos = all
#         h5file.include_asads = all
#         h5file.include_pads = all
#         h5file.veto_threshold = 300
#         h5file.range_min = 1
#         h5file.range_max = np.inf
#         h5file.min_ic = 1
#         h5file.max_ic = np.inf
#         h5file.angle_min = 0
#         h5file.angle_max = 90
#         h5file.background_bin_start = 400
#         h5file.background_bin_stop = 500
#         h5file.zscale = .65
#         h5file.near_peak_window_width = 50
#         h5file.peak_first_allowed_bin = -np.inf
#         h5file.peak_last_allowed_bin = np.inf
#         h5file.peak_mode = 'near peak'
#         h5file.background_subtract_mode = 'fixed window'
#         h5file.data_select_mode = 'near peak'
#         h5file.remove_outliers = 1
#         h5file.num_background_bins = (400,500)

#         # i wrote a new method for h5 files that returns a list of the integrated charges on each pad called ic_of_pads
#         for i in range(len(total_counts)):
#             # if event is in the energy range, add the array of integrated charge for each pad to the h5 file
#             if total_counts[i]>5.756e5 and total_counts[i]<8.54e5 and total_ranges[i]>40:
#                 ic_of_pads = h5file.ic_of_pads(i)
#                 print(np.sum(ic_of_pads))
#                 get.create_dataset('evt%d_pad_data'%(events_added_to_pad_data_h5), data=ic_of_pads)
#                 events_added_to_pad_data_h5 += 1
#                 if events_added_to_pad_data_h5 >= total_events_to_add:
#                     break
#                 print(events_added_to_pad_data_h5)
#         if events_added_to_pad_data_h5 >= total_events_to_add:
#             break
#         total_counts = []
#         total_veto = []
#         dxy = []
#         dt = []
#         total_ranges = []


# This code is for the proton-alpha data set, run 124
with h5py.File("/egr/research-tpc/dopferjo/run_121-128_proton_pad_gainmatch_events.h5", "w") as dest_h5:
    meta = dest_h5.create_group('meta')
    get = dest_h5.create_group('get')
    # THIS ARRAY NEEDS TO BE CHANGED MANUALLY TO MATCH THE NUMBER OF EVENTS TO BE WRITTEN TO THE h5 FILE
    arr = [0,0,total_events_to_add,1e20]
    meta.create_dataset('meta',data=arr)
    for run in production_runs:

        # print("Currently on Run %d"%run)
        length = []
        total_counts = np.load('/egr/research-tpc/shared/Run_Data/run_%04d_raw_viewer/run_%04dsmart/counts.npy'%(run,run))
        total_veto = np.load('/egr/research-tpc/shared/Run_Data/run_%04d_raw_viewer/run_%04dsmart/veto.npy'%(run,run))
        dxy = np.load('/egr/research-tpc/shared/Run_Data/run_%04d_raw_viewer/run_%04dsmart/dxy.npy'%(run,run))
        dt = np.load('/egr/research-tpc/shared/Run_Data/run_%04d_raw_viewer/run_%04dsmart/dt.npy'%(run,run))
        for event in range(len(dxy)):
            length.append((dxy[event]**2 + zscale*dt[event]**2)**0.5)
        total_ranges = length
            
        h5file = raw_h5_file.raw_h5_file("/egr/research-tpc/shared/Run_Data/run_%04d.h5"%run, 
                                        zscale = zscale, 
                                        flat_lookup_csv = "/egr/research-tpc/dopferjo/gadget_analysis/raw_viewer/channel_mappings/flatlookup4cobos.csv")
        f = h5py.File("/egr/research-tpc/shared/Run_Data/run_%04d.h5"%run, 'r')
        first_event, last_event = int(f['meta']['meta'][0]), int(f['meta']['meta'][2])
  
        print("First event: %d, Last event: %d"%(first_event, last_event))

        # process settings are set the same as smart config settings
        h5file.length_ic_threshold = 100
        h5file.ic_counts_threshold = 9
        h5file.view_threshold = 100
        h5file.include_cobos = all
        h5file.include_asads = all
        h5file.include_pads = all
        h5file.veto_threshold = 300
        h5file.range_min = 1
        h5file.range_max = np.inf
        h5file.min_ic = 1
        h5file.max_ic = np.inf
        h5file.angle_min = 0
        h5file.angle_max = 90
        h5file.background_bin_start = 160
        h5file.background_bin_stop = 250
        h5file.zscale = .9
        h5file.background_start_entry = 160
        h5file.background_stop_entry = 250
        h5file.exclude_width_entry = 20
        h5file.include_width_entry = 40
        h5file.near_peak_window_entry = 50
        h5file.near_peak_window_width = 50
        h5file.peak_first_allowed_bin_entry = -np.inf
        h5file.peak_last_allowed_bin_entry = np.inf
        h5file.peak_first_allowed_bin = -np.inf
        h5file.peak_last_allowed_bin = np.inf
        h5file.peak_mode = 'all data'
        h5file.background_subtract_mode = 'smart'
        h5file.data_select_mode = 'all data'
        h5file.remove_outliers = 1
        h5file.num_background_bins = (160,250)

        # i wrote a new method for h5 files that returns a list of the integrated charges on each pad called ic_of_pads
        for i in range(len(total_counts)):
            # if event is in the energy range, add the array of integrated charge for each pad to the h5 file
            # this cut is for the high energy alpha peak, the O-16 + He-4 alpha peak
            # if total_counts[i]>6.02e5 and total_counts[i]<7.22e5 and total_ranges[i]>20 and total_ranges[i]<50:
            
            # this cut is for the 1600 keV proton peak
            if total_counts[i]>1.67e5 and total_counts[i]<2.15e5 and total_ranges[i]>24 and total_ranges[i]<62:
                ic_of_pads = h5file.ic_of_pads(first_event + i)
                print("Event %d: Total IC of pads: %d"%(first_event + i, np.sum(ic_of_pads)))
                get.create_dataset('evt%d_pad_data'%(events_added_to_pad_data_h5), data=ic_of_pads)
                events_added_to_pad_data_h5 += 1
                if events_added_to_pad_data_h5 >= total_events_to_add:
                    break
                print("Events added to pad data h5: %d"%events_added_to_pad_data_h5)
        if events_added_to_pad_data_h5 >= total_events_to_add:
            break
        total_counts = []
        total_veto = []
        dxy = []
        dt = []
        total_ranges = []
        del f

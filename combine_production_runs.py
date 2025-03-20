import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from tqdm import tqdm

production_runs = [6,7,8,10,11,12,13,14,15,16,17,31,32,33,34,35,36,37,38,39,40,41,42,43,44,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,73,74,75,77,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98,99,100,101,102,103,104,105,106,107,108,126,133,134,150,151,152,153,154,155,156,157,158,159,160,161,164,165,166,167,168,169,170,171,172,173,174,175,176,177,178,179,180,181,182,183,184,185,186,187,188,189,190,191,192,194,195,196,197,198,199,200,201,202,203,204,205,206,207,208,209,210,211,212,213,214,215,216,217,218,219,220,221,222,223,224,225,226,227,228,229,230,231,232,233,234,235,236,237,238,239,240,241,242,243,245,246,247,248,249,250,251,252,253,254,255,256,257,258,259,260,261,262,263,264,265,266,267,268,269,270,271,272,273,274,275,276,278,279,280,281,282,283,284,285,286,287,288,289,290,291,292,293,294,295,296,297,298,299,300,301,302,304,305,306,307,308,309,310,311,312,313,314,315,316,317,318,319,320,321,322,323,324,325,326,327,328,329,330,331,333,334,335,336,337,338,339,340,341,342,343,344,345,346,347,348,349,350,351,352,353,354,355,356,357,358,359,360,361,362,363,364,365,366,367,368,369,370,371,372,373,374,375,376,377,378,379,380,381,382,383,384,385,386,387,388,389,390,391,392,393,394,395,396,397,398,399,400,401,402,403,404,405,406,407,408,409,410,411,412,413,414,415,416,417,418,419,420,421,422,423,424,425,426,427,428,429,430,431,432,433,434,435,436,437,438,439,440,441,442,443,444,445,446,447,448,449,450,451,452,453,454,455,456,457,458,459,460,461,462,463,464,465,466,467,468,469,470,471,472,473,474,475,476,477,478,479,480,481,482,483,484,485,486,487,488,489,490,491,492,493,494,495,496,497,498,499,500,501,502,503,504,505,506,507,508,509,510,511,512,513,514,515,516,517,518,519,520,521,522,523,524,527,528,529,530,531,532,533,534,535,536,537,538,539,540,541,542,543,544,545,546,547,548,549,550,551,552,553,554,555]
# production_runs = [155]
total_counts = []
total_ranges = []
total_veto =[]
dxy = []
dt = []

zscale = 0.65

for run in tqdm(production_runs):
    if run > 140:
        # print("Currently on Run %d"%run)
        length = []
        total_counts = np.concatenate([total_counts, np.load('/mnt/daqtesting/protondet2024/h5/run_%04d/run_%04dp10_2000torr/counts.npy'%(run,run))])
        total_veto = np.concatenate([total_veto, np.load('/mnt/daqtesting/protondet2024/h5/run_%04d/run_%04dp10_2000torr/veto.npy'%(run,run))])
        dxy = np.load('/mnt/daqtesting/protondet2024/h5/run_%04d/run_%04dp10_2000torr/dxy.npy'%(run,run))
        dt = np.load('/mnt/daqtesting/protondet2024/h5/run_%04d/run_%04dp10_2000torr/dt.npy'%(run,run))
        # total_counts = np.concatenate([total_counts, np.load('/mnt/daqtesting/protondet2024/interesting_events_without_run_number_in_event_name_without_event_447/interesting_events_without_run_number_in_event_name_without_event_447p10_2000torr/counts.npy')])
        # total_veto = np.concatenate([total_veto, np.load('/mnt/daqtesting/protondet2024/interesting_events_without_run_number_in_event_name_without_event_447/interesting_events_without_run_number_in_event_name_without_event_447p10_2000torr/veto.npy')])
        # dxy = np.load('/mnt/daqtesting/protondet2024/interesting_events_without_run_number_in_event_name_without_event_447/interesting_events_without_run_number_in_event_name_without_event_447p10_2000torr/dxy.npy')
        # dt = np.load('/mnt/daqtesting/protondet2024/interesting_events_without_run_number_in_event_name_without_event_447/interesting_events_without_run_number_in_event_name_without_event_447p10_2000torr/dt.npy')
        for event in range(len(dxy)):
            length.append((dxy[event]**2 + zscale*dt[event]**2)**0.5)
        total_ranges = np.concatenate([total_ranges,length])

# Create 6 MeV mask for plotting
mask = np.logical_and.reduce((total_veto<np.inf,
                              total_counts<6.34e5,
                              total_counts>4.14e5,
                              total_ranges>26,
                              total_ranges<39
                              ))
# 8 MeV Mask
# mask = np.logical_and.reduce((total_veto<150,
#                               total_counts<8.545e5,
#                               total_counts>5.756e5,
#                               total_ranges>40,
#                               total_ranges<50
#                               ))
# No mask
mask = np.logical_and.reduce((total_veto<300,
                              total_counts<np.inf,
                              total_counts>-np.inf,
                              total_ranges>-np.inf,
                              total_ranges<np.inf
                              ))

print('Total events: ',len(total_counts[mask]))
# print(type(total_counts))
# print(type(total_ranges))
plt.figure(0)
plt.title('RvE for Runs above 150')
plt.hist2d(total_counts[mask], total_ranges[mask], 300, norm=mpl.colors.LogNorm(), range=[[0,1.5e6],[0,200]])
# plt.hist(total_counts[mask], bins=200)
plt.colorbar()
plt.xlabel('Energy (ADC counts)')
plt.ylabel('Range (mm)')
plt.show()
# plt.savefig('RvE_for_all_runs.png')

# plt.figure(1)
# plt.title('Ranges for Production Runs Up to run_0053')
# plt.hist(total_ranges[mask], bins=80)
# plt.xlabel('Range (mm)')
# plt.ylabel('Counts')
# plt.show()

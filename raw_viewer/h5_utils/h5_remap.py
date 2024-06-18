import shutil

import h5py
import numpy as np
import tqdm


def remap_pad_numbers(input_file, output_file, flatlookup_file):
    '''
    Copy input file to output file, and then renumber the pads based on flatlookup_file
    '''
    flat_lookup = np.loadtxt(flatlookup_file, delimiter=',', dtype=int)
    chnls_to_pad = {} #maps tuples of (asad, aget, channel) to pad number
    for line in flat_lookup:
        chnls = tuple(line[0:4])
        pad = line[4]
        chnls_to_pad[chnls] = pad
    
    shutil.copyfile(input_file, output_file)
    with h5py.File(output_file, 'r+') as file:
        first_event_num, last_event_num = int(file['meta']['meta'][0]), int(file['meta']['meta'][2])
        for event_num in tqdm.tqdm(range(first_event_num, last_event_num+1)):
            data = file['get']['evt%d_data'%event_num]
            for line_num in range(len(data)):
                chnl_info = tuple(data[line_num, 0:4])
                if chnl_info in chnls_to_pad:
                    data[line_num, 4] = chnls_to_pad[chnl_info]
                else:
                    print('Event #' + str(event_num) + ' channel information ' + str(chnl_info) + ' not in pad mapping!')

if __name__ == '__main__':
    input_file = '/mnt/analysis/e21072/gastest_h5_files/run_0344.h5'
    output_file = '/mnt/analysis/e21072/h5test/run_9999.h5'

    #input_file = '/mnt/analysis/e21072/gastest_h5_files/run_0020.h5'
    #output_file = '/mnt/analysis/e21072/gastest_h5_files/run_0020_remapped.h5'
    remap_pad_numbers(input_file, output_file, 'flatlookup2cobos.csv')

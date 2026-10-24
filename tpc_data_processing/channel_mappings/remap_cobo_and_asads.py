'''
This file can be used to make a new pad lookup table from an old one.

remap_map defines this mapping by (old_cobo, old_asad)->(new_cobo, new_asad)
input_file is the channel mapping to remap, and output_file is the new map to create
'''
import numpy as np

input_path = 'flatlookup2cobos.csv'
output_path = 'flatlookup1cobos.csv'


remap_map = {(0,0):(0,0),
             (0,1):(0,1),
             (1,0):(0,2),
             (1,1):(0,3)}

with open(input_path)as input_file, open(output_path, 'w') as output_file:
    for line in input_file:
        if len(line) == 0:
            continue
        cobo, asad, aget, chnl, pad = np.fromstring(line, sep=',')
        cobo,asad = remap_map[cobo,asad]
        output_file.write('%d, %d, %d, %d, %d\n'%(cobo, asad, aget, chnl, pad))
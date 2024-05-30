'''
For use with old h5 files, such as those
stored in /mnt/analysis/e17023/alphadata_h5/
'''
import os
import numpy as np
import raw_viewer.raw_h5_file as raw_h5_file

class heritage_h5_file(raw_h5_file.raw_h5_file):
    def __init__(self, file_path):
        flat_lookup_path = os.path.join(os.path.dirname(__file__), 'channel_mappings/flatlookup2cobos.csv')
        raw_h5_file.raw_h5_file.__init__(self,file_path, flat_lookup_csv=flat_lookup_path)
        self.xy_to_pad = {tuple(self.padxy[pad]):pad for pad in range(len(self.padxy))}
        self.xy_to_chnls = {tuple(self.chnls_to_xy_coord[chnls]):chnls 
                            for chnls in self.chnls_to_xy_coord}
        
    
    def get_data(self, event_number):
        event_str = 'Event_[%d]'%event_number
        event = self.h5_file[event_str]
        data = []
        for x,y,t,A in zip(event['x'], event['y'], event['t'], event['A']):
            #find nearest pad
            best_xy = (np.inf, np.inf)
            best_dist = np.inf
            for xy in self.xy_to_pad:
                dist = np.array(xy) - np.array([x,y])
                dist = np.dot(dist, dist)
                if dist < best_dist:
                    best_dist = dist
                    best_xy = xy
            xy = best_xy
            data.append(np.zeros(517))
            data[-1][0:4] = self.xy_to_chnls[xy]
            data[-1][4] = self.xy_to_pad[xy]
            data[-1][t] = A
        return np.array(data)
            
    def get_event_num_bounds(self):
        first = 0
        while 'Event_[%d]'%first not in self.h5_file:
            first += 1
        last = first
        while 'Event_[%d]'%last in self.h5_file:
            last += 1
        return first, last 
'''
For use with old h5 files, such as those
stored in /mnt/analysis/e17023/alphadata_h5/
'''
import os
import numpy as np
from numpy.core import inf
import raw_viewer.raw_h5_file as raw_h5_file

class heritage_h5_file(raw_h5_file.raw_h5_file):
    def __init__(self, file_path):
        flat_lookup_path = os.path.join(os.path.dirname(__file__), 'channel_mappings/flatlookup2cobos.csv')
        raw_h5_file.raw_h5_file.__init__(self,file_path, flat_lookup_csv=flat_lookup_path)
        
        
    
    def get_data(self, event_number):
        event_str = 'Event_[%d]'%event_number
        event = self.h5_file[event_str]
        data = np.zeros((len(event['x']),517))
        for i,x,y,t,A in zip(range(len(event['x'])),event['x'], event['y'], event['t'], event['A']):
            xy=(x,y)
                
            data[i][0:4] = self.get_chnl_from_xy(xy)
            data[i][4] = self.get_pad_from_xy(xy)
            data[i][t] = A

        return np.array(data)

    def get_nearest_xy(self, xy):
        xy = tuple(np.round(xy, 1))
        if xy not in self.xy_to_chnls:
                #find nearest pad
                best_xy = (np.inf, np.inf)
                best_dist = np.inf
                for x,y in self.xy_to_pad:
                    dist = np.array(xy) - np.array([x,y])
                    dist = np.dot(dist, dist)
                    if dist < best_dist:
                        best_dist = dist
                        best_xy = (x,y)
                self.xy_to_chnls[xy] = self.xy_to_chnls[best_xy]
                self.xy_to_pad[xy] = self.xy_to_pad[best_xy]
                print('(%f,%f) remmapped to (%f,%f)'%(*xy, *best_xy))
        return xy
    
    def get_pad_from_xy(self, xy):
        xy = self.get_nearest_xy(xy)
        return self.xy_to_pad[xy]
    
    def get_chnl_from_xy(self, xy):
        xy = self.get_nearest_xy(xy)
        return self.xy_to_chnls[xy]

    def get_xyte(self, event_number, threshold=-np.inf, include_veto_pads=True):
        '''
        return only x,y,t,A pairs from h5 file
        '''
        event_str = 'Event_[%d]'%event_number
        event = self.h5_file[event_str]
        return event['x'], event['y'], event['t'], event['A']

    def get_xyze(self, event_number, threshold=-np.inf, include_veto_pads=True):
        '''
        return only x,y,z,A pairs from h5 file
        '''
        event_str = 'Event_[%d]'%event_number
        event = self.h5_file[event_str]
        return event['x'], event['y'], event['z'], event['A']

    def get_event_num_bounds(self):
        first = 0
        while 'Event_[%d]'%first not in self.h5_file:
            first += 1
        last = first
        while 'Event_[%d]'%last in self.h5_file:
            last += 1
        return first, last 
    
    
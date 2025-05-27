import matplotlib.pylab as plt
import matplotlib.colors as colors
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
class event_display:
    def __init__(self, pad_plane, pad_to_xy_index, chnl_to_pad, veto_pads, data_select_mode='all data', background_subtract_mode='average'):
        self.pad_plane = pad_plane
        self.pad_to_xy_index = pad_to_xy_index
        self.chnls_to_pad = chnl_to_pad
        self.veto_pads = veto_pads
        self.data_select_mode = data_select_mode
        self.background_subtract_mode = background_subtract_mode
        self.pad_backgrounds = {}
        self.cmap = plt.get_cmap('viridis')

        #color map for plotting
        cdict={'red':  ((0.0, 0.0, 0.0),
                    (0.25, 0.0, 0.0),
                    (0.5, 0.8, 1.0),
                    (0.75, 1.0, 1.0),
                    (1.0, 0.4, 1.0)),

            'green': ((0.0, 0.0, 0.0),
                    (0.25, 0.0, 0.0),
                    (0.5, 0.9, 0.9),
                    (0.75, 0.0, 0.0),
                    (1.0, 0.0, 0.0)),

            'blue':  ((0.0, 0.0, 0.4),
                    (0.25, 1.0, 1.0),
                    (0.5, 1.0, 0.8),
                    (0.75, 0.0, 0.0),
                    (1.0, 0.0, 0.0))
            }
        # cdict['alpha'] = ((0.0, 0.0, 0.0),
        #                 (0.3,0.2, 0.2),
        #                 (0.8,1.0, 1.0),
        #                 (1.0, 1.0, 1.0))
        self.cmap = LinearSegmentedColormap('test',cdict)

    def show_pad_backgrounds(self, fig_name=None, block=True):
        ave_image = np.zeros(np.shape(self.pad_plane))
        std_image = np.zeros(np.shape(self.pad_plane))
        for pad in self.pad_backgrounds:
            x,y = self.pad_to_xy_index[pad]
            ave, std = self.pad_backgrounds[pad]
            ave_image[x,y] = ave
            std_image[x,y] = std

        fig=plt.figure(fig_name)
        plt.clf()
        ave_ax = plt.subplot(1,2,1)
        ave_ax.set_title('average counts')
        ave_shown = ave_ax.imshow(ave_image, cmap=self.cmap)
        fig.colorbar(ave_shown, ax=ave_ax)

        std_ax = plt.subplot(1,2,2)
        std_ax.set_title('standard deviation')
        std_shown=std_ax.imshow(std_image, cmap=self.cmap)
        fig.colorbar(std_shown, ax=std_ax)
        #plt.colorbar(ax=std_plot)
        #plt.colorbar())
        plt.show(block=block)

    def plot_traces(self, event_num, block=True, fig_name=None):
        '''
        Note: veto pads are plotted as dotted lines
        '''
        plt.figure(fig_name)
        plt.clf()
        pads, pad_data = self.get_pad_traces(event_num)
        for pad, data in zip(pads, pad_data):
            r = pad/1024*.8
            g = (pad%512)/512*.8
            b = (pad%256)/256*.8
            if pad in VETO_PADS:
                plt.plot(data, '--', color=(r,g,b), label='%d'%pad)
            else:
                plt.plot(data, color=(r,g,b), label='%d'%pad)
        plt.legend(loc='upper right')
        plt.show(block=block)

    def plot_3d_traces(self, event_num, threshold=-np.inf, block=True, fig_name=None):
        fig = plt.figure(fig_name, figsize=(6,6))
        plt.clf()
        ax = plt.axes(projection='3d')
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        ax.set_xlim3d(-200, 200)
        ax.set_ylim3d(-200, 200)
        ax.set_zlim3d(0, 400)

        xs, ys, zs, es = self.get_xyze(event_num, threshold=threshold)

        #TODO: make generic, these are P10 values
        calib_point_1 = (0.806, 156745)
        calib_point_2 = (1.679, 320842)
        energy_1, channel_1 = calib_point_1
        energy_2, channel_2 = calib_point_2
        energy_scale_factor = (energy_2 - energy_1) / (channel_2 - channel_1)
        energy_offset = energy_1 - energy_scale_factor * channel_1

        ax.view_init(elev=45, azim=45)
        ax.scatter(xs, ys, zs, c=es, cmap=self.cmap)
        cbar = fig.colorbar(ax.get_children()[0])
        max_veto_counts, dxy, dz, energy, angle, pads_railed = self.process_event(event_num)
        length = np.sqrt(dxy**2 + dz**2)
        plt.title('event %d, total counts=%d / %f MeV\n length=%f mm, angle=%f deg\n # pads railed=%d'%(event_num, energy, 
                                                                                                energy*energy_scale_factor + energy_offset, length,
                                                                                                np.degrees(angle), len(pads_railed)))
        plt.show(block=block)

    def show_2d_projection(self, event_number, block=True, fig_name=None):
        data = self.get_data(event_number)
        image = np.zeros(np.shape(self.pad_plane))
        for line in data:
            chnl_info = tuple(line[0:4])
            if chnl_info not in self.chnls_to_pad:
                print('warning: the following channel tripped but doesn\'t have  a pad mapping: '+str(chnl_info))
                continue
            pad = self.chnls_to_pad[chnl_info]
            x,y = self.pad_to_xy_index[pad]
            image[x,y] = np.sum(line[FIRST_DATA_BIN:])
        image[image<0]=0
        trace = np.sum(data[:,FIRST_DATA_BIN:],0)
        

        fig = plt.figure(fig_name, figsize=(6,6))
        plt.clf()
        should_veto, dxy, dz, energy, angle, pads_railed_list = self.process_event(event_number)
        length = np.sqrt(dxy**2 + dz**2)
        plt.title('event %d, total counts=%d, length=%f mm, angle=%f, veto=%d'%(event_number, energy, length, np.degrees(angle), should_veto))
        plt.subplot(2,1,1)
        plt.imshow(image, norm=colors.LogNorm())
        plt.colorbar()
        plt.subplot(2,1,2)
        plt.plot(trace)
        plt.show(block=block)

    def show_traces_w_baseline_estimate(self, event_num, block=True, fig_name=None):
        '''
        plots traces without background subtraction, with backgrounds shown as ... lines
        '''
        plt.figure(fig_name)
        plt.clf()
        old_background_mode = self.background_subtract_mode
        self.background_subtract_mode = 'none' #will set back after drawing traces
        old_mode = self.data_select_mode
        self.data_select_mode = 'all data'
        pads, pad_data = self.get_pad_traces(event_num)
        for pad, data in zip(pads, pad_data):
            r = pad/1024*.8
            g = (pad%512)/512*.8
            b = (pad%256)/256*.8
            if pad in VETO_PADS:
                plt.plot(data, '--', color=(r,g,b), label='%d'%pad)
            else:
                plt.plot(data, color=(r,g,b), label='%d'%pad)
        self.background_subtract_mode = old_background_mode
        for pad, data in zip(pads, pad_data):
            r = pad/1024*.8
            g = (pad%512)/512*.8
            b = (pad%256)/256*.8
            plt.plot(self.calculate_background(data), '.', color=(r,g,b), label='%d baseline'%pad)
        self.data_select_mode = old_mode
        plt.legend(loc='upper right')
        plt.show(block=block)

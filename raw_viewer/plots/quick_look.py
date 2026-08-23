import os
import pathlib

import numpy as np
import matplotlib
import matplotlib.pylab as plt
from matplotlib.backends.backend_pdf import PdfPages
from tqdm import tqdm

from raw_viewer import process_runs
from raw_viewer.raw_h5_file import VETO_PADS

def save_images(experiment, runs, config_filename='smart2_rpr.csv'):
    if not isinstance(runs, (list, tuple, np.ndarray)):
        runs = [runs]
        
    for run in tqdm(runs, desc="Processing runs"):
        run = int(run)
        num_to_save = 10
        h5file = process_runs.get_h5_file(experiment, run, config_filename)
        h5file.cache_enable = False
        
        # save x-y projections for first, last, and random num_to_save images
        first, last = h5file.get_event_num_bounds()
        actual_num_to_save = min(num_to_save, max(1, (last - first + 1) // 3))
            
        first_events = np.arange(first, first+actual_num_to_save)
        last_events = np.arange(last-actual_num_to_save+1, last+1)
        random_events = np.random.randint(first, last, actual_num_to_save)
        events_to_save = np.unique(np.concatenate([first_events, random_events, last_events]))
        
        im_save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'quick_look_pdf', experiment)
        os.makedirs(im_save_path, exist_ok=True)
        
        pdf_path = os.path.join(im_save_path, f'run{run}_quick_look.pdf')
        with PdfPages(pdf_path) as pdf:
            for event_num in tqdm(events_to_save, desc=f"Run {run} events", leave=False):
                # Get data for 2D image
                pads, traces = h5file.get_pad_traces(event_num)
                trace_dict = {pad: trace for pad, trace in zip(pads, traces)}
                data = {pad: np.sum(trace_dict[pad]) for pad in trace_dict}
                
                # Event info
                should_veto, dxy, dz, energy, angle, pads_railed_list = h5file.process_event(event_num)
                length = np.sqrt(dxy**2 + dz**2)
                title = f'Run {run} Event {event_num}, counts={energy:.0f}, length={length:.2f} mm, angle={np.degrees(angle):.2f}, veto={should_veto}'
                
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)
                fig.suptitle(title)
                
                # 2D projection
                image = h5file.get_2d_image(data)
                cmap = matplotlib.colormaps['viridis'].copy() if hasattr(matplotlib, 'colormaps') else matplotlib.cm.get_cmap('viridis').copy()
                cmap.set_under(color='black')
                vmin = np.min(image[image >= 0]) if np.any(image >= 0) else 0
                vmax = np.max(image[image < np.inf]) if np.any(image < np.inf) else 1
                if vmin == vmax:
                    vmax = vmin + 1
                im = ax1.imshow(image, cmap=cmap, vmin=vmin, vmax=vmax)
                fig.colorbar(im, ax=ax1)
                ax1.set_title('Padplane Image')
                
                # Traces
                # set baseline subtraction to fixed window before saving traces
                old_baseline_mode = h5file.background_subtract_mode
                old_background_bounds = h5file.num_background_bins
                h5file.background_subtract_mode = 'fixed window'
                h5file.num_background_bins = (10,20)
                
                t_pads, t_pad_data = h5file.get_pad_traces(event_num)
                
                for pad, pad_data in zip(t_pads, t_pad_data):
                    r = pad/1024 * 0.8
                    g = (pad%512)/512 * 0.8
                    b = (pad%256)/256 * 0.8
                    if pad in VETO_PADS:
                        ax2.plot(pad_data, '--', color=(r,g,b), label=f'{pad}')
                    else:
                        ax2.plot(pad_data, color=(r,g,b), label=f'{pad}')
                        
                if len(t_pads) > 0 and len(t_pads) <= 30:
                    ax2.legend(loc='upper left', bbox_to_anchor=(1.0, 1.0), title="Pad #", borderaxespad=0.1, ncol=2)
                    
                ax2.set_title('Traces')
                ax2.set_xlabel('Time')
                ax2.set_ylabel('ADC Counts')
                
                h5file.background_subtract_mode = old_baseline_mode
                h5file.num_background_bins = old_background_bounds
                
                pdf.savefig(fig)
                plt.close(fig)

if __name__ == '__main__':
    from e23035_analysis import e23035_runs
    get_runs = np.unique(e23035_runs.run_df['GET'].dropna().astype(int))
    save_images('e23035', get_runs)

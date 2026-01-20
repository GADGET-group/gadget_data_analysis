import numpy as np
import matplotlib.pyplot as plt

from raw_viewer import process_runs

experiment = 'e23035'
bins = np.linspace(0, 4096, 1000)

def show_hist(runs, pads=process_runs.raw_h5_file.VETO_PADS, bins=bins):
    plt.figure()
    max_pad_counts = process_runs.get_quantity('pad_max',experiment, runs)
    for pad in pads:
        plt.hist(max_pad_counts[:, pad], label=pad, bins=bins, alpha=0.4)
    plt.legend()
    plt.yscale('log')
    plt.show(block=False)
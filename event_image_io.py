"""Render the existing pad-plane/trace representation without GUI figure state."""
import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg


class TraceImageRenderer:
    def __init__(self, width=224, height=73, dpi=96):
        self.figure = Figure(figsize=(width / dpi, height / dpi), dpi=dpi)
        self.canvas = FigureCanvasAgg(self.figure)
        self.ax = self.figure.add_subplot(111)

    def compose(self, pad_plane, trace):
        pad_plane = np.asarray(pad_plane)
        trace = np.asarray(trace)
        if pad_plane.ndim != 3 or pad_plane.shape[2] != 3:
            raise ValueError('The pad-plane image must have three colour channels.')
        if trace.ndim != 1 or not len(trace) or not np.all(np.isfinite(trace)):
            raise ValueError('The event trace must be a nonempty, finite vector.')
        self.ax.clear()
        self.ax.set_axis_off()
        self.ax.fill_between(np.arange(len(trace)), trace, color='b', alpha=1)
        self.canvas.draw()
        trace_rgba = np.asarray(self.canvas.buffer_rgba()).copy()
        rgb = np.clip(pad_plane, 0, 255).astype(np.uint8)
        if rgb.shape[1] != trace_rgba.shape[1]:
            raise ValueError('Pad-plane and trace image widths differ.')
        alpha = np.full((*rgb.shape[:2], 1), 255, dtype=np.uint8)
        return np.concatenate((np.concatenate((rgb, alpha), axis=2), trace_rgba), axis=0)

import numpy as np
import tpc_utils_ as tpc_c


def get_peaks(signal: np.ndarray) -> np.ndarray:
    """
    Extract the peaks from the output of the segmentation cnn.

    Parameters
    ----------
    signal : np.ndarray
        The output of the deconvolution and segmentation cnn.
        Must be a 2d array Nx1024, with N signals.
        The first 512 elements are the output of the segmentation cnn.
        The last 512 elements are the signal after the deconvolution.

    Returns
    -------
    peaks : np.ndarray
        The peaks of the signal."""

    return tpc_c.get_peaks(signal)

from typing import Tuple
import numpy as np
import tpc_utils_ as tpc_c
from typing import Tuple


def background(
    signal: np.ndarray,
    number_it: int,
    direction: int,
    filter_order: int,
    smoothing: bool,
    smoothing_window: int,
    compton: bool,
) -> np.ndarray:
    """
    Calculates the background of a signal.
    See TSpectrum documentation for more information.

    Parameters
    ----------
    signal : np.ndarray
        The signal to be analyzed.
    number_it : int
        Maximal width of clipping window.
    direction : int
        Direction of change of clipping window.
        Possible values: 0 for increasing, 1 for decreasing.
    filter_order : int
        Order of clipping filter.
        Possible values: 0, 1, 2 or 3.
    smoothing : bool
        Whether to smooth the background.
    smoothing_window : int
        Width of smoothing window.
    compton : bool
        Whether to use Compton edge.

    Returns
    -------
    background : np.ndarray
        The background of the signal.
    """
    return tpc_c.background(
        signal,
        number_it,
        direction,
        filter_order,
        smoothing,
        smoothing_window,
        compton,
    )


def search_high_res(
    signal: np.ndarray,
    sigma: float,
    threshold: float,
    remove_bkg: bool,
    number_it: int,
    markov: bool,
    aver_window: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    This function searches for peaks in source spectrum. It is based on
    deconvolution method. First the background is removed (if desired), then
    Markov smoothed spectrum is calculated (if desired), then the response
    function is generated according to given sigma and deconvolution is
    carried out. The order of peaks is arranged according to their heights in
    the spectrum after background elimination. The highest peak is the first in
    the list. On success it returns number of found peaks.

    Parameters
    ----------
    signal : np.ndarray
        The signal to be analyzed.
    sigma : float
        The sigma of the response function. Must be equal or greater than 1.
    threshold : float
        The threshold for the peaks (between 0 and 100).
    remove_bkg : bool
        Whether to remove background.
    number_it : int
        Number of iterations for the deconvolution.
    markov : bool
        Whether to use Markov smoothing.
    aver_window : int
        Width of the averaging window.

    Returns
    -------
    response_signal : np.ndarray
        The response functiona after the deconvolution.
    peaks : np.ndarray
        The peaks of the signal.
    """
    return tpc_c.search_high_res(
        signal, sigma, threshold, remove_bkg, number_it, markov, aver_window
    )

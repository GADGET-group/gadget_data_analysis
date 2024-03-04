import numpy as np
import tpc_utils_ as tpc_c
from typing import Tuple


def pRansac(
    pointcloud: np.ndarray,
    n_iter: int,
    min_dist: float,
    min_samples: int = 2,
    mode: int = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Performs RANSAC on a pointcloud.
    pRansac finds the best fitting lines to a pointcloud.
    3D line => versor*t + point, where t is the hyperparameter.

    Parameters
    ----------
    pointcloud : np.ndarray
        The pointcloud to be analyzed.
    n_iter : int
        Number of iterations.
    min_dist : float
        Minimum distance between points.
    min_samples : int
        Minimum number of points for the line.
    mode : int
        Mode of Random Sample.
        Possible values: 0 for uniform random, 1 for gaussian sampling,
        2 for weighted sampling and 3 for weighted gaussian sampling.

    Returns
    -------
    inliers : np.ndarray
        Numpy array that contains the inliers for each 3D line identified.
    versors : np.ndarray
        Numpy array that contains the versors for each 3D line identified.
    points : np.ndarray
        Numpy array that contains the points for each 3D line identified.
    """
    return tpc_c.ransac(pointcloud, n_iter, min_dist, min_samples, mode)


def fit_3D(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fits a 3D line to a pointcloud.
    3D line => versor*t + point, where t is the hyperparameter.

    Parameters
    ----------
    points : np.ndarray
        The points to perform the fit. Must be 4D.

    Returns
    -------
    versor : np.ndarray
        Versor of the 3D line.
    point : np.ndarray
        Point of the 3D line.
    """
    return tpc_c.fit3D(points)

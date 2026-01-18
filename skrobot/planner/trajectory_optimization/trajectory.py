"""Trajectory utilities for trajectory optimization."""

import numpy as np


def interpolate_trajectory(start_angles, end_angles, n_waypoints):
    """Create linear interpolation between start and end configurations.

    Parameters
    ----------
    start_angles : array-like
        Starting joint angles (n_joints,).
    end_angles : array-like
        Ending joint angles (n_joints,).
    n_waypoints : int
        Number of waypoints including start and end.

    Returns
    -------
    numpy.ndarray
        Interpolated trajectory (n_waypoints, n_joints).
    """
    start = np.array(start_angles)
    end = np.array(end_angles)
    t = np.linspace(0, 1, n_waypoints)[:, np.newaxis]
    return start + t * (end - start)

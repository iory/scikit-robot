"""Interpolation of timed trajectories.

Plain functions, no state: each takes knots and returns something to sample.
(Unrelated to the stateful ``skrobot.interpolator`` classes removed in 2026.)
"""

import numpy as np
from scipy.interpolate import CubicHermiteSpline


def hermite_trajectory(times, positions, velocities):
    """Piecewise cubic Hermite trajectory through timed knots.

    The shape a joint trajectory controller follows between its points:
    each segment matches the positions and velocities at both of its ends
    (:class:`scipy.interpolate.CubicHermiteSpline`). A backend that
    executes trajectories itself (a simulator, a test double) can sample it
    at its own clock.

    Parameters
    ----------
    times : array_like
        ``(n_knots,)`` non-decreasing knot times [s]. Two knots with the
        same time are a zero-length segment: the trajectory jumps there,
        to the later knot's position.
    positions : array_like
        ``(n_knots, ...)`` positions at the knots, e.g.
        ``(n_knots, n_joints)``.
    velocities : array_like
        Velocities at the knots, shaped like ``positions``.

    Returns
    -------
    callable
        ``f(t)`` returning the position at time ``t`` as a new array.
        Before the first knot it holds the first position, after the last
        knot the last one (no extrapolation). At the time of a jump it
        returns the position reached before the jump, except at the first
        knot, where it returns the position after it.

    Raises
    ------
    ValueError
        If ``times`` decreases or the shapes do not match.
    """
    times = np.asarray(times, dtype=np.float64)
    positions = np.asarray(positions, dtype=np.float64)
    velocities = np.asarray(velocities, dtype=np.float64)
    if times.ndim != 1 or len(times) == 0:
        raise ValueError('times must be a non-empty 1-D array')
    if positions.shape[0] != len(times) \
            or velocities.shape != positions.shape:
        raise ValueError(
            'positions {} and velocities {} must have one row per time '
            '({})'.format(positions.shape, velocities.shape, len(times)))
    if np.any(np.diff(times) < 0):
        raise ValueError('times must not decrease: {}'.format(times))

    # CubicHermiteSpline needs strictly increasing times, so the knots are
    # split at zero-length segments into pieces, each one spline:
    # (end time, position at the end, spline or None for a jump).
    pieces = []
    n = len(times)
    k = 0
    while k < n - 1:
        if times[k + 1] == times[k]:
            pieces.append((times[k], positions[k + 1], None))
            k += 1
            continue
        j = k + 1
        while j + 1 < n and times[j + 1] > times[j]:
            j += 1
        pieces.append((times[j], positions[j], CubicHermiteSpline(
            times[k:j + 1], positions[k:j + 1], velocities[k:j + 1],
            axis=0)))
        k = j
    first_time = times[0]
    first = positions[0].copy()
    last = positions[-1].copy()

    def evaluate(t):
        if t < first_time:
            return first.copy()
        for piece_end, end_position, spline in pieces:
            if t <= piece_end:
                if spline is None or t == piece_end:
                    return end_position.copy()
                return spline(t)
        return last.copy()
    return evaluate

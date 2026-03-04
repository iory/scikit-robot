import warnings

import numpy as np

from skrobot.coordinates.math import angle_between_vectors
from skrobot.coordinates.math import normalize_vector


def midcoords(p, c1, c2):
    """Returns mid (or p) coordinates of given two coordinates c1 and c2.

    .. deprecated::
        Use :meth:`skrobot.coordinates.Coordinates.interpolate` instead.
        ``midcoords(p, c1, c2)`` is equivalent to ``c1.interpolate(c2, p)``.

    Parameters
    ----------
    p : float
        ratio of c1:c2
    c1 : skrobot.coordinates.Coordinates
        Coordinates
    c2 : skrobot.coordinates.Coordinates
        Coordinates

    Returns
    -------
    coordinates : skrobot.coordinates.Coordinates
        midcoords

    Examples
    --------
    >>> from skrobot.coordinates import Coordinates
    >>> from skrobot.coordinates.geo import midcoords
    >>> c1 = Coordinates()
    >>> c2 = Coordinates(pos=[0.1, 0, 0])
    >>> c = midcoords(0.5, c1, c2)
    >>> c.translation
    array([0.05, 0.  , 0.  ])
    """
    warnings.warn(
        "midcoords(p, c1, c2) is deprecated. "
        "Use c1.interpolate(c2, p) instead.",
        DeprecationWarning,
        stacklevel=2
    )
    return c1.interpolate(c2, p)


def orient_coords_to_axis(target_coords, v, axis='z', eps=0.005):
    """Orient axis to the direction

    .. deprecated::
        Use :meth:`skrobot.coordinates.Coordinates.align_axis_to_direction` instead.
        ``orient_coords_to_axis(c, v, axis)`` is equivalent to
        ``c.align_axis_to_direction(v, axis)``.

    Orient axis in target_coords to the direction specified by v.

    Parameters
    ----------
    target_coords : skrobot.coordinates.Coordinates
    v : list or numpy.ndarray
        position of target [x, y, z]
    axis : list or string or numpy.ndarray
        see convert_to_axis_vector function
    eps : float (optional)
        eps

    Returns
    -------
    target_coords : skrobot.coordinates.Coordinates

    Examples
    --------
    >>> import numpy as np
    >>> from skrobot.coordinates import Coordinates
    >>> from skrobot.coordinates.geo import orient_coords_to_axis
    >>> c = Coordinates()
    >>> oriented_coords = orient_coords_to_axis(c, [1, 0, 0])
    >>> oriented_coords.translation
    array([0., 0., 0.])
    >>> oriented_coords.rpy_angle()
    (array([0.        , 1.57079633, 0.        ]),
     array([3.14159265, 1.57079633, 3.14159265]))

    >>> c = Coordinates(pos=[0, 1, 0])
    >>> oriented_coords = orient_coords_to_axis(c, [0, 1, 0])
    >>> oriented_coords.translation
    array([0., 1., 0.])
    >>> oriented_coords.rpy_angle()
    (array([ 0.        , -0.        , -1.57079633]),
     array([ 3.14159265, -3.14159265,  1.57079633]))

    >>> c = Coordinates(pos=[0, 1, 0]).rotate(np.pi / 3, 'y')
    >>> oriented_coords = orient_coords_to_axis(c, [0, 1, 0])
    >>> oriented_coords.translation
    array([0., 1., 0.])
    >>> oriented_coords.rpy_angle()
    (array([-5.15256299e-17,  1.04719755e+00, -1.57079633e+00]),
     array([3.14159265, 2.0943951 , 1.57079633]))
    """
    warnings.warn(
        "orient_coords_to_axis(c, v, axis) is deprecated. "
        "Use c.align_axis_to_direction(v, axis) instead.",
        DeprecationWarning,
        stacklevel=2
    )
    return target_coords.align_axis_to_direction(v, axis=axis, eps=eps)


def rotate_points(points, a, b):
    """Rotate given points based on a starting and ending vector.

    Axis vector k is calculated from the any two nonzero vectors a and b.
    Rotated points are calculated from following Rodrigues rotation formula.

    .. math::

        `P_{rot} = P \\cos \\theta +
        (k \\times P) \\sin \\theta + k (k \\cdot P) (1 - \\cos \\theta)`

    Parameters
    ----------
    points : numpy.ndarray
        Input points. The shape should be (3, ) or (N, 3).
    a : numpy.ndarray
        nonzero vector.
    b : numpy.ndarray
        nonzero vector.

    Returns
    -------
    points_rot : numpy.ndarray
        rotated points.
    """
    if points.ndim == 1:
        points = points[None, :]

    a = normalize_vector(a)
    b = normalize_vector(b)
    k = normalize_vector(np.cross(a, b))
    theta = angle_between_vectors(a, b, normalize=False)

    points_rot = points * np.cos(theta) \
        + np.cross(k, points) * np.sin(theta) \
        + k * np.dot(k, points.T).reshape(-1, 1) * (1 - np.cos(theta))
    return points_rot

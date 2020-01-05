import numpy as np

from skrobot.coordinates import make_coords
from skrobot.coordinates.math import _wrap_axis
from skrobot.coordinates.math import midpoint
from skrobot.coordinates.math import midrot
from skrobot.coordinates.math import normalize_vector


def midcoords(p, c1, c2):
    """Returns mid (or p) coordinates of given two coordinates c1 and c2.

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
    return make_coords(pos=midpoint(p, c1.worldpos(), c2.worldpos()),
                       rot=midrot(p, c1.worldrot(), c2.worldrot()))


def orient_coords_to_axis(target_coords, v, axis='z', eps=0.005):
    """Orient axis to the direction

    Orient axis in target_coords to the direction specified by v.

    Parameters
    ----------
    target_coords : skrobot.coordinates.Coordinates
    v : list or numpy.ndarray
        position of target [x, y, z]
    axis : list or string or numpy.ndarray
        see _wrap_axis function
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
    v = np.array(v, 'f')
    if np.linalg.norm(v) == 0.0:
        v = np.array([0, 0, 1], 'f')
    nv = normalize_vector(v)
    axis = _wrap_axis(axis)
    ax = target_coords.rotate_vector(axis)
    rot_axis = np.cross(ax, nv)
    rot_angle_cos = np.dot(nv, ax)
    if np.isclose(rot_angle_cos, 1.0, atol=eps):
        return target_coords
    elif np.isclose(rot_angle_cos, -1.0, atol=eps):
        for rot_axis2 in [np.array([1, 0, 0]), np.array([0, 1, 0])]:
            rot_angle_cos2 = np.dot(ax, rot_axis2)
            if not np.isclose(abs(rot_angle_cos2), 1.0, atol=eps):
                rot_axis = rot_axis2 - rot_angle_cos2 * ax
                break
    target_coords.rotate(
        np.arccos(rot_angle_cos), rot_axis, 'world')
    return target_coords

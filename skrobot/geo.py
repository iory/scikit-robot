import numpy as np

from skrobot.coordinates import make_coords
from skrobot.math import _wrap_axis
from skrobot.math import midpoint
from skrobot.math import midrot
from skrobot.math import normalize_vector


def midcoords(p, c1, c2):
    """Returns mid (or p) coordinates of given two coordinates c1 and c2

    Args:
        TODO
    """
    return make_coords(pos=midpoint(p, c1.worldpos(), c2.worldpos()),
                       rot=midrot(p, c1.worldrot(), c2.worldrot()))


def orient_coords_to_axis(target_coords, v, axis='z', eps=0.005):
    """

    Orient axis in target_coords to the direction specified by "v" destructively.
    v must be non-zero vector.

    Parameters
    ----------
    target_coords : Coordinates
    v : list or np.ndarray
        [x, y, z]
    axis : list or string or np.ndarray
        see _wrap_axis function

    Returns
    -------
    target_coords : Coordinates
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

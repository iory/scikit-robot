from math import asin
from math import atan2
from math import cos
from math import sin

import numpy as np


def _wrap_axis(axis):
    if isinstance(axis, str):
        if axis in ['x', 'xx']:
            axis = np.array([1, 0, 0])
        elif axis in ['y', 'yy']:
            axis = np.array([0, 1, 0])
        elif axis in ['z', 'zz']:
            axis = np.array([0, 0, 1])
        elif axis == '-x':
            axis = np.array([-1, 0, 0])
        elif axis == '-y':
            axis = np.array([0, -1, 0])
        elif axis == '-z':
            axis = np.array([0, 0, -1])
        elif axis in ['xy', 'yx']:
            axis = np.array([1, 1, 0])
        elif axis in ['yz', 'zy']:
            axis = np.array([0, 1, 1])
        elif axis in ['zx', 'xz']:
            axis = np.array([1, 0, 1])
        else:
            raise NotImplementedError
    elif isinstance(axis, list):
        if not len(axis) == 3:
            raise ValueError
        axis = np.array(axis)
    elif isinstance(axis, np.ndarray):
        if not axis.shape == (3,):
            raise ValueError
    elif isinstance(axis, bool):
        if axis is True:
            return np.array([0, 0, 0])
        else:
            return np.array([1, 1, 1])
    elif axis is None:
        return np.array([1, 1, 1])
    else:
        raise ValueError
    return axis


def sr_inverse(J, k=1.0, weight_vector=None):
    """returns sr-inverse of given mat"""
    r, _ = J.shape

    # without weight
    if weight_vector is None:
        return sr_inverse_org(J, k)

    # k=0 => sr-inverse = pseudo-inverse
    if k == 0.0:
        return np.linalg.pinv(J)

    # with weight
    weight_matrix = np.diag(weight_vector)

    # umat = J W J^T + kI
    # ret = W J^T (J W J^T + kI)^(-1)
    weight_J = np.matmul(weight_matrix, J.T)
    umat = np.matmul(J, weight_J) + k * np.eye(r)
    ret = np.matmul(weight_J, np.linalg.inv(umat))
    return ret


def sr_inverse_org(J, k=1.0):
    """J^T (JJ^T + kI_m)^(-1)"""
    r, _ = J.shape
    return np.matmul(J.T,
                     np.linalg.inv(np.matmul(J, J.T) + k * np.eye(r)))


def manipulability(J):
    """return manipulability of given matrix.
    https://www.jstage.jst.go.jp/article/jrsj1983/2/1/2_1_63/_article/-char/ja/
    """
    return np.sqrt(max(0.0, np.linalg.det(np.matmul(J, J.T))))


def midpoint(p, a, b):
    return a + (b - a) * p


def rotation_matrix(theta, axis):
    """
    Return the rotation matrix associated with counterclockwise rotation about
    the given axis by theta radians.
    """
    axis = _wrap_axis(axis)
    axis = axis / np.sqrt(np.dot(axis, axis))
    a = np.cos(theta / 2.0)
    b, c, d = -axis * np.sin(theta / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                     [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                     [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])


def rotate_matrix(matrix, theta, axis, world=None):
    if world is False or world is None:
        return np.dot(matrix, rotation_matrix(theta, axis))
    return np.dot(rotation_matrix(theta, axis), matrix)


def rpy_matrix(az, ay, ax):
    """RPY-MATRIX (az ay ax) creates a new rotation matrix which has been
    rotated ax radian around x-axis in WORLD, ay radian around y-axis in
    WORLD, and az radian around z axis in WORLD, in this order.
    These angles can be extracted by the RPY-ANGLE function."""
    r = rotation_matrix(ax, 'x')
    r = rotate_matrix(r, ay, 'y')
    r = rotate_matrix(r, az, 'z')
    return r


def rpy_angle(matrix):
    """Decomposing a rotation matrix"""
    r = np.arctan2(matrix[2, 1], matrix[2, 2])
    p = np.arctan2(- matrix[2, 0],
                   np.sqrt(matrix[2, 1] ** 2 + matrix[2, 2] ** 2))
    y = np.arctan2(matrix[1, 0], matrix[0, 0])
    rpy = np.array([y, p, r])
    return rpy, np.pi - rpy


def normalize_vector(v, ord=2):
    if np.allclose(v, 0) is True:
        return v
    return v / np.linalg.norm(v, ord=ord)


def matrix2quaternion(m):
    """Returns quaternion of given rotation matrix.
    """
    m = np.array(m, dtype=np.float64)
    q0_2 = (1 + m[0, 0] + m[1, 1] + m[2, 2]) / 4.0
    q1_2 = (1 + m[0, 0] - m[1, 1] - m[2, 2]) / 4.0
    q2_2 = (1 - m[0, 0] + m[1, 1] - m[2, 2]) / 4.0
    q3_2 = (1 - m[0, 0] - m[1, 1] + m[2, 2]) / 4.0
    mq_2 = max(q0_2, q1_2, q2_2, q3_2)
    if np.isclose(mq_2, q0_2):
        q0 = np.sqrt(q0_2)
        q1 = ((m[2, 1] - m[1, 2]) / (4.0 * q0))
        q2 = ((m[0, 2] - m[2, 0]) / (4.0 * q0))
        q3 = ((m[1, 0] - m[0, 1]) / (4.0 * q0))
    elif np.isclose(mq_2, q1_2):
        q1 = np.sqrt(q1_2)
        q0 = ((m[2, 1] - m[1, 2]) / (4.0 * q1))
        q2 = ((m[1, 0] + m[0, 1]) / (4.0 * q1))
        q3 = ((m[0, 2] + m[2, 0]) / (4.0 * q1))
    elif np.isclose(mq_2, q2_2):
        q2 = np.sqrt(q2_2)
        q0 = ((m[0, 2] - m[2, 0]) / (4.0 * q2))
        q1 = ((m[1, 0] + m[0, 1]) / (4.0 * q2))
        q3 = ((m[1, 2] + m[2, 1]) / (4.0 * q2))
    elif np.isclose(mq_2, q3_2):
        q3 = np.sqrt(q3_2)
        q0 = ((m[1, 0] - m[0, 1]) / (4.0 * q3))
        q1 = ((m[0, 2] + m[2, 0]) / (4.0 * q3))
        q2 = ((m[1, 2] + m[2, 1]) / (4.0 * q3))
    else:
        raise ValueError('matrix {} is invalid'.format(m))
    return np.array([q0, q1, q2, q3])


def quaternion2matrix(q):
    """Returns matrix of given quaternion"""
    q0 = q[0]
    q1 = q[1]
    q2 = q[2]
    q3 = q[3]
    norm = np.linalg.norm(q)
    if not np.isclose(norm, 1.0):
        raise ValueError("quaternion q's norm is not 1")
    m = np.zeros((3, 3))
    m[0, 0] = q0 * q0 + q1 * q1 - q2 * q2 - q3 * q3
    m[0, 1] = 2 * (q1 * q2 - q0 * q3)
    m[0, 2] = 2 * (q1 * q3 + q0 * q2)

    m[1, 0] = 2 * (q1 * q2 + q0 * q3)
    m[1, 1] = q0 * q0 - q1 * q1 + q2 * q2 - q3 * q3
    m[1, 2] = 2 * (q2 * q3 - q0 * q1)

    m[2, 0] = 2 * (q1 * q3 - q0 * q2)
    m[2, 1] = 2 * (q2 * q3 + q0 * q1)
    m[2, 2] = q0 * q0 - q1 * q1 - q2 * q2 + q3 * q3
    return m


def matrix_log(m):
    """returns matrix log of given m, it returns [-pi, pi]"""
    qq = matrix2quaternion(m)
    q0 = qq[0]
    q = qq[1:]
    th = 2.0 * np.arctan(np.linalg.norm(q) / q0)
    if th > np.pi:
        th = th - 2.0 * np.pi
    elif th < - np.pi:
        th = th + 2.0 * np.pi
    return th * normalize_vector(q)


def quaternion2rpy(q):
    """
    Roll-pitch-yaw angles of a quaternion.

    Parameters
    ----------
    quat : (4,) array
        Quaternion in `[w x y z]` format.

    Returns
    -------
    rpy : (3,) array
        Array of yaw-pitch-roll angles, in [rad].
    """
    roll = atan2(
        2 * q[2] * q[3] + 2 * q[0] * q[1],
        q[3] ** 2 - q[2] ** 2 - q[1] ** 2 + q[0] ** 2)
    pitch = -asin(
        2 * q[1] * q[3] - 2 * q[0] * q[2])
    yaw = atan2(
        2 * q[1] * q[2] + 2 * q[0] * q[3],
        q[1] ** 2 + q[0] ** 2 - q[3] ** 2 - q[2] ** 2)
    rpy = np.array([yaw, pitch, roll])
    return rpy, np.pi - rpy


def rpy2quaternion(rpy):
    """
    Quaternion frmo yaw-pitch-roll angles.

    Parameters
    ----------
    rpy : (3,) array
        Vector of yaw-pitch-roll angles in [rad].

    Returns
    -------
    quat : (4,) array
        Quaternion in `[w x y z]` format.
    """
    yaw, pitch, roll = rpy
    cr, cp, cy = cos(roll / 2.), cos(pitch / 2.), cos(yaw / 2.)
    sr, sp, sy = sin(roll / 2.), sin(pitch / 2.), sin(yaw / 2.)
    return np.array([
        cr * cp * cy + sr * sp * sy,
        -cr * sp * sy + cp * cy * sr,
        cr * cy * sp + sr * cp * sy,
        cr * cp * sy - sr * cy * sp])

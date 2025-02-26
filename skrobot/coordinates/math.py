from __future__ import absolute_import

import math
from math import acos
from math import asin
from math import atan2
from math import cos
from math import pi
from math import sin
import warnings

import numpy as np


# epsilon for testing whether a number is close to zero
_EPS = np.finfo(float).eps * 4.0
_AXIS_VECTORS = {
    'x': np.array([1, 0, 0]),
    'y': np.array([0, 1, 0]),
    'z': np.array([0, 0, 1]),
    '-x': np.array([-1, 0, 0]),
    '-y': np.array([0, -1, 0]),
    '-z': np.array([0, 0, -1]),
    'xy': np.array([1, 1, 0]),
    'yx': np.array([1, 1, 0]),
    'yz': np.array([0, 1, 1]),
    'zy': np.array([0, 1, 1]),
    'zx': np.array([1, 0, 1]),
    'xz': np.array([1, 0, 1]),
}


def convert_to_axis_vector(axis):
    """Convert axis to float vector.

    Parameters
    ----------
    axis : list or numpy.ndarray or str or bool or None
        rotation axis indicated by number or string.

    Returns
    -------
    axis : numpy.ndarray
        converted axis

    Examples
    --------
    >>> from skrobot.coordinates.math import convert_to_axis_vector
    >>> convert_to_axis_vector('x')
    array([1, 0, 0])
    >>> convert_to_axis_vector('y')
    array([0, 1, 0])
    >>> convert_to_axis_vector('z')
    array([0, 0, 1])
    >>> convert_to_axis_vector('xy')
    array([1, 1, 0])
    >>> convert_to_axis_vector([1, 1, 1])
    array([1, 1, 1])
    >>> convert_to_axis_vector(True)
    array([0, 0, 0])
    >>> convert_to_axis_vector(False)
    array([1, 1, 1])
    """
    if isinstance(axis, str):
        try:
            return _AXIS_VECTORS[axis]
        except KeyError:
            raise NotImplementedError(
                "Axis conversion for '{}' is not implemented.".format(axis))
    elif isinstance(axis, list):
        if len(axis) != 3:
            raise ValueError("Axis list must have exactly three elements.")
        return np.array(axis)
    elif isinstance(axis, np.ndarray):
        if axis.shape != (3,):
            raise ValueError("Axis ndarray must be of shape (3,).")
        return axis
    elif isinstance(axis, bool):
        # If True, returns a zero vector; if False, returns a ones vector
        return np.zeros(3) if axis else np.ones(3)
    elif axis is None:
        return np.ones(3)
    else:
        raise ValueError("Invalid type for axis. "
                         "Must be one of: str, list, ndarray, bool, None.")


def _wrap_axis(axis):
    warnings.warn(
        'Function `_wrap_axis` is deprecated. '
        'Please use `convert_to_axis_vector` instead',
        DeprecationWarning)
    return convert_to_axis_vector(axis)


def to_numpy_array(arr):
    if isinstance(arr, (list, tuple)):
        return np.array(arr)
    elif isinstance(arr, np.ndarray):
        return arr
    else:
        raise TypeError("Input must be a list, tuple, or numpy.ndarray.")


def _check_valid_rotation(rotation):
    """Checks that the given rotation matrix is valid."""
    rotation = np.array(rotation)
    if not isinstance(
            rotation,
            np.ndarray) or not np.issubdtype(
            rotation.dtype,
            np.number):
        raise ValueError('Rotation must be specified as numeric numpy array')

    if len(rotation.shape) != 2 or \
       rotation.shape[0] != 3 or rotation.shape[1] != 3:
        raise ValueError('Rotation must be specified as a 3x3 ndarray')

    if np.abs(np.linalg.det(rotation) - 1.0) > 1e-3:
        raise ValueError('Illegal rotation. Must have determinant == 1.0, '
                         'get {}'.format(np.linalg.det(rotation)))
    return rotation


def _check_valid_translation(translation):
    """Checks that the translation vector is valid."""
    if not isinstance(
            translation,
            np.ndarray) or not np.issubdtype(
            translation.dtype,
            np.number):
        raise ValueError(
            'Translation must be specified as numeric numpy array')

    t = translation.squeeze()
    if len(t.shape) != 1 or t.shape[0] != 3:
        raise ValueError(
            'Translation must be specified as a 3-vector, '
            '3x1 ndarray, or 1x3 ndarray')


def wxyz2xyzw(quat):
    """Convert quaternion [w, x, y, z] to [x, y, z, w] order.

    Parameters
    ----------
    quat : list or numpy.ndarray
        quaternion [w, x, y, z]
    Returns
    -------
    quaternion : numpy.ndarray
        quaternion [x, y, z, w]

    Examples
    --------
    >>> from skrobot.coordinates.math import wxyz2xyzw
    >>> wxyz2xyzw([1, 2, 3, 4])
    array([2, 3, 4, 1])
    """
    if isinstance(quat, (list, tuple)):
        quat = np.array(quat)
    return np.roll(quat, -1, axis=quat.ndim - 1)


def xyzw2wxyz(quat):
    """Convert quaternion [x, y, z, w] to [w, x, y, z] order.

    Parameters
    ----------
    quat : list or numpy.ndarray
        quaternion [x, y, z, w]

    Returns
    -------
    quaternion : numpy.ndarray
        quaternion [w, x, y, z]

    Examples
    --------
    >>> from skrobot.coordinates.math import xyzw2wxyz
    >>> xyzw2wxyz([1, 2, 3, 4])
    array([4, 1, 2, 3])
    """
    if isinstance(quat, (list, tuple)):
        quat = np.array(quat)
    return np.roll(quat, 1, axis=quat.ndim - 1)


def triple_product(a, b, c):
    """Returns Triple Product

    See https://en.wikipedia.org/wiki/Triple_product.

    Geometrically, the scalar triple product

    :math:`a\\cdot(b \\times c)`

    is the (signed) volume of the parallelepiped defined
    by the three vectors given.

    Parameters
    ----------
    a : numpy.ndarray
        vector a
    b : numpy.ndarray
        vector b
    c : numpy.ndarray
        vector c

    Returns
    -------
    triple product : float
        calculated triple product

    Examples
    --------
    >>> from skrobot.math import triple_product
    >>> triple_product([1, 1, 1], [1, 1, 1], [1, 1, 1])
    0
    >>> triple_product([1, 0, 0], [0, 1, 0], [0, 0, 1])
    1
    """
    return np.dot(a, np.cross(b, c))


def sr_inverse(J, k=1.0, weight_vector=None):
    """Returns SR-inverse of given Jacobian.

    Calculate Singularity-Robust Inverse
    See: `Inverse Kinematic Solutions With Singularity Robustness \
          for Robot Manipulator Control`

    Parameters
    ----------
    J : numpy.ndarray
        jacobian
    k : float
        coefficients
    weight_vector : None or numpy.ndarray
        weight vector

    Returns
    -------
    ret : numpy.ndarray
        result of SR-inverse
    """
    if J.ndim == 3:
        return batch_sr_inverse(J, k, weight_vector=weight_vector)
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
    """Return SR-inverse of given J

    Definition of SR-inverse is following.

    :math:`J^* = J^T(JJ^T + kI_m)^{-1}`

    Parameters
    ----------
    J : numpy.ndarray
        jacobian
    k : float
        coefficients

    Returns
    -------
    sr_inverse : numpy.ndarray
        calculated SR-inverse
    """
    r, _ = J.shape
    return np.matmul(J.T,
                     np.linalg.inv(np.matmul(J, J.T) + k * np.eye(r)))


def batch_sr_inverse(J_batch, k_values, weight_vector=None):
    """Compute the SR-inverse for a batch of Jacobian matrices.

    This function computes the SR-inverse for each matrix in a batch by
    applying a regularization term scaled by `k_values` and optionally
    using a weighted pseudo-inverse if `weight_vector` is provided.

    Parameters
    ----------
    J_batch : ndarray
        A 3D numpy array of shape (batch_size, r, c) where each
        slice (J_batch[i]) represents a Jacobian matrix.
    k_values : ndarray
        A 1D numpy array of length batch_size, where each element
        k_values[i] represents the regularization parameter
        for the i-th matrix in the batch.
    weight_vector : ndarray, optional
        A 1D numpy array of length c (the number of columns in
        each Jacobian matrix), which is used to create a diagonal
        weight matrix W. If None, the identity matrix is used as
        the weight matrix (default is None).

    Returns
    -------
    ndarray
        A 3D numpy array of shape (batch_size, c, r), where each slice
        is the SR-inverse of the corresponding Jacobian matrix in the
        input batch.

    Notes
    -----
    The SR-inverse is computed using the formula:
        (W*J^T*(J*W*J^T + k*I)^-1),
    where J is the Jacobian matrix, W is the weight matrix
    (either identity or derived from `weight_vector`),
    J^T is the transpose of J, I is the identity matrix, and k
    is the regularization parameter.
    """
    batch_size, r, c = J_batch.shape
    I = np.eye(r)
    # Adjust k_values to be shape (batch_size, r, r) for broadcasting
    k_matrix = k_values[:, np.newaxis, np.newaxis] * np.tile(
        I, (batch_size, 1, 1))
    if weight_vector is None:
        W = np.eye(c)
    else:
        W = np.diag(weight_vector)
    W = W.reshape((c, c))
    J_T = J_batch.transpose(0, 2, 1)
    weighted_J_T = np.matmul(W, J_T)
    umat = np.matmul(J_batch, weighted_J_T) + k_matrix
    umat_inv = np.linalg.inv(umat)
    sr_inverse = np.matmul(weighted_J_T, umat_inv)
    return sr_inverse


def jacobian_inverse(J, manipulability_limit=0.1,
                     manipulability_gain=0.001,
                     weight=None):
    """Calculate the pseudo-inverse or SR-inverse depending on manipulability.

    """
    if J.ndim == 2:  # Single Jacobian matrix
        m = manipulability(J)
        if m < manipulability_limit:
            k = manipulability_gain * ((1.0 - m / manipulability_limit) ** 2)
        else:
            k = 0
        return sr_inverse(J, k, weight)

    batch_size, r, c = J.shape
    # Calculate manipulability for each Jacobian
    m = manipulability(J)

    # Calculate k values for the entire batch based on manipulability
    k_values = np.zeros(batch_size)
    low_manipulability_mask = m < manipulability_limit
    if np.any(low_manipulability_mask):
        k_values[low_manipulability_mask] = manipulability_gain * (
            (1.0 - m[low_manipulability_mask] / manipulability_limit) ** 2)

    # Compute SR-inverse for the entire batch using calculated k values
    sr_inverses = batch_sr_inverse(J, k_values, weight)
    # Return single matrix or batch depending on input
    return sr_inverses


def manipulability(J):
    """Compute manipulability for a single matrix or each matrix in a batch.

    This function can handle both a single Jacobian matrix and a batch of
    Jacobian matrices. For a single matrix, it returns the manipulability
    as a float. For a batch, it returns an array of manipulability values.

    Definition of manipulability is following.

    :math:`w = \\sqrt{\\det J(\\theta)J^T(\\theta)}`

    Parameters
    ----------
    J : numpy.ndarray
        Jacobian matrix or batch of Jacobian matrices. It should be of
        shape (n, m) for a single matrix or (batch_size, n, m) for a batch.

    Returns
    -------
    w : float
        Manipulability or array of manipulability values.
    """
    if J.ndim == 2:  # Single matrix case
        return np.sqrt(max(0.0, np.linalg.det(np.matmul(J, J.T))))
    elif J.ndim == 3:  # Batch of matrices
        batch_size, _, _ = J.shape
        JJT = np.matmul(J, np.transpose(J, axes=(0, 2, 1)))
        det_JJT = np.linalg.det(JJT)
        det_JJT = np.maximum(0.0, det_JJT)  # Ensure non-negative before sqrt
        w = np.sqrt(det_JJT)
        return w
    else:
        raise ValueError("Input must be a 2D or 3D numpy array")


def midpoint(p, a, b):
    """Return midpoint

    Parameters
    ----------
    p : float
        ratio of a:b
    a : numpy.ndarray
        vector
    b : numpy.ndarray
        vector

    Returns
    -------
    midpoint : numpy.ndarray
        midpoint

    Examples
    --------
    >>> import numpy as np
    >>> from skrobot.coordinates.math import midpoint
    >>> midpoint(0.5, np.ones(3), np.zeros(3))
    >>> array([0.5, 0.5, 0.5])
    """
    return a + (b - a) * p


def midrot(p, r1, r2):
    """Returns mid (or p) rotation matrix of given two matrix r1 and r2.

    Parameters
    ----------
    p : float
        ratio of r1:r2
    r1 : numpy.ndarray
        3x3 rotation matrix
    r2 : numpy.ndarray
        3x3 rotation matrix

    Returns
    -------
    r : numpy.ndarray
        3x3 rotation matrix

    Examples
    --------
    >>> import numpy as np
    >>> from skrobot.coordinates.math import midrot
    >>> midrot(0.5,
            np.eye(3),
            np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]]))
    array([[ 0.70710678,  0.        ,  0.70710678],
           [ 0.        ,  1.        ,  0.        ],
           [-0.70710678,  0.        ,  0.70710678]])
    >>> from skrobot.coordinates.math import rpy_angle
    >>> np.rad2deg(rpy_angle(midrot(0.5,
                   np.eye(3),
                   np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]])))[0])
    array([ 0., 45.,  0.])
    """
    r1 = _check_valid_rotation(r1)
    r2 = _check_valid_rotation(r2)
    r = np.matmul(r1.T, r2)
    omega = matrix_log(r)
    r = matrix_exponent(omega, p)
    return np.matmul(r1, r)


def transform(m, v):
    """Return transform m v

    Parameters
    ----------
    m : numpy.ndarray
        3 x 3 rotation matrix.
    v  : numpy.ndarray or list
        input vector.

    Returns
    -------
    np.matmul(m, v) : numpy.ndarray
        transformed vector.
    """
    m = np.array(m)
    v = np.array(v)
    return np.matmul(m, v)


def rotation_matrix(theta, axis, skip_normalization=False):
    """Return the rotation matrix.

    Return the rotation matrix associated with counterclockwise rotation
    about the given axis by theta radians.

    Parameters
    ----------
    theta : float
        radian
    axis : str or list or numpy.ndarray
        rotation axis such that 'x', 'y', 'z'
        [0, 0, 1], [0, 1, 0], [1, 0, 0]
    skip_normalization : bool
        if `True`, skip normalization for axis.

    Returns
    -------
    rot : numpy.ndarray
        rotation matrix about the given axis by theta radians.

    Examples
    --------
    >>> import numpy as np
    >>> from skrobot.coordinates.math import rotation_matrix
    >>> rotation_matrix(np.pi / 2.0, [1, 0, 0])
    array([[ 1.00000000e+00,  0.00000000e+00,  0.00000000e+00],
           [ 0.00000000e+00,  2.22044605e-16, -1.00000000e+00],
           [ 0.00000000e+00,  1.00000000e+00,  2.22044605e-16]])
    >>> rotation_matrix(np.pi / 2.0, 'y')
    array([[ 2.22044605e-16,  0.00000000e+00,  1.00000000e+00],
           [ 0.00000000e+00,  1.00000000e+00,  0.00000000e+00],
           [-1.00000000e+00,  0.00000000e+00,  2.22044605e-16]])
    """
    if not skip_normalization:
        axis = convert_to_axis_vector(axis)
        axis = axis / np.sqrt(np.dot(axis, axis))
    a = np.cos(theta / 2.0)
    b, c, d = -axis * np.sin(theta / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                     [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                     [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])


def rotate_vector(vec, theta, axis):
    """Rotate vector.

    Rotate vec with respect to axis.

    Parameters
    ----------
    vec : list or numpy.ndarray
        target vector
    theta : float
        rotation angle
    axis : list or numpy.ndarray or str
        axis of rotation.

    Returns
    -------
    rotated_vec : numpy.ndarray
        rotated vector.

    Examples
    --------
    >>> from numpy import pi
    >>> from skrobot.coordinates.math import rotate_vector
    >>> rotate_vector([1, 0, 0], pi / 6.0, [1, 0, 0])
    array([1., 0., 0.])
    >>> rotate_vector([1, 0, 0], pi / 6.0, [0, 1, 0])
    array([ 0.8660254,  0.       , -0.5      ])
    >>> rotate_vector([1, 0, 0], pi / 6.0, [0, 0, 1])
    array([0.8660254, 0.5      , 0.       ])
    """
    rot = rotation_matrix(theta, axis)
    rotated_vec = transform(rot, vec)
    return rotated_vec


def rotate_matrix(matrix, theta, axis, world=None, skip_normalization=False):
    if world is False or world is None:
        return np.dot(
            matrix,
            rotation_matrix(theta, axis,
                            skip_normalization=skip_normalization))
    return np.dot(
        rotation_matrix(theta, axis, skip_normalization=skip_normalization),
        matrix)


def rpy_matrix(az, ay, ax):
    """Return rotation matrix from yaw-pitch-roll

    This function creates a new rotation matrix which has been
    rotated ax radian around x-axis in WORLD, ay radian around y-axis in WORLD,
    and az radian around z axis in WORLD, in this order. These angles can be
    extracted by the rpy function.

    Parameters
    ----------
    az : float
        rotated around z-axis(yaw) in radian.
    ay : float
        rotated around y-axis(pitch) in radian.
    ax : float
        rotated around x-axis(roll) in radian.

    Returns
    -------
    r : numpy.ndarray
        rotation matrix

    Examples
    --------
    >>> import numpy as np
    >>> from skrobot.coordinates.math import rpy_matrix
    >>> yaw = np.pi / 2.0
    >>> pitch = np.pi / 3.0
    >>> roll = np.pi / 6.0
    >>> rpy_matrix(yaw, pitch, roll)
    array([[ 1.11022302e-16, -8.66025404e-01,  5.00000000e-01],
           [ 5.00000000e-01,  4.33012702e-01,  7.50000000e-01],
           [-8.66025404e-01,  2.50000000e-01,  4.33012702e-01]])
    """
    r = rotation_matrix(ax, 'x')
    r = rotate_matrix(r, ay, 'y', world=True)
    r = rotate_matrix(r, az, 'z', world=True)
    return r


def rpy_angle(matrix):
    """Decomposing a rotation matrix to yaw-pitch-roll.

    Parameters
    ----------
    matrix : list or numpy.ndarray
        3x3 rotation matrix

    Returns
    -------
    rpy : tuple(numpy.ndarray, numpy.ndarray)
        pair of rpy in yaw-pitch-roll order.

    Examples
    --------
    >>> import numpy as np
    >>> from skrobot.coordinates.math import rpy_matrix
    >>> from skrobot.coordinates.math import rpy_angle
    >>> yaw = np.pi / 2.0
    >>> pitch = np.pi / 3.0
    >>> roll = np.pi / 6.0
    >>> rot = rpy_matrix(yaw, pitch, roll)
    >>> rpy_angle(rot)
    (array([1.57079633, 1.04719755, 0.52359878]),
     array([ 4.71238898,  2.0943951 , -2.61799388]))
    """
    if np.sqrt(matrix[1, 0] ** 2 + matrix[0, 0] ** 2) < _EPS:
        a = 0.0
    else:
        a = np.arctan2(matrix[1, 0], matrix[0, 0])
    sa = np.sin(a)
    ca = np.cos(a)
    b = np.arctan2(-matrix[2, 0], ca * matrix[0, 0] + sa * matrix[1, 0])
    c = np.arctan2(sa * matrix[0, 2] - ca * matrix[1, 2],
                   -sa * matrix[0, 1] + ca * matrix[1, 1])
    rpy = np.array([a, b, c])

    a = a + np.pi
    sa = np.sin(a)
    ca = np.cos(a)
    b = np.arctan2(-matrix[2, 0], ca * matrix[0, 0] + sa * matrix[1, 0])
    c = np.arctan2(sa * matrix[0, 2] - ca * matrix[1, 2],
                   -sa * matrix[0, 1] + ca * matrix[1, 1])
    return rpy, np.array([a, b, c])


def normalize_vector(v, ord=2):
    """Return normalized vector

    Parameters
    ----------
    v : list or numpy.ndarray
        vector
    ord : int (optional)
        ord of np.linalg.norm

    Returns
    -------
    v : numpy.ndarray
        normalized vector

    Examples
    --------
    >>> from skrobot.coordinates.math import normalize_vector
    >>> normalize_vector([1, 1, 1])
    array([0.57735027, 0.57735027, 0.57735027])
    >>> normalize_vector([0, 0, 0])
    array([0., 0., 0.])
    """
    v = np.array(v, dtype=np.float64)
    norm = np.linalg.norm(v, ord=ord)
    if norm == 0:
        return v
    return v / norm


def matrix2quaternion(m):
    """Returns quaternion of given rotation matrix.

    Parameters
    ----------
    m : list or numpy.ndarray
        3x3 rotation matrix

    Returns
    -------
    quaternion : numpy.ndarray
        quaternion [w, x, y, z] order

    Examples
    --------
    >>> import numpy
    >>> from skrobot.coordinates.math import matrix2quaternion
    >>> matrix2quaternion(np.eye(3))
    array([1., 0., 0., 0.])
    """
    m = np.array(m, dtype=np.float64)
    if m.ndim == 2:
        tr = m[0, 0] + m[1, 1] + m[2, 2]
        if tr > 0:
            S = math.sqrt(tr + 1.0) * 2
            qw = 0.25 * S
            qx = (m[2, 1] - m[1, 2]) / S
            qy = (m[0, 2] - m[2, 0]) / S
            qz = (m[1, 0] - m[0, 1]) / S
        elif (m[0, 0] > m[1, 1]) and (m[0, 0] > m[2, 2]):
            S = math.sqrt(1. + m[0, 0] - m[1, 1] - m[2, 2]) * 2
            qw = (m[2, 1] - m[1, 2]) / S
            qx = 0.25 * S
            qy = (m[0, 1] + m[1, 0]) / S
            qz = (m[0, 2] + m[2, 0]) / S
        elif m[1, 1] > m[2, 2]:
            S = math.sqrt(1. + m[1, 1] - m[0, 0] - m[2, 2]) * 2
            qw = (m[0, 2] - m[2, 0]) / S
            qx = (m[0, 1] + m[1, 0]) / S
            qy = 0.25 * S
            qz = (m[1, 2] + m[2, 1]) / S
        else:
            S = math.sqrt(1. + m[2, 2] - m[0, 0] - m[1, 1]) * 2
            qw = (m[1, 0] - m[0, 1]) / S
            qx = (m[0, 2] + m[2, 0]) / S
            qy = (m[1, 2] + m[2, 1]) / S
            qz = 0.25 * S
        return np.array([qw, qx, qy, qz])
    elif m.ndim == 3:
        r1, r2, r3 = m[:, 0, :], m[:, 1, :], m[:, 2, :]
        r11, r12, r13 = r1[:, 0], r1[:, 1], r1[:, 2]
        r21, r22, r23 = r2[:, 0], r2[:, 1], r2[:, 2]
        r31, r32, r33 = r3[:, 0], r3[:, 1], r3[:, 2]

        q0 = 0.25 * (r11 + r22 + r33 + 1)
        q1 = 0.25 * (r11 - r22 - r33 + 1)
        q2 = 0.25 * (-r11 + r22 - r33 + 1)
        q3 = 0.25 * (-r11 - r22 + r33 + 1)

        q0[np.where(q0 < 0.0)] = 0.
        q1[np.where(q1 < 0.0)] = 0.
        q2[np.where(q2 < 0.0)] = 0.
        q3[np.where(q3 < 0.0)] = 0.

        q0 = np.sqrt(q0)
        q1 = np.sqrt(q1)
        q2 = np.sqrt(q2)
        q3 = np.sqrt(q3)

        ones = np.ones_like(r11)
        aranges = np.arange(ones.shape[0])
        signs_array = np.array([
            [ones, np.sign(r32 - r23), np.sign(r13 - r31), np.sign(r21 - r12)],
            [np.sign(r32 - r23), ones, np.sign(r21 + r12), np.sign(r13 + r31)],
            [np.sign(r13 - r31), np.sign(r21 + r12), ones, np.sign(r32 + r23)],
            [np.sign(r21 - r12), np.sign(r31 + r13), np.sign(r32 + r23), ones],
        ])

        argmaxes = np.argmax(np.array([q0, q1, q2, q3]), axis=0)
        signs = signs_array[:, argmaxes, aranges]

        res = np.array([q0, q1, q2, q3]).T * signs.T
        resq = res / np.linalg.norm(res, axis=1, keepdims=True)
        return resq
    else:
        raise ValueError(
            'Unsupported rotation matrix shape. '
            'Supports rotation matrices of (N, 3, 3) and (3, 3).')


def quaternion2matrix(q, normalize=False):
    """Returns matrix of given quaternion.

    Parameters
    ----------
    quaternion : list or numpy.ndarray
        quaternion [w, x, y, z] order
    normalize : bool
        if normalize is True, input quaternion is normalized.

    Returns
    -------
    rot : numpy.ndarray
        3x3 rotation matrix

    Examples
    --------
    >>> import numpy
    >>> from skrobot.coordinates.math import quaternion2matrix
    >>> quaternion2matrix([1, 0, 0, 0])
    array([[1., 0., 0.],
           [0., 1., 0.],
           [0., 0., 1.]])
    """
    q = np.array(q)
    if normalize:
        q = quaternion_normalize(q)
    else:
        norm = quaternion_norm(q)
        if not np.allclose(norm, 1.0):
            raise ValueError("quaternion q's norm is not 1")
    if q.ndim == 1:
        q0 = q[0]
        q1 = q[1]
        q2 = q[2]
        q3 = q[3]

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
    elif q.ndim == 2:
        m = np.zeros((q.shape[0], 3, 3), dtype=np.float64)
        m[:, 0, 0] = q[:, 0] * q[:, 0] + \
            q[:, 1] * q[:, 1] - q[:, 2] * q[:, 2] - q[:, 3] * q[:, 3]
        m[:, 0, 1] = 2 * (q[:, 1] * q[:, 2] - q[:, 0] * q[:, 3])
        m[:, 0, 2] = 2 * (q[:, 1] * q[:, 3] + q[:, 0] * q[:, 2])

        m[:, 1, 0] = 2 * (q[:, 1] * q[:, 2] + q[:, 0] * q[:, 3])
        m[:, 1, 1] = q[:, 0] * q[:, 0] - \
            q[:, 1] * q[:, 1] + q[:, 2] * q[:, 2] - q[:, 3] * q[:, 3]
        m[:, 1, 2] = 2 * (q[:, 2] * q[:, 3] - q[:, 0] * q[:, 1])

        m[:, 2, 0] = 2 * (q[:, 1] * q[:, 3] - q[:, 0] * q[:, 2])
        m[:, 2, 1] = 2 * (q[:, 2] * q[:, 3] + q[:, 0] * q[:, 1])
        m[:, 2, 2] = q[:, 0] * q[:, 0] - \
            q[:, 1] * q[:, 1] - q[:, 2] * q[:, 2] + q[:, 3] * q[:, 3]
    return m


def matrix_log(m):
    """Returns matrix log of given rotation matrix, it returns [-pi, pi]

    Parameters
    ----------
    m : list or numpy.ndarray
        3x3 rotation matrix

    Returns
    -------
    matrixlog : numpy.ndarray
        vector of shape (3, )

    Examples
    --------
    >>> import numpy as np
    >>> from skrobot.coordinates.math import matrix_log
    >>> matrix_log(np.eye(3))
    array([0., 0., 0.])
    """
    # calc logarithm of quaternion
    q = matrix2quaternion(m)
    q_w = q[0]
    q_xyz = q[1:]
    theta = 2.0 * np.arctan(np.linalg.norm(q_xyz) / q_w)
    if theta > np.pi:
        theta = theta - 2.0 * np.pi
    elif theta < - np.pi:
        theta = theta + 2.0 * np.pi
    return theta * normalize_vector(q_xyz)


def matrix_exponent(omega, p=1.0):
    """Returns exponent of given omega.

    This function is similar to cv2.Rodrigues.
    Convert rvec (which is log quaternion) to rotation matrix.

    Parameters
    ----------
    omega : list or numpy.ndarray
        vector of shape (3,)

    Returns
    -------
    rot : numpy.ndarray
        exponential matrix of given omega

    Examples
    --------
    >>> import numpy as np
    >>> from skrobot.coordinates.math import matrix_exponent
    >>> matrix_exponent([1, 1, 1])
    array([[ 0.22629564, -0.18300792,  0.95671228],
           [ 0.95671228,  0.22629564, -0.18300792],
           [-0.18300792,  0.95671228,  0.22629564]])
    >>> matrix_exponent([1, 0, 0])
    array([[ 1.        ,  0.        ,  0.        ],
           [ 0.        ,  0.54030231, -0.84147098],
           [ 0.        ,  0.84147098,  0.54030231]])
    """
    w = np.linalg.norm(omega)
    amat = outer_product_matrix(normalize_vector(omega))
    return np.eye(3) + np.sin(w * p) * amat + \
        (1.0 - np.cos(w * p)) * np.matmul(amat, amat)


def outer_product_matrix(v):
    """Returns outer product matrix of given v.

    Returns following outer product matrix.

    .. math::
        \\left(
            \\begin{array}{ccc}
              0 & -v_2 & v_1 \\\\
              v_2 & 0 & -v_0 \\\\
              -v_1 & v_0 & 0
            \\end{array}
        \\right)

    Parameters
    ----------
    v : numpy.ndarray or list
        [x, y, z]

    Returns
    -------
    matrix : numpy.ndarray
        3x3 rotation matrix.

    Examples
    --------
    >>> from skrobot.coordinates.math import outer_product_matrix
    >>> outer_product_matrix([1, 2, 3])
    array([[ 0, -3,  2],
           [ 3,  0, -1],
           [-2,  1,  0]])
    """
    return np.array([[0, -v[2], v[1]],
                     [v[2], 0, -v[0]],
                     [-v[1], v[0], 0]])


def cross_product(a, b):
    """Return cross product.

    Parameters
    ----------
    a : numpy.ndarray
        3-dimensional vector.
    b : numpy.ndarray
        3-dimensional vector.

    Returns
    -------
    cross_prod : numpy.ndarray
        calculated cross product
    """
    return np.dot(outer_product_matrix(a), b)


def quaternion2rpy(q):
    """Returns Roll-pitch-yaw angles of a given quaternion.

    Parameters
    ----------
    q : numpy.ndarray or list
        Quaternion in [w x y z] format.

    Returns
    -------
    rpy : numpy.ndarray
        Array of yaw-pitch-roll angles, in radian.

    Examples
    --------
    >>> from skrobot.coordinates.math import quaternion2rpy
    >>> quaternion2rpy([1, 0, 0, 0])
    (array([ 0., -0.,  0.]), array([3.14159265, 3.14159265, 3.14159265]))
    >>> quaternion2rpy([0, 1, 0, 0])
    (array([ 0.        , -0.        ,  3.14159265]),
     array([3.14159265, 3.14159265, 0.        ]))
    """
    q = to_numpy_array(q)
    if q.ndim == 1:
        roll = atan2(
            2 * q[2] * q[3] + 2 * q[0] * q[1],
            q[3] ** 2 - q[2] ** 2 - q[1] ** 2 + q[0] ** 2)
        pitch = -asin(
            2 * q[1] * q[3] - 2 * q[0] * q[2])
        yaw = atan2(
            2 * q[1] * q[2] + 2 * q[0] * q[3],
            q[1] ** 2 + q[0] ** 2 - q[3] ** 2 - q[2] ** 2)
        rpy = np.array([yaw, pitch, roll])
    elif q.ndim == 2:
        roll = np.arctan2(
            2 * q[:, 2] * q[:, 3] + 2 * q[:, 0] * q[:, 1],
            q[:, 3] ** 2 - q[:, 2] ** 2 - q[:, 1] ** 2 + q[:, 0] ** 2)
        pitch = -np.sin(
            2 * q[:, 1] * q[:, 3] - 2 * q[:, 0] * q[:, 2])
        yaw = np.arctan2(
            2 * q[:, 1] * q[:, 2] + 2 * q[:, 0] * q[:, 3],
            q[:, 1] ** 2 + q[:, 0] ** 2 - q[:, 3] ** 2 - q[:, 2] ** 2)
        rpy = np.concatenate([yaw[:, None], pitch[:, None], roll[:, None]],
                             axis=1)
    return rpy, np.pi - rpy


def rpy2quaternion(rpy):
    """Return Quaternion from yaw-pitch-roll angles.

    Parameters
    ----------
    rpy : numpy.ndarray or list
        Vector of yaw-pitch-roll angles in radian.

    Returns
    -------
    quat : numpy.ndarray
        Quaternion in [w x y z] format.

    Examples
    --------
    >>> import numpy as np
    >>> from skrobot.coordinates.math import rpy2quaternion
    >>> rpy2quaternion([0, 0, 0])
    array([1., 0., 0., 0.])
    >>> yaw = np.pi / 3.0
    >>> rpy2quaternion([yaw, 0, 0])
    array([0.8660254, 0.       , 0.       , 0.5      ])
    >>> rpy2quaternion([np.pi * 2 - yaw, 0, 0])
    array([-0.8660254, -0.       ,  0.       ,  0.5      ])
    """
    yaw, pitch, roll = rpy
    cr, cp, cy = cos(roll / 2.), cos(pitch / 2.), cos(yaw / 2.)
    sr, sp, sy = sin(roll / 2.), sin(pitch / 2.), sin(yaw / 2.)
    return np.array([
        cr * cp * cy + sr * sp * sy,
        -cr * sp * sy + cp * cy * sr,
        cr * cy * sp + sr * cp * sy,
        cr * cp * sy - sr * cy * sp])


def rotation_matrix_from_rpy(rpy):
    """Returns Rotation matrix from yaw-pitch-roll angles.

    Parameters
    ----------
    rpy : numpy.ndarray or list
        Vector of yaw-pitch-roll angles in radian.

    Returns
    -------
    rot : numpy.ndarray
        3x3 rotation matrix

    Examples
    --------
    >>> import numpy as np
    >>> from skrobot.coordinates.math import rotation_matrix_from_rpy
    >>> rotation_matrix_from_rpy([0, np.pi / 3, 0])
    array([[ 0.5      ,  0.       ,  0.8660254],
           [ 0.       ,  1.       ,  0.       ],
           [-0.8660254,  0.       ,  0.5      ]])
    """
    return quaternion2matrix(quat_from_rpy(rpy))


def rotation_matrix_from_axis(
        first_axis=(1, 0, 0), second_axis=(0, 1, 0), axes='xy'):
    """Return rotation matrix orienting first_axis

    Parameters
    ----------
    first_axis : list or tuple or numpy.ndarray
        direction of first axis
    second_axis : list or tuple or numpy.ndarray
        direction of second axis.
        This input axis is normalized using Gram-Schmidt.
    axes : str
        valid inputs are 'xy', 'yx', 'xz', 'zx', 'yz', 'zy'.
        first index indicates first_axis's axis.

    Returns
    -------
    rotation_matrix : numpy.ndarray
        Rotation matrix

    Examples
    --------
    >>> from skrobot.coordinates.math import rotation_matrix_from_axis
    >>> rotation_matrix_from_axis((1, 0, 0), (0, 1, 0))
    array([[1., 0., 0.],
           [0., 1., 0.],
           [0., 0., 1.]])
    >>> rotation_matrix_from_axis((1, 1, 1), (0, 1, 0))
    array([[ 0.57735027, -0.40824829, -0.70710678],
           [ 0.57735027,  0.81649658,  0.        ],
           [ 0.57735027, -0.40824829,  0.70710678]])
    """
    if axes not in ['xy', 'yx', 'xz', 'zx', 'yz', 'zy']:
        raise ValueError("Valid axes are 'xy', 'yx', 'xz', 'zx', 'yz', 'zy'.")
    e1 = normalize_vector(first_axis)
    if np.linalg.norm(e1, ord=2) == 0:
        raise ValueError(
            'Invalid first_axis value. '
            'Norm of axis should be greater than 0.0. Get first_axis: ({})'.
            format(first_axis))
    e2 = normalize_vector(second_axis - np.dot(second_axis, e1) * e1)
    if np.linalg.norm(e2, ord=2) == 0:
        raise ValueError(
            'Invalid second_axis({}) with respect to first_axis({}).'
            .format(first_axis, second_axis))
    if axes in ['xy', 'zx', 'yz']:
        third_axis = cross_product(e1, e2)
    else:
        third_axis = cross_product(e2, e1)
    e3 = normalize_vector(
        third_axis - np.dot(third_axis, e1) * e1 - np.dot(third_axis, e2) * e2)
    first_index = ord(axes[0]) - ord('x')
    second_index = ord(axes[1]) - ord('x')
    third_index = ((first_index + 1) ^ (second_index + 1)) - 1
    indices = [first_index, second_index, third_index]
    return np.vstack([e1, e2, e3])[np.argsort(indices)].T


def rodrigues(axis, theta=None, skip_normalization=False):
    """Rodrigues formula.

    See: `Rodrigues' rotation formula - Wikipedia
    <https://en.wikipedia.org/wiki/Rodrigues%27_rotation_formula>`_.

    See: `Axis-angle representation - Wikipedia
    <https://en.wikipedia.org/wiki/Axis%E2%80%93angle_representation>`_.

    Parameters
    ----------
    axis : numpy.ndarray or list
        If single axis, it should be [x, y, z] vector.
        If multiple axes, it should be an Nx3 matrix where each row is an axis.
        You can give axis-angle representation to `axis` if `theta` is None.
    theta: float, numpy.ndarray, or None (optional)
        If single theta, it is a float. For multiple rotations,
        it should be an Nx1 vector.
        If None is given, calculate theta from the norm of each axis.

    Returns
    -------
    mat : numpy.ndarray
        Rotation matrix or matrices. 3x3 if single axis,
        Nx3x3 if multiple axes.

    Examples
    --------
    >>> import numpy
    >>> from skrobot.coordinates.math import rodrigues
    >>> rodrigues([1, 0, 0], 0)
    array([[1., 0., 0.],
           [0., 1., 0.],
           [0., 0., 1.]])
    >>> rodrigues([[1, 1, 1], [0, 1, 0]], [np.pi, np.pi/2])
    array([[[ -0.33333333,  0.66666667,  0.66666667],
            [  0.66666667, -0.33333333,  0.66666667],
            [  0.66666667,  0.66666667, -0.33333333]],
           [[  0. ,  0. ,  1. ],
            [  0. ,  1. ,  0. ],
            [ -1. ,  0. ,  0. ]]])
    """
    axis = np.asarray(axis, dtype=np.float64)
    if axis.ndim == 1:
        axis = axis[np.newaxis, :]  # Convert to 1x3 for single axis

    if theta is None:
        theta = np.linalg.norm(axis, axis=1)
    elif np.isscalar(theta):
        theta = np.full(axis.shape[0], theta, dtype=np.float64)

    # Normalize the axes
    norm = axis / np.linalg.norm(axis, axis=1)[:, np.newaxis]

    # Cross-product matrix construction
    zeros = np.zeros(norm.shape[0])
    cross_prod = np.array([
        [zeros, -norm[:, 2], norm[:, 1]],
        [norm[:, 2], zeros, -norm[:, 0]],
        [-norm[:, 1], norm[:, 0], zeros]
    ]).transpose((2, 0, 1))  # Shape Nx3x3

    # Compute cosines and sines
    ctheta = np.cos(theta)
    stheta = np.sin(theta)

    # Eye matrix and cross-product terms
    eye_matrix = np.eye(3).reshape((1, 3, 3))
    cross_prod_stheta = cross_prod * stheta[:, np.newaxis, np.newaxis]
    cross_prod_squared = np.matmul(cross_prod, cross_prod)
    cross_prod_squared_ctheta = cross_prod_squared * (1 - ctheta)[
        :, np.newaxis, np.newaxis]

    # Sum the components
    mat = eye_matrix + cross_prod_stheta + cross_prod_squared_ctheta
    return mat.squeeze()  # Remove single-dimensional entries from the shape


def rotation_angle(mat, return_angular_velocity=False):
    """Inverse Rodrigues formula Convert Rotation-Matrix to Axis-Angle.

    Return theta and axis.
    If given unit matrix, return None.

    Parameters
    ----------
    mat : numpy.ndarray
        rotation matrix, shape (3, 3)
    return_angular_velocity : bool, optional
        If True, returns the angular velocity vector
        instead of the axis-angle representation.

    Returns
    -------
    theta, axis or angular_velocity : tuple(float, numpy.ndarray)
    or numpy.ndarray
        Rotation angle in radian and rotation axis,
        or angular velocity vector if `return_angular_velocity` is True.

    Examples
    --------
    >>> import numpy
    >>> from skrobot.coordinates.math import rotation_angle
    >>> rotation_angle(numpy.eye(3)) is None
    True
    >>> rotation_angle(numpy.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]]))
    (1.5707963267948966, array([0., 1., 0.]))
    """
    mat = _check_valid_rotation(mat)
    if np.array_equal(mat, np.eye(3)):
        if return_angular_velocity:
            return None
        else:
            return None, None
    theta = np.arccos((np.trace(mat) - 1) / 2)
    if abs(theta) < _EPS:
        raise ValueError('Rotation Angle is too small. \nvalue : {}'.
                         format(theta))
    axis = 1.0 / (2 * np.sin(theta)) \
        * np.array([mat[2, 1] - mat[1, 2],
                    mat[0, 2] - mat[2, 0],
                    mat[1, 0] - mat[0, 1]])

    if return_angular_velocity:
        angular_velocity = axis * theta
        return angular_velocity

    return theta, axis


def rotation_distance(mat1, mat2):
    """Return the distance of rotation matrixes.

    Parameters
    ----------
    mat1 : list or numpy.ndarray
    mat2 : list or numpy.ndarray
        3x3 matrix

    Returns
    -------
    diff_theta : float
        distance of rotation matrixes in radian.

    Examples
    --------
    >>> import numpy
    >>> from skrobot.coordinates.math import rotation_distance
    >>> rotation_distance(numpy.eye(3), numpy.eye(3))
    0.0
    >>> rotation_distance(
            numpy.eye(3),
            numpy.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]]))
    1.5707963267948966
    """
    mat1 = _check_valid_rotation(mat1)
    mat2 = _check_valid_rotation(mat2)
    q1 = matrix2quaternion(mat1)
    q2 = matrix2quaternion(mat2)
    diff_theta = quaternion_distance(q1, q2)
    return diff_theta


def quaternion_multiply(quaternion1, quaternion0):
    """Return multiplication of two quaternions.

    Parameters
    ----------
    quaternion0 : list or numpy.ndarray
        [w, x, y, z]
    quaternion1 : list or numpy.ndarray
        [w, x, y, z]

    Returns
    -------
    quaternion : numpy.ndarray
        [w, x, y, z]

    Examples
    --------
    >>> q = quaternion_multiply([4, 1, -2, 3], [8, -5, 6, 7])
    >>> numpy.allclose(q, [28, -44, -14, 48])
    True
    """
    quaternion1 = np.array(quaternion1)
    if quaternion1.ndim == 1:
        w0, x0, y0, z0 = quaternion0
        w1, x1, y1, z1 = quaternion1
        return np.array((
            -x1 * x0 - y1 * y0 - z1 * z0 + w1 * w0,
            x1 * w0 + y1 * z0 - z1 * y0 + w1 * x0,
            -x1 * z0 + y1 * w0 + z1 * x0 + w1 * y0,
            x1 * y0 - y1 * x0 + z1 * w0 + w1 * z0), dtype=np.float64)
    elif quaternion1.ndim == 2:
        w0, x0, y0, z0 = np.split(quaternion0, 4, 1)
        w1, x1, y1, z1 = np.split(quaternion1, 4, 1)
        result = np.array((
            -x1 * x0 - y1 * y0 - z1 * z0 + w1 * w0,
            x1 * w0 + y1 * z0 - z1 * y0 + w1 * x0,
            -x1 * z0 + y1 * w0 + z1 * x0 + w1 * y0,
            x1 * y0 - y1 * x0 + z1 * w0 + w1 * z0), dtype=np.float64)
        return np.transpose(np.squeeze(result)).reshape(
            quaternion1.shape[0], 4)
    else:
        raise ValueError


def quaternion_conjugate(quaternion):
    """Return conjugate of quaternion.

    Parameters
    ----------
    quaternion : list or numpy.ndarray
        quaternion [w, x, y, z]

    Returns
    -------
    conjugate of quaternion : numpy.ndarray
        [w, x, y, z]

    Examples
    --------
    >>> q0 = random_quaternion()
    >>> q1 = quaternion_conjugate(q0)
    >>> np.allclose(quaternion_multiply(q0, q1), [1.0, 0, 0, 0])
    True
    """
    quaternion = np.array(quaternion, dtype=np.float64)
    if quaternion.ndim == 1:
        return np.array((quaternion[0], -quaternion[1],
                         -quaternion[2], -quaternion[3]),
                        dtype=np.float64)
    elif quaternion.ndim == 2:
        return quaternion * np.array([[1, -1, -1, -1]])
    else:
        raise ValueError


def quaternion_inverse(quaternion):
    """Return inverse of quaternion.

    Parameters
    ----------
    quaternion : list or numpy.ndarray
        [w, x, y, z]

    Returns
    -------
    inverse of quaternion : numpy.ndarray
        [w, x, y, z]

    Examples
    --------
    >>> q0 = random_quaternion()
    >>> q1 = quaternion_inverse(q0)
    >>> np.allclose(quaternion_multiply(q0, q1), [1, 0, 0, 0])
    True
    """
    q = np.array(quaternion, dtype=np.float64)
    return quaternion_conjugate(q) / np.sum(
        q * q, axis=q.ndim - 1, keepdims=True)


def quaternion_slerp(q0, q1, fraction, spin=0, shortestpath=True):
    """Return spherical linear interpolation between two quaternions.

    Parameters
    ----------
    q0 : list or numpy.ndarray
        start quaternion
    q1 : list or numpy.ndarray
        end quaternion
    fraction : float
        ratio
    spin : int
        TODO
    shortestpath : bool
        TODO

    Returns
    -------
    quaternion : numpy.ndarray
        spherical linear interpolated quaternion

    Examples
    --------
    >>> q0 = random_quaternion()
    >>> q1 = random_quaternion()
    >>> q = quaternion_slerp(q0, q1, 0.0)
    >>> numpy.allclose(q, q0)
    True
    >>> q = quaternion_slerp(q0, q1, 1.0, 1)
    >>> numpy.allclose(q, q1)
    True
    >>> q = quaternion_slerp(q0, q1, 0.5)
    >>> angle = math.acos(numpy.dot(q0, q))
    >>> numpy.allclose(2.0, math.acos(numpy.dot(q0, q1)) / angle) or \
        numpy.allclose(2.0, math.acos(-numpy.dot(q0, q1)) / angle)
    True
    """
    q0 = normalize_vector(q0)
    q1 = normalize_vector(q1)
    if fraction == 0.0:
        return q0
    elif fraction == 1.0:
        return q1
    d = np.dot(q0, q1)
    if abs(abs(d) - 1.0) < 0.0:
        return q0
    if shortestpath and d < 0.0:
        # invert rotation
        d = -d
        q1 *= -1.0
    theta = acos(d)
    angle = theta + spin * pi
    if abs(angle) < _EPS:
        return q0
    isin = 1.0 / sin(angle)
    q = (sin((1.0 - fraction) * angle) * q0
         + sin(fraction * angle) * q1) * isin
    return q


def quaternion_distance(q1, q2, absolute=False):
    """Return the distance of quaternion.

    Parameters
    ----------
    q1 : list or numpy.ndarray
    q2 : list or numpy.ndarray
        [w, x, y, z] order
    absolute : bool
        if True, return distance accounting for the sign ambiguity.

    Returns
    -------
    diff_theta : float
        distance of q1 and q2 in radian.

    Examples
    --------
    >>> from skrobot.coordinates.math import quaternion_distance
    >>> quaternion_distance([1, 0, 0, 0], [1, 0, 0, 0])
    0.0
    >>> quaternion_distance([1, 0, 0, 0], [0, 1, 0, 0])
    3.141592653589793
    >>> distance = quaternion_distance(
            [1, 0, 0, 0],
            [0.8660254, 0, 0.5, 0])
    >>> np.rad2deg(distance)
    60.00000021683236
    """
    q1 = to_numpy_array(q1)
    q2 = to_numpy_array(q2)
    dot = np.clip(np.sum(q1 * q2, q1.ndim - 1), -1, 1)
    diff_theta = 2 * np.arccos(dot)
    if absolute is True:
        diff_theta = np.abs(diff_theta)
    return diff_theta


def quaternion_absolute_distance(q1, q2):
    """Return the absolute distance of quaternion.

    Parameters
    ----------
    q1 : list or numpy.ndarray
    q2 : list or numpy.ndarray
        [w, x, y, z] order

    Returns
    -------
    diff_theta : float
        absolute distance of q1 and q2 in radian.

    Examples
    --------
    >>> from skrobot.coordinates.math import quaternion_absolute_distance
    >>> quaternion_absolute_distance([1, 0, 0, 0], [1, 0, 0, 0])
    0.0
    >>> quaternion_absolute_distance(
            [1, 0, 0, 0],
            [0, 0.7071067811865476, 0, 0.7071067811865476])
    3.141592653589793
    >>> quaternion_absolute_distance(
            [-1, 0, 0, 0],
            [0, 0.7071067811865476, 0, 0.7071067811865476])
    """
    return quaternion_distance(q1, q2, True)


def quaternion_norm(q):
    """Return the norm of quaternion.

    Parameters
    ----------
    q : list or numpy.ndarray
        [w, x, y, z] order

    Returns
    -------
    norm_q : float
        quaternion norm of q

    Examples
    --------
    >>> from skrobot.coordinates.math import quaternion_norm
    >>> q = [1, 1, 1, 1]
    >>> quaternion_norm(q)
    2.0
    >>> q = [0, 0.7071067811865476, 0, 0.7071067811865476]
    >>> quaternion_norm(q)
    1.0
    """
    q = np.array(q)
    if q.ndim == 1:
        norm_q = np.sqrt(np.dot(q.T, q))
    elif q.ndim == 2:
        norm_q = np.sqrt(np.sum(q * q, axis=1, keepdims=True))
    else:
        raise ValueError
    return norm_q


def quaternion_normalize(q):
    """Return the normalized quaternion.

    Parameters
    ----------
    q : list or numpy.ndarray
        [w, x, y, z] order

    Returns
    -------
    normalized_q : numpy.ndarray
        normalized quaternion

    Examples
    --------
    >>> from skrobot.coordinates.math import quaternion_normalize
    >>> from skrobot.coordinates.math import quaternion_norm
    >>> q = quaternion_normalize([1, 1, 1, 1])
    >>> quaternion_norm(q)
    1.0
    """
    q = np.array(q)
    normalized_q = q / quaternion_norm(q)
    return normalized_q


def quaternion_from_axis_angle(theta, axis):
    """Return the quaternion from axis angle

    This function returns quaternion associated with counterclockwise
    rotation about the given axis by theta radians.

    Parameters
    ----------
    theta : float
        radian
    axis : list or numpy.ndarray
        length is 3. Automatically normalize in this function

    Returns
    -------
    quaternion : numpy.ndarray
        [w, x, y, z] order

    Examples
    --------
    >>> import numpy
    >>> from skrobot.coordinates.math import quaternion_from_axis_angle
    >>> quaternion_from_axis_angle(0.1, [1, 0, 0])
    array([0.99875026, 0.04997917, 0.        , 0.        ])
    >>> quaternion_from_axis_angle(numpy.pi, [1, 0, 0])
    array([6.123234e-17, 1.000000e+00, 0.000000e+00, 0.000000e+00])
    >>> quaternion_from_axis_angle(0, [1, 0, 0])
    array([1., 0., 0., 0.])
    >>> quaternion_from_axis_angle(numpy.pi, [1, 0, 1])
    array([6.12323400e-17, 7.07106781e-01, 0.00000000e+00, 7.07106781e-01])
    """
    axis = normalize_vector(axis)
    s = sin(theta / 2)
    x = axis[0] * s
    y = axis[1] * s
    z = axis[2] * s
    w = cos(theta / 2)
    return np.array([w, x, y, z], dtype=np.float64)


def rotation_vector_to_quaternion(rvec):
    """Convert rotation vector to quaternion.

    Parameters
    ----------
    rvec : list or numpy.ndarray
        vector of shape (3,)

    Returns
    -------
    q : numpy.ndarray
        quaternion
    """
    rvec = np.array(rvec).reshape(-1)
    theta = np.linalg.norm(rvec)
    if theta == 0:
        return np.array([1, 0, 0, 0], dtype=np.float64)
    axis = rvec / theta
    return quaternion_from_axis_angle(theta, axis)


def axis_angle_from_quaternion(quat):
    """Converts a quaternion into the axis-angle representation.

    Parameters
    ----------
    quat : numpy.ndarray
        quaternion [w, x, y, z]

    Returns
    -------
    axis_angle : numpy.ndarray
        axis-angle representation of vector

    Examples
    --------
    >>> from skrobot.coordinates.math import axis_angle_from_quaternion
    >>> axis_angle_from_quaternion([1, 0, 0, 0])
    array([0, 0, 0])
    >>> axis_angle_from_quaternion([0, 7.07106781e-01, 0, 7.07106781e-01])
    array([2.22144147, 0.        , 2.22144147])
    """
    quat = np.array(quat, dtype=np.float64)
    x, y, z, w = quat
    sinang = y ** 2 + z ** 2 + w ** 2
    if sinang == 0:
        return np.array([0, 0, 0])
    if x < 0:
        _quat = - quat
    else:
        _quat = quat
    sinang = np.sqrt(sinang)
    f = 2.0 * np.arctan2(sinang, _quat[0]) / sinang
    return f * np.array([_quat[1], _quat[2], _quat[3]])


def axis_angle_from_matrix(rotation):
    """Converts the rotation of a matrix into axis-angle representation.

    Parameters
    ----------
    rotation : numpy.ndarray
        3x3 rotation matrix

    Returns
    -------
    axis_angle : numpy.ndarray
        axis-angle representation of vector

    Examples
    --------
    >>> import numpy
    >>> from skrobot.coordinates.math import axis_angle_from_matrix
    >>> axis_angle_from_matrix(numpy.eye(3))
    array([0, 0, 0])
    >>> axis_angle_from_matrix(
        numpy.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]]))
    array([0.        , 1.57079633, 0.        ])
    """
    return axis_angle_from_quaternion(quat_from_rotation_matrix(rotation))


def angle_between_vectors(v1, v2, normalize=True,
                          directed=True):
    """Returns the smallest angle in radians between two vectors.

    Parameters
    ----------
    v1 : numpy.ndarray, list[float] or tuple(float)
        input vector.
    v2 : numpy.ndarray, list[float] or tuple(float)
        input vector.
    normalize : bool
        If normalize is True, normalize v1 and v2.
    directed : bool
        If directed is `False`, the input vectors are
        interpreted as undirected axes.

    Returns
    -------
    theta : float
        smallest angle between v1 and v2.
    """
    if normalize:
        v1 = normalize_vector(v1)
        v2 = normalize_vector(v2)
    dot = np.dot(v1, v2)
    return np.arccos(np.clip(dot if directed else np.fabs(dot), -1.0, 1.0))


def counter_clockwise_angle_between_vectors(v1, v2, normal_vector=None):
    """Returns the counter clockwise angle in radians between two vectors.

    Parameters
    ----------
    v1 : numpy.ndarray, list[float] or tuple(float)
        input vector.
    v2 : numpy.ndarray, list[float] or tuple(float)
        input vector.
    normal_vector : numpy.ndarray, list[float] or tuple(float) or None
        Base plane's normal vector.

    Returns
    -------
    theta : float
        counter clockwise angle between v1 and v2.
        Return values in [0 radian, 2 * np.pi radian].
    """
    if normal_vector is None:
        normal_vector = normal_vector(np.cross(v1, v2))
    # https://stackoverflow.com/questions/14066933/direct-way-of-computing-clockwise-angle-between-2-vectors  # NOQA
    det = triple_product(normal_vector, v1, v2)
    dot = np.dot(v1, v2)
    angle = np.arctan2(-det, -dot) + np.pi
    if np.isclose(angle, 2 * np.pi):
        angle = 0.0
    return angle


def clockwise_angle_between_vectors(v1, v2, normal_vector=None):
    """Returns the clockwise angle in radians between two vectors.

    Parameters
    ----------
    v1 : numpy.ndarray, list[float] or tuple(float)
        input vector.
    v2 : numpy.ndarray, list[float] or tuple(float)
        input vector.
    normal_vector : numpy.ndarray, list[float] or tuple(float) or None
        Base plane's normal vector.

    Returns
    -------
    theta : float
        clockwise angle between v1 and v2.
        Return values in [0 radian, 2 * np.pi radian].
    """
    if normal_vector is None:
        normal_vector = normal_vector(np.cross(v1, v2))
    normal_vector = - np.array(normal_vector)
    det = triple_product(normal_vector, v1, v2)
    dot = np.dot(v1, v2)
    angle = np.arctan2(-det, -dot) + np.pi
    if np.isclose(angle, 2 * np.pi):
        angle = 0.0
    return angle


def is_parallel_two_vectors(v1, v2):
    """Return if Two Vectors Are Parallel or not.

    Parameters
    ----------
    v1 : numpy.ndarray, list[float] or tuple(float)
        input vector.
    v2 : numpy.ndarray, list[float] or tuple(float)
        input vector.

    Returns
    -------
    parallel : bool
        parallel or not.
    """
    return np.all(np.cross(v1, v2) == 0.0)


def random_rotation():
    """Generates a random 3x3 rotation matrix.

    Returns
    -------
    rot : numpy.ndarray
        randomly generated 3x3 rotation matrix

    Examples
    --------
    >>> from skrobot.coordinates.math import random_rotation
    >>> random_rotation()
    array([[-0.00933428, -0.90465681, -0.42603865],
          [-0.50305571, -0.36396787,  0.78387648],
          [-0.86420358,  0.2216381 , -0.4516954 ]])
    >>> random_rotation()
    array([[-0.6549113 ,  0.09499001, -0.749712  ],
          [-0.47962794, -0.81889635,  0.31522342],
          [-0.58399334,  0.5660262 ,  0.58186434]])
    """
    rot = quaternion2matrix(random_quaternion())
    return rot


def random_translation():
    """Generates a random translation vector.

    Returns
    -------
    translation : numpy.ndarray
        A 3-entry random translation vector.

    Examples
    --------
    >>> from skrobot.coordinates.math import random_translation
    >>> random_translation()
    array([0.03299473, 0.81481471, 0.57782565])
    >>> random_translation()
    array([0.10835455, 0.46549158, 0.73277675])
    """
    return np.random.rand(3)


def random_quaternion():
    """Generate uniform random unit quaternion.

    Returns
    -------
    quaternion : numpy.ndarray
        generated random unit quaternion [w, x, y, z]

    Examples
    --------
    >>> from skrobot.coordinates.math import random_quaternion
    >>> random_quaternion()
    array([-0.02156994,  0.5404561 , -0.72781116, -0.42158374])
    >>> random_quaternion()
    array([-0.47302116,  0.020306  , -0.37539238,  0.79681818])
    >>> from skrobot.coordinates.math import quaternion_norm
    >>> q = random_quaternion()
    >>> numpy.allclose(1.0, quaternion_norm(q))
    True
    >>> q.shape
    (4,)
    """
    rand = np.random.rand(3)
    r1 = np.sqrt(1.0 - rand[0])
    r2 = np.sqrt(rand[0])
    pi2 = np.pi * 2.0
    t1 = pi2 * rand[1]
    t2 = pi2 * rand[2]
    return np.array((cos(t2) * r2,
                     sin(t1) * r1,
                     cos(t1) * r1,
                     sin(t2) * r2),
                    dtype=np.float64)


def make_matrix(r, c):
    """Wrapper of numpy array.

    Parameters
    ----------
    r : int
        row of matrix
    c : int
        column of matrix

    Returns
    -------
    np.zeros((r, c), 'f') : numpy.ndarray
        matrix
    """
    return np.zeros((r, c), 'f')


inverse_rodrigues = rotation_angle
quat_from_rotation_matrix = matrix2quaternion
quat_from_rpy = rpy2quaternion
rotation_matrix_from_quat = quaternion2matrix
rpy_from_quat = quaternion2rpy
rvec_to_quaternion = rotation_vector_to_quaternion

# clockwise angle
ccw_angle_between_vectors = counter_clockwise_angle_between_vectors
cw_angle_between_vectors = clockwise_angle_between_vectors

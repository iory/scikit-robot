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
    ret = np.mamul(weight_J, np.linalg.inv(umat))
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
    rpy = np.array([r, p, y])
    return rpy, np.pi - rpy


def normalize_vector(v, ord=2):
    return v / np.linalg.norm(v, ord=ord)

from logging import getLogger

from sympy import Abs
from sympy import Float
from sympy import Matrix
from sympy import Rational
from sympy import S
from sympy import eye
from sympy import pi
from sympy import sqrt
from sympy import trigsimp

import numpy as np

from robot.math import axis_angle_from_matrix

logger = getLogger(__name__)


def convert_real_to_rational(x, precision=8):
    if Abs(x) < 10 ** -precision:
        return S.Zero
    r0 = Rational(str(round(Float(float(x), 30), precision)))
    if x == 0:
        return r0
    r1 = 1 / Rational(str(round(Float(1 / float(x), 30), precision)))
    return r0 if len(str(r0)) < len(str(r1)) else r1


def get_matrix_from_quaternion(quat):
    """
    Get Rotation Matirx from quaternion

    Parameters
    ----------
    quat :
        quaternion is [cos(angle/2), v*sin(angle/2)]

    Returns
    -------
    rotation_matrix : np.ndarray
        4x4 matrix
    """
    M = eye(4)
    qq1 = 2 * quat[1] * quat[1]
    qq2 = 2 * quat[2] * quat[2]
    qq3 = 2 * quat[3] * quat[3]
    M[0, 0] = 1 - qq2 - qq3
    M[0, 1] = 2 * (quat[1] * quat[2] - quat[0] * quat[3])
    M[0, 2] = 2 * (quat[1] * quat[3] + quat[0] * quat[2])
    M[1, 0] = 2 * (quat[1] * quat[2] + quat[0] * quat[3])
    M[1, 1] = 1 - qq1 - qq3
    M[1, 2] = 2 * (quat[2] * quat[3] - quat[0] * quat[1])
    M[2, 0] = 2 * (quat[1] * quat[3] - quat[0] * quat[2])
    M[2, 1] = 2 * (quat[2] * quat[3] + quat[0] * quat[1])
    M[2, 2] = 1 - qq1 - qq2
    return M


def normalize_rotation(M, precision=8):
    right = Matrix(3, 1, [convert_real_to_rational(
        x, precision - 3) for x in M[0, 0:3]])
    right = right / right.norm()
    up = Matrix(3, 1, [convert_real_to_rational(x, precision - 3)
                       for x in M[1, 0:3]])
    up = up - right * right.dot(up)
    up = up / up.norm()
    d = right.cross(up)
    for i in range(3):
        # don't round the rotational part anymore
        # since it could lead to unnormalized rotations!
        M[0, i] = right[i]
        M[1, i] = up[i]
        M[2, i] = d[i]
        M[i, 3] = convert_real_to_rational(M[i, 3])
        M[3, i] = S.Zero
    M[3, 3] = S.One
    return M


def rodrigues2(axis, cosangle, sinangle):
    skewsymmetric = Matrix(3, 3,
                           [S.Zero, -axis[2], axis[1],
                            axis[2], S.Zero, -axis[0],
                            -axis[1], axis[0], S.Zero])
    return eye(3) + sinangle * skewsymmetric + (S.One - cosangle) * \
        skewsymmetric * skewsymmetric


def get_matrix_from_numpy(T):
    return Matrix(4, 4, [x for x in T.flat])


def affine_inverse(affine_matrix):
    T = eye(4)
    T[0:3, 0:3] = affine_matrix[0:3, 0:3].transpose()
    T[0:3, 3] = -affine_matrix[0:3, 0:3].transpose() * affine_matrix[0:3, 3]
    return T


def affine_simplify(T):
    return Matrix(T.shape[0], T.shape[1], [trigsimp(x.expand()) for x in T])


def numpy_vector_to_sympy(v, precision=8):
    return Matrix(len(v), 1, [convert_real_to_rational(x, precision) for x in v])


def multiply_matrices(Ts):
    Tfinal = eye(4)
    for T in Ts:
        Tfinal = Tfinal * T
    return Tfinal


def round_matrix(T, precision=8):
    """
    given a sympy matrix, will round the matrix and snap all its
    values to 15, 30, 45, 60, and 90 degrees.

    Parameters
    ----------
    T : np.ndarray or list or sympy.Matirx
        4x4 matrix
    precision : int
        precision of values

    Returns
    -------
    TODO
    """
    if isinstance(T, list):
        T = np.array(T, dtype=np.float64)
    if isinstance(T, np.ndarray):
        T = Matrix(T)
    if T.shape != (4, 4):
        raise ValueError('input matrix shape should be (4, 4) '
                         ', we given {}'.format(T.shape))
    Teval = T.evalf()
    axisangle = axis_angle_from_matrix(
        [[Teval[0, 0], Teval[0, 1], Teval[0, 2]],
         [Teval[1, 0], Teval[1, 1], Teval[1, 2]],
         [Teval[2, 0], Teval[2, 1], Teval[2, 2]]])
    angle = sqrt(axisangle[0]**2 + axisangle[1]**2 + axisangle[2]**2)
    if abs(angle) < 10**(- precision):
        # rotation is identity
        M = eye(4)
    else:
        axisangle = axisangle / angle
        logger.debug('rotation angle: %f, axis=[%f,%f,%f]' %
                     (angle * 180 / pi).evalf(),
                     axisangle[0], axisangle[1], axisangle[2])
        accurate_axis_angle = Matrix(
            3, 1, [convert_real_to_rational(x, precision - 3)
                   for x in axisangle])
        accurate_axis_angle = accurate_axis_angle / accurate_axis_angle.norm()
        # angle is not a multiple of 90, can get long fractions.
        # so check if there's any way to simplify it
        if abs(angle - 3 * pi / 2) < 10**(-precision + 2):
            quat = [-S.One / sqrt(2),
                    accurate_axis_angle[0] / sqrt(2),
                    accurate_axis_angle[1] / sqrt(2),
                    accurate_axis_angle[2] / sqrt(2)]
        elif abs(angle - pi) < 10**(-precision + 2):
            quat = [S.Zero, accurate_axis_angle[0],
                    accurate_axis_angle[1], accurate_axis_angle[2]]
        elif abs(angle - 2 * pi / 3) < 10**(-precision + 2):
            quat = [Rational(1, 2),
                    accurate_axis_angle[0] * sqrt(3) / 2,
                    accurate_axis_angle[1] * sqrt(3) / 2,
                    accurate_axis_angle[2] * sqrt(3) / 2]
        elif abs(angle - pi / 2) < 10**(-precision + 2):
            quat = [S.One / sqrt(2),
                    accurate_axis_angle[0] / sqrt(2),
                    accurate_axis_angle[1] / sqrt(2),
                    accurate_axis_angle[2] / sqrt(2)]
        elif abs(angle - pi / 3) < 10**(-precision + 2):
            quat = [sqrt(3) / 2,
                    accurate_axis_angle[0] / 2,
                    accurate_axis_angle[1] / 2,
                    accurate_axis_angle[2] / 2]
        elif abs(angle - pi / 4) < 10**(-precision + 2):
            quat = [sqrt(sqrt(2) + 2) / 2,
                    sqrt(-sqrt(2) + 2) / 2 * accurate_axis_angle[0],
                    sqrt(-sqrt(2) + 2) / 2 * accurate_axis_angle[1],
                    sqrt(-sqrt(2) + 2) / 2 * accurate_axis_angle[2]]
        elif abs(angle - pi / 6) < 10**(-precision + 2):
            quat = [sqrt(2) / 4 + sqrt(6) / 4,
                    (-sqrt(2) / 4 + sqrt(6) / 4) * accurate_axis_angle[0],
                    (-sqrt(2) / 4 + sqrt(6) / 4) * accurate_axis_angle[1],
                    (-sqrt(2) / 4 + sqrt(6) / 4) * accurate_axis_angle[2]]
        else:
            return normalize_rotation(T)

        M = get_matrix_from_quaternion(quat)
    for i in range(3):
        M[i, 3] = convert_real_to_rational(T[i, 3], precision)
    return M
